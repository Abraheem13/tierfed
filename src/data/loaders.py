"""Concrete dataset loaders.

synthetic  : controllable covariate/concept shift; used by CI and for stress tests.
diabetes   : UCI Diabetes 130-US Hospitals, Strack et al. (2014) cleaning
             INCLUDING one encounter per patient, patient-grouped splits, and a
             share-capped specialty partition. (v2 skipped the dedup because the
             UCI API returns patient_nbr in `ids`, not `features`.)
eicu       : eICU-CRD with the TRUE hospitalid (credentialed PhysioNet access).
eicu_demo  : the open eICU-CRD demo subset (PhysioNet, ODbL), TRUE hospitalid.
heart      : UCI Heart Disease, four real institutions (Cleveland, Hungarian
             Institute of Cardiology, University Hospitals Zurich/Basel, VA Long
             Beach) as in FLamby Fed-Heart-Disease (du Terrail et al., 2022).
mimic      : MIMIC-IV admissions cohort, sites = admission location.
medmnist   : MedMNIST v2; default PneumoniaMNIST, which is natively binary
             (v2 binarised the 7-class DermaMNIST into a 97%-positive task).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .base import FederatedDataset, register
from .partition import build_partition

DATA_ROOT = Path(os.environ.get("NFL_DATA_ROOT", "data"))


def _encode(df: pd.DataFrame, numeric, categorical, fit_idx=None):
    fit_idx = np.arange(len(df)) if fit_idx is None else fit_idx
    blocks = []
    if numeric:
        sc = StandardScaler().fit(df.loc[fit_idx, numeric].astype(float).values)
        blocks.append(sc.transform(df[numeric].astype(float).values))
    if categorical:
        try:
            oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            oh = OneHotEncoder(handle_unknown="ignore", sparse=False)
        oh.fit(df.loc[fit_idx, categorical].astype(str).values)
        blocks.append(oh.transform(df[categorical].astype(str).values))
    return np.hstack(blocks).astype(np.float32)


def _split(y, groups, seed, test_size=0.2):
    """Patient-grouped train/test split (falls back to stratified rows)."""
    if groups is None:
        return train_test_split(np.arange(len(y)), test_size=test_size, stratify=y, random_state=seed)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(np.zeros(len(y)), y, groups))
    return tr, te


def _split_and_partition(x, y, name, meta, n_clients, partition, seed, site_ids=None,
                         groups=None, patient_groups=None, alpha=0.5, test_size=0.2,
                         min_size=50, max_share=0.15):
    idx_tr, idx_te = _split(y, patient_groups, seed, test_size)
    pg_tr = None if patient_groups is None else np.asarray(patient_groups)[idx_tr]
    parts, rep = build_partition(partition, y[idx_tr], n_clients, seed=seed,
                                 site_ids=None if site_ids is None else np.asarray(site_ids)[idx_tr],
                                 groups=None if groups is None else np.asarray(groups)[idx_tr],
                                 patient_groups=pg_tr, alpha=alpha, min_size=min_size, max_share=max_share)
    meta = {**meta, "partition_report": rep, "partition": partition, "seed": seed,
            "patient_grouped_split": patient_groups is not None}
    return FederatedDataset(name, x[idx_tr], y[idx_tr], x[idx_te], y[idx_te], parts, meta, groups=pg_tr)


# --------------------------------------------------------------------------- #
@register("synthetic")
def load_synthetic(n_clients: int = 20, n_per_client: int = 400, n_features: int = 40,
                   prevalence: float = 0.11, covariate_shift: float = 1.0,
                   concept_shift: float = 0.3, seed: int = 0, partition: str = "natural",
                   alpha: float = 0.5, **_) -> FederatedDataset:
    rng = np.random.default_rng(seed)
    w0 = rng.normal(size=n_features)
    xs, ys, sites = [], [], []
    for k in range(n_clients):
        shift = rng.normal(scale=covariate_shift, size=n_features)
        wk = w0 + concept_shift * rng.normal(size=n_features)
        nk = int(n_per_client * rng.uniform(0.4, 2.0))
        x = rng.normal(size=(nk, n_features)) + shift
        logit = x @ wk / np.sqrt(n_features)
        thr = np.quantile(logit, 1 - prevalence)
        y = (logit + rng.normal(scale=0.8, size=nk) > thr).astype(int)
        xs.append(x); ys.append(y); sites.append(np.full(nk, k))
    x = np.vstack(xs).astype(np.float32); y = np.concatenate(ys); site = np.concatenate(sites)
    return _split_and_partition(x, y, "synthetic",
                                {"covariate_shift": covariate_shift, "concept_shift": concept_shift},
                                n_clients, partition, seed, site_ids=site, alpha=alpha, min_size=10)


# --------------------------------------------------------------------------- #
DIAB_NUMERIC = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
                "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
DIAB_CATEGORICAL = ["race", "gender", "age", "admission_type_id", "discharge_disposition_id",
                    "admission_source_id", "medical_specialty", "max_glu_serum", "A1Cresult",
                    "insulin", "change", "diabetesMed"]


def _load_diabetes_frame(csv: Optional[str]) -> pd.DataFrame:
    """Read the local CSV if present; otherwise fetch once from the UCI API
    (with timeout and retries) and cache it as data/raw/diabetic_data.csv so
    that no later run touches the network."""
    path = Path(csv) if csv else DATA_ROOT / "raw" / "diabetic_data.csv"
    if path.exists():
        return pd.read_csv(path, na_values=["?", "None", "Unknown/Invalid"], low_memory=False)
    import socket, time
    from ucimlrepo import fetch_ucirepo
    socket.setdefaulttimeout(60)
    last = None
    for attempt in range(5):
        try:
            rep = fetch_ucirepo(id=296)
            break
        except Exception as e:          # transient server drops happen
            last = e; time.sleep(10 * (attempt + 1))
    else:
        raise RuntimeError(f"could not download UCI 296 after 5 attempts: {last}")
    parts = [rep.data.features, rep.data.targets]
    if rep.data.ids is not None:                       # encounter_id, patient_nbr live here
        parts.insert(0, rep.data.ids)
    df = pd.concat(parts, axis=1)
    df = df.replace({"?": np.nan, "None": np.nan, "Unknown/Invalid": np.nan})
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        df.to_csv(tmp, index=False); tmp.replace(path)
    except Exception:
        pass
    return df


@register("diabetes")
def load_diabetes(n_clients: int = 20, partition: str = "lpt", seed: int = 0,
                  alpha: float = 0.5, csv: Optional[str] = None, max_share: float = 0.15,
                  dedup: bool = True, **_) -> FederatedDataset:
    """UCI id=296. Cleaning follows Strack et al. (2014): drop expired/hospice
    discharge dispositions, keep the FIRST encounter per patient."""
    df = _load_diabetes_frame(csv)
    df.columns = [c.strip() for c in df.columns]
    if "discharge_disposition_id" in df:
        df = df[~df["discharge_disposition_id"].astype(str).isin(["11", "13", "14", "19", "20", "21"])]
    has_patient = "patient_nbr" in df.columns
    if has_patient and "encounter_id" in df.columns:
        df = df.sort_values("encounter_id")
    if dedup and has_patient:
        df = df.drop_duplicates(subset="patient_nbr", keep="first")
    df = df.reset_index(drop=True)
    target = "readmitted" if "readmitted" in df else df.columns[-1]
    y = (df[target].astype(str).str.strip() == "<30").astype(int).values
    num = [c for c in DIAB_NUMERIC if c in df]
    cat = [c for c in DIAB_CATEGORICAL if c in df]
    df[cat] = df[cat].fillna("missing")
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df[num] = df[num].fillna(df[num].median(numeric_only=True))
    x = _encode(df, num, cat)
    groups = df["medical_specialty"].astype(str).values if "medical_specialty" in df else None
    patients = df["patient_nbr"].values if has_patient else None
    meta = {"n_numeric": len(num), "n_categorical": len(cat), "deduplicated": bool(dedup and has_patient),
            "n_encounters_after_cleaning": int(len(df)), "has_patient_id": has_patient}
    return _split_and_partition(x, y, "diabetes", meta, n_clients, partition, seed,
                                groups=groups, patient_groups=patients, alpha=alpha, max_share=max_share)


# --------------------------------------------------------------------------- #
def _fetch(url: str, path: Path, tries: int = 5) -> Path:
    """Download once to `path` (atomic), with retries; later runs read the cache."""
    if path.exists():
        return path
    import time, urllib.request
    path.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data); tmp.replace(path)
            return path
        except Exception as e:
            last = e; time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"could not download {url}: {last}")


EICU_DEMO_URL = "https://physionet.org/files/eicu-crd-demo/2.0.1/patient.csv.gz"


def _eicu_frame(root: Path, demo: bool) -> pd.DataFrame:
    if demo:
        f = _fetch(EICU_DEMO_URL, root / "patient.csv.gz")
        return pd.read_csv(f, compression="gzip", low_memory=False)
    f = root / "patient.csv"
    if not f.exists() and (root / "patient.csv.gz").exists():
        return pd.read_csv(root / "patient.csv.gz", compression="gzip", low_memory=False)
    return pd.read_csv(f, low_memory=False)


@register("eicu_demo")
def load_eicu_demo(n_clients: int = 0, partition: str = "natural", seed: int = 0,
                   root: Optional[str] = None, min_site: int = 80, target: str = "mortality", **kw):
    return load_eicu(n_clients=n_clients, partition=partition, seed=seed,
                     root=root or str(DATA_ROOT / "eicu_demo"), min_site=min_site,
                     target=target, demo=True, **kw)


@register("eicu")
def load_eicu(n_clients: int = 30, partition: str = "natural", seed: int = 0,
              root: Optional[str] = None, min_site: int = 200,
              target: str = "mortality", demo: bool = False, **_) -> FederatedDataset:
    """Features are admission-time only (no outcome-derived offsets). Hospitals
    with fewer than `min_site` stays are DROPPED rather than pooled, so every
    client is one real hospital."""
    root = Path(root) if root else DATA_ROOT / "eicu"
    pat = _eicu_frame(root, demo)
    pat = pat[pat["age"].notna() & pat["hospitaldischargestatus"].notna()].reset_index(drop=True)
    counts = pat["hospitalid"].value_counts()
    pat = pat[pat["hospitalid"].isin(counts[counts >= min_site].index)].reset_index(drop=True)
    if pat["hospitalid"].nunique() < 2:
        raise ValueError(f"fewer than two hospitals with >= {min_site} stays")
    pat["age_num"] = pd.to_numeric(pat["age"].astype(str).str.replace("> 89", "90", regex=False),
                                   errors="coerce").fillna(65)
    if target == "mortality":
        y = (pat["hospitaldischargestatus"].astype(str).str.lower() == "expired").astype(int).values
    else:
        y = (pd.to_numeric(pat["unitdischargeoffset"], errors="coerce").fillna(0) > 3 * 1440).astype(int).values
    num = [c for c in ["age_num", "admissionweight", "admissionheight", "unitvisitnumber"] if c in pat]
    cat = [c for c in ["gender", "ethnicity", "unittype", "unitadmitsource",
                       "unitstaytype", "hospitaladmitsource"] if c in pat]
    pat[cat] = pat[cat].fillna("missing")
    pat[num] = pat[num].apply(pd.to_numeric, errors="coerce")
    pat[num] = pat[num].fillna(pat[num].median(numeric_only=True))
    x = _encode(pat, num, cat)
    patients = pat["uniquepid"].values if "uniquepid" in pat else None
    name = "eicu_demo" if demo else "eicu"
    meta = {"target": target, "true_site_ids": True, "n_hospitals": int(pat["hospitalid"].nunique()),
            "n_stays": int(len(pat)), "min_site": int(min_site), "prevalence": float(y.mean())}
    # the global split is patient-grouped, so a site can lose a few stays; use a
    # partition floor below min_site so that no real hospital is merged away
    return _split_and_partition(x, y, name, meta, n_clients, partition, seed,
                                site_ids=pat["hospitalid"].values, patient_groups=patients,
                                min_size=max(10, min_site // 2))


HEART_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.{}.data"
HEART_SITES = ["cleveland", "hungarian", "switzerland", "va"]
HEART_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang",
              "oldpeak", "slope", "ca", "thal", "num"]


@register("heart")
def load_heart(n_clients: int = 4, partition: str = "natural", seed: int = 0,
               root: Optional[str] = None, **_) -> FederatedDataset:
    """Fed-Heart-Disease: 13 clinical features, target = any angiographic disease
    (num > 0), one client per institution. Missing values ('?') are imputed with
    the pooled median (label-free, as for the other tabular loaders); `ca` and
    `thal` are mostly missing outside Cleveland, so missingness indicators are added."""
    root = Path(root) if root else DATA_ROOT / "heart"
    frames = []
    for site in HEART_SITES:
        f = _fetch(HEART_URL.format(site), root / f"processed.{site}.data")
        df = pd.read_csv(f, header=None, names=HEART_COLS, na_values=["?"])
        df["site"] = site
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    y = (pd.to_numeric(df["num"], errors="coerce").fillna(0) > 0).astype(int).values
    num = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    cat = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c + "_missing"] = df[c].isna().astype(float)
    df[num] = df[num].fillna(df[num].median(numeric_only=True))
    df[cat] = df[cat].astype(str).replace("nan", "missing")
    x = _encode(df, num + [c + "_missing" for c in num], cat)
    meta = {"n_sites": 4, "true_site_ids": True, "n_patients": int(len(df)), "prevalence": float(y.mean())}
    return _split_and_partition(x, y, "heart", meta, n_clients, partition, seed,
                                site_ids=df["site"].values, min_size=10)


@register("mimic")
def load_mimic(n_clients: int = 20, partition: str = "natural", seed: int = 0,
               root: Optional[str] = None, **_) -> FederatedDataset:
    root = Path(root) if root else DATA_ROOT / "mimic"
    adm = pd.read_csv(root / "admissions.csv", low_memory=False)
    adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")
    adm["dischtime"] = pd.to_datetime(adm["dischtime"], errors="coerce")
    adm = adm.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
    nxt = adm.groupby("subject_id")["admittime"].shift(-1)
    y = ((nxt - adm["dischtime"]).dt.days.between(0, 30)).fillna(False).astype(int).values
    adm["los_days"] = (adm["dischtime"] - adm["admittime"]).dt.total_seconds() / 86400.0
    num = [c for c in ["los_days"] if c in adm]
    cat = [c for c in ["admission_type", "admission_location", "insurance", "language",
                       "marital_status", "race", "discharge_location"] if c in adm]
    adm[cat] = adm[cat].fillna("missing")
    adm[num] = adm[num].fillna(adm[num].median(numeric_only=True))
    x = _encode(adm, num, cat)
    return _split_and_partition(x, y, "mimic", {"label": "readmission_30d"},
                                n_clients, partition, seed, site_ids=adm["admission_location"].astype(str).values,
                                patient_groups=adm["subject_id"].values)


# --------------------------------------------------------------------------- #
@register("medmnist")
def load_medmnist(flag: str = "pneumoniamnist", n_clients: int = 20, partition: str = "dirichlet",
                  alpha: float = 0.5, seed: int = 0, size: int = 28, min_size: int = 50,
                  **_) -> FederatedDataset:
    """MedMNIST v2. Binary collections are used as-is; a multi-class collection
    raises unless `binarise_class` is passed explicitly (v2 silently binarised)."""
    import medmnist
    from medmnist import INFO
    info = INFO[flag]
    if len(info["label"]) != 2:
        raise ValueError(f"{flag} has {len(info['label'])} classes; this benchmark uses binary collections "
                         f"(pneumoniamnist, breastmnist).")
    cls = getattr(medmnist, info["python_class"])
    tr = cls(split="train", download=True, size=size)
    va = cls(split="val", download=True, size=size)
    te = cls(split="test", download=True, size=size)
    xtr = np.concatenate([tr.imgs, va.imgs]).astype(np.float32) / 255.0
    xte = te.imgs.astype(np.float32) / 255.0
    if xtr.ndim == 3:
        xtr, xte = xtr[:, None], xte[:, None]
    else:
        xtr, xte = xtr.transpose(0, 3, 1, 2), xte.transpose(0, 3, 1, 2)
    ytr = np.concatenate([tr.labels, va.labels]).reshape(-1).astype(int)
    yte = te.labels.reshape(-1).astype(int)
    parts, rep = build_partition(partition, ytr, n_clients, seed=seed, alpha=alpha, min_size=min_size)
    return FederatedDataset(f"medmnist:{flag}", xtr, ytr, xte, yte, parts,
                            {"modality": "imaging", "flag": flag, "partition_report": rep,
                             "partition": partition, "seed": seed})
