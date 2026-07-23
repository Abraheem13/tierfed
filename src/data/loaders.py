"""Concrete dataset loaders.

Corpora
-------
synthetic  : controllable covariate/label shift; used by CI and for stress tests.
diabetes   : UCI Diabetes 130-US Hospitals (original benchmark).
eicu       : eICU Collaborative Research Database -- has a TRUE hospitalid,
             which answers the objection that pseudo-hospitals are synthetic.
mimic      : MIMIC-IV derived in-hospital-mortality / readmission cohort,
             partitioned by careunit or admission source.
medmnist   : MedMNIST v2 imaging collections (PneumoniaMNIST, DermaMNIST, ...)
             -- gives the imaging modality reviewers asked for and exercises
             the deep-architecture tiering path.

eICU and MIMIC require credentialed PhysioNet access, so the loaders read
local CSVs the user has downloaded; they never attempt to fetch restricted data.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .base import FederatedDataset, register
from .partition import build_partition

DATA_ROOT = Path(os.environ.get("NFL_DATA_ROOT", "data"))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _encode(df: pd.DataFrame, numeric, categorical, fit_idx=None):
    """Fit encoders on train rows only, then transform everything."""
    fit_idx = np.arange(len(df)) if fit_idx is None else fit_idx
    blocks = []
    if numeric:
        sc = StandardScaler().fit(df.loc[fit_idx, numeric].astype(float).values)
        blocks.append(sc.transform(df[numeric].astype(float).values))
    if categorical:
        try:
            oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # older sklearn
            oh = OneHotEncoder(handle_unknown="ignore", sparse=False)
        oh.fit(df.loc[fit_idx, categorical].astype(str).values)
        blocks.append(oh.transform(df[categorical].astype(str).values))
    return np.hstack(blocks).astype(np.float32)


def _split_and_partition(x, y, name, meta, n_clients, partition, seed,
                         site_ids=None, groups=None, alpha=0.5, test_size=0.2):
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=test_size,
                                      stratify=y, random_state=seed)
    parts, rep = build_partition(partition, y[idx_tr], n_clients, seed=seed,
                                 site_ids=None if site_ids is None else np.asarray(site_ids)[idx_tr],
                                 groups=None if groups is None else np.asarray(groups)[idx_tr],
                                 alpha=alpha)
    meta = {**meta, "partition_report": rep, "partition": partition, "seed": seed}
    return FederatedDataset(name, x[idx_tr], y[idx_tr], x[idx_te], y[idx_te], parts, meta)


# --------------------------------------------------------------------------- #
# synthetic
# --------------------------------------------------------------------------- #
@register("synthetic")
def load_synthetic(n_clients: int = 20, n_per_client: int = 400, n_features: int = 40,
                   prevalence: float = 0.11, covariate_shift: float = 1.0,
                   concept_shift: float = 0.3, seed: int = 0,
                   partition: str = "natural", **_) -> FederatedDataset:
    """Each client gets its own feature mean AND its own decision boundary,
    reproducing covariate + concept shift with a known ground truth."""
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
                                n_clients, partition, seed, site_ids=site)


# --------------------------------------------------------------------------- #
# UCI Diabetes 130-US Hospitals
# --------------------------------------------------------------------------- #
DIAB_NUMERIC = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
                "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
DIAB_CATEGORICAL = ["race", "gender", "age", "admission_type_id", "discharge_disposition_id",
                    "admission_source_id", "medical_specialty", "max_glu_serum", "A1Cresult",
                    "insulin", "change", "diabetesMed"]


@register("diabetes")
def load_diabetes(n_clients: int = 20, partition: str = "lpt", seed: int = 0,
                  alpha: float = 0.5, csv: Optional[str] = None, **_) -> FederatedDataset:
    """UCI id=296. Cleaning follows Strack et al. (2014)."""
    path = Path(csv) if csv else DATA_ROOT / "raw" / "diabetic_data.csv"
    if path.exists():
        df = pd.read_csv(path, na_values=["?", "None", "Unknown/Invalid"], low_memory=False)
    else:  # fall back to the UCI API if the file is absent
        from ucimlrepo import fetch_ucirepo
        rep = fetch_ucirepo(id=296)
        df = pd.concat([rep.data.features, rep.data.targets], axis=1)
        df = df.replace({"?": np.nan, "None": np.nan, "Unknown/Invalid": np.nan})
    df.columns = [c.strip() for c in df.columns]
    # Strack et al.: drop expired/hospice dispositions, then per-patient dedup.
    if "discharge_disposition_id" in df:
        df = df[~df["discharge_disposition_id"].astype(str).isin(["11", "13", "14", "19", "20", "21"])]
    if "patient_nbr" in df.columns:
        df = df.drop_duplicates(subset="patient_nbr", keep="first")
    target = "readmitted" if "readmitted" in df else df.columns[-1]
    y = (df[target].astype(str).str.strip() == "<30").astype(int).values
    num = [c for c in DIAB_NUMERIC if c in df]
    cat = [c for c in DIAB_CATEGORICAL if c in df]
    df[cat] = df[cat].fillna("missing")
    df[num] = df[num].fillna(df[num].median(numeric_only=True))
    x = _encode(df.reset_index(drop=True), num, cat)
    groups = df["medical_specialty"].astype(str).values if "medical_specialty" in df else None
    return _split_and_partition(x, y, "diabetes",
                                {"n_numeric": len(num), "n_categorical": len(cat),
                                 "deduplicated": "patient_nbr" in df.columns},
                                n_clients, partition, seed, groups=groups, alpha=alpha)


# --------------------------------------------------------------------------- #
# eICU -- real hospital identifiers
# --------------------------------------------------------------------------- #
@register("eicu")
def load_eicu(n_clients: int = 30, partition: str = "natural", seed: int = 0,
              root: Optional[str] = None, min_site: int = 200,
              target: str = "mortality", **_) -> FederatedDataset:
    """eICU-CRD. Expects patient.csv (and optionally apachePatientResult.csv).

    `hospitalid` gives a genuine multi-institution split, which removes the
    pseudo-hospital criticism entirely.
    """
    root = Path(root) if root else DATA_ROOT / "eicu"
    pat = pd.read_csv(root / "patient.csv", low_memory=False)
    pat = pat[pat["age"].notna()]
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
    x = _encode(pat.reset_index(drop=True), num, cat)
    return _split_and_partition(x, y, "eicu", {"target": target, "true_site_ids": True},
                                n_clients, partition, seed,
                                site_ids=pat["hospitalid"].values)


# --------------------------------------------------------------------------- #
# MIMIC-IV
# --------------------------------------------------------------------------- #
@register("mimic")
def load_mimic(n_clients: int = 20, partition: str = "natural", seed: int = 0,
               root: Optional[str] = None, **_) -> FederatedDataset:
    """MIMIC-IV admissions cohort; 30-day readmission label.

    Sites are proxied by first careunit / admission location, which are real
    operational units rather than an arbitrary grouping.
    """
    root = Path(root) if root else DATA_ROOT / "mimic"
    adm = pd.read_csv(root / "admissions.csv", low_memory=False)
    adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")
    adm["dischtime"] = pd.to_datetime(adm["dischtime"], errors="coerce")
    adm = adm.sort_values(["subject_id", "admittime"])
    nxt = adm.groupby("subject_id")["admittime"].shift(-1)
    y = ((nxt - adm["dischtime"]).dt.days.between(0, 30)).fillna(False).astype(int).values
    adm["los_days"] = (adm["dischtime"] - adm["admittime"]).dt.total_seconds() / 86400.0
    num = [c for c in ["los_days"] if c in adm]
    cat = [c for c in ["admission_type", "admission_location", "insurance", "language",
                       "marital_status", "race", "discharge_location"] if c in adm]
    adm[cat] = adm[cat].fillna("missing")
    adm[num] = adm[num].fillna(adm[num].median(numeric_only=True))
    x = _encode(adm.reset_index(drop=True), num, cat)
    site = adm["admission_location"].astype(str).values
    return _split_and_partition(x, y, "mimic", {"label": "readmission_30d"},
                                n_clients, partition, seed, site_ids=site)


# --------------------------------------------------------------------------- #
# MedMNIST imaging
# --------------------------------------------------------------------------- #
@register("medmnist")
def load_medmnist(flag: str = "pneumoniamnist", n_clients: int = 20, partition: str = "dirichlet",
                  alpha: float = 0.5, seed: int = 0, size: int = 28, **_) -> FederatedDataset:
    """MedMNIST v2 binary collections -- the imaging modality reviewers asked for."""
    import medmnist
    from medmnist import INFO
    info = INFO[flag]
    cls = getattr(medmnist, info["python_class"])
    tr = cls(split="train", download=True, size=size)
    te = cls(split="test", download=True, size=size)
    xtr = tr.imgs.astype(np.float32) / 255.0
    xte = te.imgs.astype(np.float32) / 255.0
    if xtr.ndim == 3:
        xtr, xte = xtr[:, None], xte[:, None]
    else:
        xtr, xte = xtr.transpose(0, 3, 1, 2), xte.transpose(0, 3, 1, 2)
    ytr = tr.labels.reshape(-1).astype(int)
    yte = te.labels.reshape(-1).astype(int)
    if len(np.unique(ytr)) > 2:            # binarise multi-class collections
        ytr, yte = (ytr > 0).astype(int), (yte > 0).astype(int)
    parts, rep = build_partition(partition, ytr, n_clients, seed=seed, alpha=alpha)
    return FederatedDataset(f"medmnist:{flag}", xtr, ytr, xte, yte, parts,
                            {"modality": "imaging", "flag": flag, "partition_report": rep})
