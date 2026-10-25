"""Loss-threshold membership inference (Yeom et al., CSF 2018).

Given per-sample losses of training members and non-members of a client's
personalised model, the attacker predicts "member" when the loss is small.
The attack's AUC (0.5 = no leakage) is a standard, model-agnostic measure of
how much a released model memorises its training set. It is computed against
every client's deployed (personalised) model and averaged.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def membership_inference(loss_members: np.ndarray, loss_nonmembers: np.ndarray) -> float:
    y = np.concatenate([np.ones(len(loss_members)), np.zeros(len(loss_nonmembers))])
    s = -np.concatenate([loss_members, loss_nonmembers])       # lower loss => member
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))
