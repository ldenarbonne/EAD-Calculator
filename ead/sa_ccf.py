"""Exposure value of loans and off-balance-sheet items under the
Standardised Approach (CRR3 Art. 111 and Annex I).

EAD = max(drawn - provisions, 0) + CCF x max(undrawn - remaining provisions, 0)

Specific credit risk adjustments ("provisions") are deducted from the
on-balance-sheet amount first; any excess is deducted from the nominal value
of the off-balance-sheet item before the CCF is applied (Art. 111(1)).
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from .parameters import CRR3_BUCKET_CCF, PRODUCT_CATALOGUE, ucc_transitional_factor

REQUIRED_COLUMNS = ["facility_id", "product_type", "drawn", "undrawn"]
OPTIONAL_COLUMNS = {
    "counterparty": "",
    "provisions": 0.0,
    "original_maturity_years": 1.0,
    "risk_weight": 1.0,
}


def crr3_ccf(product_type: str, reporting_date: date) -> tuple[int, float]:
    """Return (Annex I bucket, applicable CCF) for a product at a given date."""
    if product_type not in PRODUCT_CATALOGUE:
        raise ValueError(f"Unknown product type: {product_type!r}")
    bucket = PRODUCT_CATALOGUE[product_type][0]
    ccf = CRR3_BUCKET_CCF[bucket]
    if bucket == 5:
        ccf *= ucc_transitional_factor(reporting_date)
    return bucket, ccf


def crr2_ccf(product_type: str, original_maturity_years: float) -> float:
    """Return the pre-CRR3 (CRR2 Annex I) CCF, used for impact comparison."""
    if product_type not in PRODUCT_CATALOGUE:
        raise ValueError(f"Unknown product type: {product_type!r}")
    ccf = PRODUCT_CATALOGUE[product_type][1]
    if ccf is None:  # undrawn commitments: medium-low vs medium risk
        return 0.20 if original_maturity_years <= 1.0 else 0.50
    return ccf


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    out = df.copy()
    for col, default in OPTIONAL_COLUMNS.items():
        if col not in out.columns:
            out[col] = default
        out[col] = out[col].fillna(default)
    for col in ["drawn", "undrawn", "provisions", "original_maturity_years", "risk_weight"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    if (out[["drawn", "undrawn", "provisions"]] < 0).any().any():
        raise ValueError("Drawn, undrawn and provision amounts must be non-negative.")
    return out


def compute_sa_ccf(df: pd.DataFrame, reporting_date: date) -> pd.DataFrame:
    """Compute exposure values facility by facility.

    Parameters
    ----------
    df:
        One row per facility with at least ``facility_id``, ``product_type``,
        ``drawn`` and ``undrawn``. Optional: ``counterparty``, ``provisions``,
        ``original_maturity_years`` (CRR2 comparison only) and ``risk_weight``
        (decimal, e.g. 1.0 for 100%).
    reporting_date:
        Drives the CRR3 transitional factor for unconditionally cancellable
        commitments.

    Returns
    -------
    DataFrame with bucket, CCFs, on- and off-balance exposure values, EAD,
    RWA and the CRR2 comparison.
    """
    out = _prepare(df)
    if out.empty:
        return out.assign(
            bucket=pd.Series(dtype=int), ccf_crr3=pd.Series(dtype=float),
            ead_on_balance=pd.Series(dtype=float), ead_off_balance=pd.Series(dtype=float),
            ead=pd.Series(dtype=float), rwa=pd.Series(dtype=float),
            ccf_crr2=pd.Series(dtype=float), ead_crr2=pd.Series(dtype=float),
        )

    buckets, ccfs, ccfs_crr2 = [], [], []
    for product, maturity in zip(out["product_type"], out["original_maturity_years"]):
        bucket, ccf = crr3_ccf(product, reporting_date)
        buckets.append(bucket)
        ccfs.append(ccf)
        ccfs_crr2.append(crr2_ccf(product, maturity))

    out["bucket"] = buckets
    out["ccf_crr3"] = ccfs
    out["ccf_crr2"] = ccfs_crr2

    provisions_on = out[["provisions", "drawn"]].min(axis=1)
    provisions_off = out["provisions"] - provisions_on
    out["ead_on_balance"] = out["drawn"] - provisions_on
    net_undrawn = (out["undrawn"] - provisions_off).clip(lower=0.0)
    out["ead_off_balance"] = out["ccf_crr3"] * net_undrawn
    out["ead"] = out["ead_on_balance"] + out["ead_off_balance"]
    out["rwa"] = out["ead"] * out["risk_weight"]
    out["ead_crr2"] = out["ead_on_balance"] + out["ccf_crr2"] * net_undrawn
    return out


def summarise_by_bucket(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results by CRR3 bucket."""
    if results.empty:
        return pd.DataFrame(columns=["bucket", "undrawn", "ead_off_balance", "ead", "ead_crr2", "rwa"])
    summary = (
        results.groupby("bucket", as_index=False)[
            ["undrawn", "ead_off_balance", "ead", "ead_crr2", "rwa"]
        ].sum()
    )
    summary["ccf"] = summary["bucket"].map(CRR3_BUCKET_CCF)
    return summary
