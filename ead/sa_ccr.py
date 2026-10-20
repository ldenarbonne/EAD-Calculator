"""Standardised Approach for Counterparty Credit Risk (SA-CCR).

Implements CRR Art. 274-280f (as introduced by CRR2 and amended by CRR3),
which transpose BCBS 279:

    EAD = alpha x (RC + PFE)
    PFE = multiplier x AddOn_aggregate

Covered: interest rate, FX, credit (single name and index), equity
(single name and index) and commodity derivatives, linear trades and
European options, margined and unmargined netting sets.

Not covered in this version: CDO tranches, basis and volatility transactions,
the simplified SA-CCR and the original exposure method (CRR Art. 281-282).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, log, sqrt

import numpy as np
import pandas as pd
from scipy.stats import norm

from .parameters import (
    ALPHA_DEFAULT,
    BUSINESS_DAYS_PER_YEAR,
    COMMODITY_HEDGING_SETS,
    CORRELATION,
    IR_BUCKET_BOUNDS,
    IR_BUCKET_CORRELATION,
    MULTIPLIER_FLOOR,
    SUPERVISORY_FACTOR,
    SUPERVISORY_OPTION_VOL,
)

TRADE_COLUMNS = [
    "netting_set", "trade_id", "asset_class", "hedging_set", "risk_entity",
    "rating", "is_index", "instrument", "direction", "option_type",
    "option_position", "underlying_price", "strike", "lambda_shift",
    "notional", "mtm", "start_years", "end_years", "maturity_years",
    "exercise_years",
]

TEN_DAYS_IN_YEARS = 10 / BUSINESS_DAYS_PER_YEAR


# ---------------------------------------------------------------------------
# Netting-set parameters
# ---------------------------------------------------------------------------

@dataclass
class NettingSet:
    """Collateral and margining terms of one netting set.

    All amounts are expressed in the reporting currency.

    Attributes
    ----------
    name: identifier matching the ``netting_set`` column of the trades.
    margined: True if the netting set is subject to a variation margin agreement.
    collateral: C, haircut value of net collateral held (VM and IM, received
        minus posted). Positive means collateral received.
    threshold: TH, threshold below which no variation margin is called.
    mta: MTA, minimum transfer amount.
    nica: NICA, net independent collateral amount.
    mpor_days: margin period of risk in business days (margined sets only).
    alpha: regulatory multiplier (1.4 by default).
    """

    name: str
    margined: bool = False
    collateral: float = 0.0
    threshold: float = 0.0
    mta: float = 0.0
    nica: float = 0.0
    mpor_days: float = 10.0
    alpha: float = ALPHA_DEFAULT


# ---------------------------------------------------------------------------
# Trade-level building blocks
# ---------------------------------------------------------------------------

def supervisory_duration(start: float, end: float) -> float:
    """SD = (exp(-0.05 S) - exp(-0.05 E)) / 0.05  (CRR Art. 279b(1)(a))."""
    start = max(start, 0.0)
    end = max(end, start, 0.0)
    return (exp(-0.05 * start) - exp(-0.05 * end)) / 0.05


def maturity_factor(maturity: float, margined: bool, mpor_days: float = 10.0) -> float:
    """Maturity factor (CRR Art. 279c(1)).

    Unmargined: sqrt(min(max(M, 10 business days), 1 year) / 1 year).
    Margined:   1.5 x sqrt(MPOR / 1 year).
    """
    if margined:
        return 1.5 * sqrt(mpor_days / BUSINESS_DAYS_PER_YEAR)
    m = min(max(maturity, TEN_DAYS_IN_YEARS), 1.0)
    return sqrt(m)


def option_delta(
    option_type: str,
    position: str,
    price: float,
    strike: float,
    vol: float,
    expiry: float,
    shift: float = 0.0,
) -> float:
    """Supervisory delta of a European option (CRR Art. 279a(1)(a)).

    d1 = [ln((P + lambda)/(K + lambda)) + 0.5 sigma^2 T] / (sigma sqrt(T))
    Bought call:  N(d1)     Sold call: -N(d1)
    Bought put:  -N(-d1)    Sold put:   N(-d1)

    ``shift`` is the lambda used for interest rate or commodity options when
    prices can be negative (EBA RTS on SA-CCR, as amended under CRR3).
    """
    option_type = option_type.upper()
    position = position.upper()
    p, k = price + shift, strike + shift
    if p <= 0 or k <= 0:
        raise ValueError("Underlying and strike (plus shift) must be strictly positive.")
    t = max(expiry, TEN_DAYS_IN_YEARS)
    d1 = (log(p / k) + 0.5 * vol**2 * t) / (vol * sqrt(t))
    if option_type == "CALL":
        delta = norm.cdf(d1)
    elif option_type == "PUT":
        delta = -norm.cdf(-d1)
    else:
        raise ValueError(f"Unknown option type: {option_type!r}")
    if position == "SOLD":
        delta = -delta
    elif position != "BOUGHT":
        raise ValueError(f"Unknown option position: {position!r}")
    return float(delta)


def _is_electricity(row: pd.Series) -> bool:
    return str(row.get("risk_entity", "")).strip().lower() == "electricity"


def trade_supervisory_factor(row: pd.Series) -> float:
    """Supervisory factor for one trade."""
    ac = row["asset_class"]
    if ac == "IR":
        return SUPERVISORY_FACTOR["IR"]
    if ac == "FX":
        return SUPERVISORY_FACTOR["FX"]
    if ac == "CREDIT":
        table = "CREDIT_INDEX" if row["is_index"] else "CREDIT_SINGLE"
        rating = str(row["rating"]).upper()
        if rating not in SUPERVISORY_FACTOR[table]:
            expected = ", ".join(SUPERVISORY_FACTOR[table])
            raise ValueError(
                f"Trade {row['trade_id']}: rating {rating!r} invalid for "
                f"{'index' if row['is_index'] else 'single-name'} credit ({expected})."
            )
        return SUPERVISORY_FACTOR[table][rating]
    if ac == "EQUITY":
        return SUPERVISORY_FACTOR["EQUITY_INDEX" if row["is_index"] else "EQUITY_SINGLE"]
    if ac == "COMMODITY":
        key = "COMMODITY_ELECTRICITY" if _is_electricity(row) else "COMMODITY_OTHER"
        return SUPERVISORY_FACTOR[key]
    raise ValueError(f"Trade {row['trade_id']}: unknown asset class {ac!r}.")


def trade_option_vol(row: pd.Series) -> float:
    ac = row["asset_class"]
    if ac in ("IR", "FX"):
        return SUPERVISORY_OPTION_VOL[ac]
    if ac == "CREDIT":
        return SUPERVISORY_OPTION_VOL["CREDIT_INDEX" if row["is_index"] else "CREDIT_SINGLE"]
    if ac == "EQUITY":
        return SUPERVISORY_OPTION_VOL["EQUITY_INDEX" if row["is_index"] else "EQUITY_SINGLE"]
    key = "COMMODITY_ELECTRICITY" if _is_electricity(row) else "COMMODITY_OTHER"
    return SUPERVISORY_OPTION_VOL[key]


def ir_bucket(end_years: float) -> int:
    """Interest-rate maturity bucket from the end date E (CRR Art. 280a)."""
    low, high = IR_BUCKET_BOUNDS
    if end_years < low:
        return 1
    if end_years <= high:
        return 2
    return 3


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def _normalise_trades(trades: pd.DataFrame) -> pd.DataFrame:
    df = trades.copy()
    for col in TRADE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    text_cols = ["netting_set", "trade_id", "asset_class", "hedging_set", "risk_entity",
                 "rating", "instrument", "direction", "option_type", "option_position"]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()
    for col in ["asset_class", "instrument", "direction", "option_type", "option_position", "rating"]:
        df[col] = df[col].str.upper()
    df.loc[df["hedging_set"].eq("") & df["asset_class"].isin(["CREDIT", "EQUITY"]), "hedging_set"] = "ALL"
    df["hedging_set"] = np.where(df["asset_class"].eq("COMMODITY"),
                                 df["hedging_set"].str.upper(), df["hedging_set"])
    df["is_index"] = df["is_index"].map(
        lambda v: str(v).strip().lower() in ("true", "1", "yes", "y") if not isinstance(v, bool) else v
    )
    num_cols = ["underlying_price", "strike", "lambda_shift", "notional", "mtm",
                "start_years", "end_years", "maturity_years", "exercise_years"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["lambda_shift"] = df["lambda_shift"].fillna(0.0)
    df["start_years"] = df["start_years"].fillna(0.0)
    df["mtm"] = df["mtm"].fillna(0.0)
    df["end_years"] = df["end_years"].fillna(df["maturity_years"])
    df["maturity_years"] = df["maturity_years"].fillna(df["end_years"])
    df.loc[df["instrument"].eq(""), "instrument"] = "LINEAR"
    return df


def _validate(df: pd.DataFrame) -> None:
    errors = []
    for _, r in df.iterrows():
        tid = r["trade_id"] or "<no id>"
        if r["asset_class"] not in ("IR", "FX", "CREDIT", "EQUITY", "COMMODITY"):
            errors.append(f"{tid}: asset_class must be IR, FX, CREDIT, EQUITY or COMMODITY.")
            continue
        if pd.isna(r["notional"]) or r["notional"] < 0:
            errors.append(f"{tid}: notional must be a non-negative number.")
        if pd.isna(r["maturity_years"]):
            errors.append(f"{tid}: maturity_years (or end_years) is required.")
        if r["asset_class"] in ("IR", "FX") and not r["hedging_set"]:
            errors.append(f"{tid}: hedging_set (currency or currency pair) is required.")
        if r["asset_class"] in ("CREDIT", "EQUITY", "COMMODITY") and not r["risk_entity"]:
            errors.append(f"{tid}: risk_entity is required for {r['asset_class']} trades.")
        if r["asset_class"] == "COMMODITY" and r["hedging_set"] not in COMMODITY_HEDGING_SETS:
            errors.append(f"{tid}: commodity hedging_set must be one of {', '.join(COMMODITY_HEDGING_SETS)}.")
        if r["instrument"] == "LINEAR" and r["direction"] not in ("LONG", "SHORT"):
            errors.append(f"{tid}: direction must be LONG or SHORT for linear trades.")
        if r["instrument"] == "OPTION":
            if r["option_type"] not in ("CALL", "PUT") or r["option_position"] not in ("BOUGHT", "SOLD"):
                errors.append(f"{tid}: options need option_type CALL/PUT and option_position BOUGHT/SOLD.")
            if any(pd.isna(r[c]) for c in ("underlying_price", "strike", "exercise_years")):
                errors.append(f"{tid}: options need underlying_price, strike and exercise_years.")
        if r["instrument"] not in ("LINEAR", "OPTION"):
            errors.append(f"{tid}: instrument must be LINEAR or OPTION.")
    if errors:
        raise ValueError("Invalid trade data:\n- " + "\n- ".join(errors))


# ---------------------------------------------------------------------------
# Trade-level computation
# ---------------------------------------------------------------------------

def compute_trade_level(trades: pd.DataFrame, netting_set: NettingSet) -> pd.DataFrame:
    """Return the trades with SD, adjusted notional, delta, MF and add-on inputs."""
    df = _normalise_trades(trades)
    _validate(df)
    if df.empty:
        return df

    sfs, sds, adj, mfs, deltas, buckets = [], [], [], [], [], []
    for _, r in df.iterrows():
        sf = trade_supervisory_factor(r)
        if r["asset_class"] in ("IR", "CREDIT"):
            sd = supervisory_duration(r["start_years"], r["end_years"])
            d = r["notional"] * sd
        else:
            sd = np.nan
            d = r["notional"]
        mf = maturity_factor(r["maturity_years"], netting_set.margined, netting_set.mpor_days)
        if r["instrument"] == "OPTION":
            delta = option_delta(
                r["option_type"], r["option_position"], r["underlying_price"], r["strike"],
                trade_option_vol(r), r["exercise_years"], r["lambda_shift"],
            )
        else:
            delta = 1.0 if r["direction"] == "LONG" else -1.0
        sfs.append(sf)
        sds.append(sd)
        adj.append(d)
        mfs.append(mf)
        deltas.append(delta)
        buckets.append(ir_bucket(r["end_years"]) if r["asset_class"] == "IR" else np.nan)

    df["supervisory_factor"] = sfs
    df["supervisory_duration"] = sds
    df["adjusted_notional"] = adj
    df["maturity_factor"] = mfs
    df["delta"] = deltas
    df["ir_bucket"] = buckets
    df["effective_notional"] = df["delta"] * df["adjusted_notional"] * df["maturity_factor"]
    df["trade_addon_signed"] = df["supervisory_factor"] * df["effective_notional"]
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _ir_addon(group: pd.DataFrame) -> tuple[float, dict]:
    d = {k: group.loc[group["ir_bucket"] == k, "effective_notional"].sum() for k in (1, 2, 3)}
    en_sq = d[1] ** 2 + d[2] ** 2 + d[3] ** 2
    for (i, j), rho in IR_BUCKET_CORRELATION.items():
        en_sq += 2 * rho * d[i] * d[j]
    en = sqrt(max(en_sq, 0.0))
    return SUPERVISORY_FACTOR["IR"] * en, {"D1": d[1], "D2": d[2], "D3": d[3], "effective_notional": en}


def _single_factor_addon(group: pd.DataFrame, corr_single: float, corr_index: float) -> tuple[float, dict]:
    """Credit / equity (and commodity with a common correlation) aggregation."""
    entity = group.groupby(["risk_entity", "is_index"], as_index=False)["trade_addon_signed"].sum()
    entity["rho"] = np.where(entity["is_index"], corr_index, corr_single)
    systematic = (entity["rho"] * entity["trade_addon_signed"]).sum()
    idiosyncratic = ((1 - entity["rho"] ** 2) * entity["trade_addon_signed"] ** 2).sum()
    addon = sqrt(systematic**2 + idiosyncratic)
    return addon, {"systematic": systematic, "idiosyncratic": idiosyncratic,
                   "entities": len(entity)}


def compute_netting_set(trades: pd.DataFrame, netting_set: NettingSet) -> "NettingSetResult":
    """Full SA-CCR calculation for one netting set."""
    tl = compute_trade_level(trades, netting_set)
    hedging_rows = []
    if not tl.empty:
        for (ac, hs), group in tl.groupby(["asset_class", "hedging_set"], sort=True):
            if ac == "IR":
                addon, detail = _ir_addon(group)
            elif ac == "FX":
                en = group["effective_notional"].sum()
                addon, detail = SUPERVISORY_FACTOR["FX"] * abs(en), {"effective_notional": en}
            elif ac == "CREDIT":
                addon, detail = _single_factor_addon(group, CORRELATION["CREDIT_SINGLE"],
                                                     CORRELATION["CREDIT_INDEX"])
            elif ac == "EQUITY":
                addon, detail = _single_factor_addon(group, CORRELATION["EQUITY_SINGLE"],
                                                     CORRELATION["EQUITY_INDEX"])
            else:  # COMMODITY: commodity types inside a hedging set, rho = 40%
                addon, detail = _single_factor_addon(group, CORRELATION["COMMODITY"],
                                                     CORRELATION["COMMODITY"])
            hedging_rows.append({"asset_class": ac, "hedging_set": hs, "addon": addon,
                                 "trades": len(group), **detail})
    hedging = pd.DataFrame(hedging_rows)
    addon_by_class = (hedging.groupby("asset_class", as_index=False)["addon"].sum()
                      if not hedging.empty else pd.DataFrame(columns=["asset_class", "addon"]))
    addon_agg = float(hedging["addon"].sum()) if not hedging.empty else 0.0

    v = float(tl["mtm"].sum()) if not tl.empty else 0.0
    c = netting_set.collateral
    rc_unmargined = max(v - c, 0.0)
    if netting_set.margined:
        rc = max(v - c, netting_set.threshold + netting_set.mta - netting_set.nica, 0.0)
    else:
        rc = rc_unmargined

    mult = pfe_multiplier(v, c, addon_agg)
    pfe = mult * addon_agg
    ead = netting_set.alpha * (rc + pfe)

    ead_unmargined = None
    capped = False
    if netting_set.margined and not tl.empty:
        unmargined_ns = NettingSet(name=netting_set.name, margined=False, collateral=c,
                                   alpha=netting_set.alpha)
        unmargined = compute_netting_set(trades, unmargined_ns)
        ead_unmargined = unmargined.ead
        if ead_unmargined < ead:
            ead, capped = ead_unmargined, True

    return NettingSetResult(
        netting_set=netting_set, trades=tl, hedging_sets=hedging,
        addon_by_class=addon_by_class, addon_aggregate=addon_agg, mtm=v,
        replacement_cost=rc, multiplier=mult, pfe=pfe, ead=ead,
        ead_unmargined=ead_unmargined, capped_at_unmargined=capped,
    )


def pfe_multiplier(v: float, c: float, addon_aggregate: float,
                   floor: float = MULTIPLIER_FLOOR) -> float:
    """multiplier = min{1; floor + (1 - floor) exp[(V - C) / (2 (1 - floor) AddOn)]}."""
    if addon_aggregate <= 0:
        return 1.0
    return min(1.0, floor + (1 - floor) * exp((v - c) / (2 * (1 - floor) * addon_aggregate)))


@dataclass
class NettingSetResult:
    netting_set: NettingSet
    trades: pd.DataFrame
    hedging_sets: pd.DataFrame
    addon_by_class: pd.DataFrame
    addon_aggregate: float
    mtm: float
    replacement_cost: float
    multiplier: float
    pfe: float
    ead: float
    ead_unmargined: float | None = None
    capped_at_unmargined: bool = False
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "netting_set": self.netting_set.name,
            "margined": self.netting_set.margined,
            "V (net MtM)": self.mtm,
            "C (collateral)": self.netting_set.collateral,
            "RC": self.replacement_cost,
            "AddOn aggregate": self.addon_aggregate,
            "multiplier": self.multiplier,
            "PFE": self.pfe,
            "alpha": self.netting_set.alpha,
            "EAD": self.ead,
            "EAD unmargined": self.ead_unmargined,
            "capped at unmargined EAD": self.capped_at_unmargined,
        }


def compute_portfolio(trades: pd.DataFrame, netting_sets: list[NettingSet]) -> list[NettingSetResult]:
    """Run SA-CCR for every netting set referenced in ``trades``."""
    trades = trades.copy()
    trades["netting_set"] = trades["netting_set"].fillna("").astype(str).str.strip()
    params = {ns.name: ns for ns in netting_sets}
    results = []
    for name, group in trades.groupby("netting_set", sort=True):
        ns = params.get(name, NettingSet(name=name))
        results.append(compute_netting_set(group, ns))
    return results


def ead_collateral_sensitivity(trades: pd.DataFrame, netting_set: NettingSet,
                               collateral_grid: np.ndarray) -> pd.DataFrame:
    """EAD, RC and multiplier as a function of collateral held."""
    rows = []
    for c in collateral_grid:
        ns = NettingSet(**{**netting_set.__dict__, "collateral": float(c)})
        res = compute_netting_set(trades, ns)
        rows.append({"collateral": float(c), "ead": res.ead, "rc": res.replacement_cost,
                     "pfe": res.pfe, "multiplier": res.multiplier})
    return pd.DataFrame(rows)
