"""Regulatory parameters used by the EAD calculator.

All values are taken from Regulation (EU) No 575/2013 as amended by
Regulation (EU) 2019/876 (CRR2) and Regulation (EU) 2024/1623 (CRR3),
and from the Basel Committee standard BCBS 279 (SA-CCR, March 2014).

They are deliberately kept in a single module so that they can be reviewed
and, if needed, recalibrated in one place.
"""

from __future__ import annotations

from datetime import date

# ---------------------------------------------------------------------------
# 1. Standardised credit conversion factors (SA-CCF) - CRR3 Art. 111 & Annex I
# ---------------------------------------------------------------------------

#: CRR3 Article 111(2): percentage applied to the nominal value per bucket.
CRR3_BUCKET_CCF: dict[int, float] = {
    1: 1.00,  # Bucket 1 - full risk
    2: 0.50,  # Bucket 2
    3: 0.40,  # Bucket 3
    4: 0.20,  # Bucket 4 - trade finance
    5: 0.10,  # Bucket 5 - unconditionally cancellable commitments (UCC)
}

#: Product catalogue: label -> (CRR3 Annex I bucket, CRR2 CCF, short description).
#: The CRR2 column is used only for the CRR2 vs CRR3 impact comparison.
#: ``None`` for CRR2 means "depends on original maturity" (undrawn commitments:
#: 20% if original maturity <= 1 year, 50% otherwise).
PRODUCT_CATALOGUE: dict[str, tuple[int, float | None, str]] = {
    "Credit substitute guarantee": (
        1, 1.00, "Guarantees having the character of credit substitutes"),
    "Acceptance / endorsement of bills": (
        1, 1.00, "Acceptances, endorsements, transactions with recourse"),
    "Forward asset purchase / forward deposit": (
        1, 1.00, "Forward purchases, forward deposits, partly-paid shares"),
    "Commitment the client must draw": (
        1, 1.00, "Commitments where the client must draw certain amounts"),
    "Performance bond / bid bond / warranty": (
        2, 0.50, "Transaction-related contingent items (non-credit event)"),
    "Transaction-related standby LC": (
        2, 0.50, "Standby letters of credit related to particular transactions"),
    "NIF / RUF": (
        2, 0.50, "Note issuance and revolving underwriting facilities"),
    "Undrawn committed credit facility": (
        3, None, "Undrawn commitments that are not unconditionally cancellable"),
    "Trade finance documentary credit": (
        4, 0.20, "Short-term self-liquidating trade LCs (shipment as collateral)"),
    "Unconditionally cancellable commitment (UCC)": (
        5, 0.00, "Commitments the institution can cancel at any time without notice"),
}

#: CRR3 Article 495d - transitional factor applied to the 10% UCC percentage.
UCC_TRANSITIONAL_SCHEDULE: list[tuple[date, date, float]] = [
    (date(2025, 1, 1), date(2029, 12, 31), 0.00),
    (date(2030, 1, 1), date(2030, 12, 31), 0.25),
    (date(2031, 1, 1), date(2031, 12, 31), 0.50),
    (date(2032, 1, 1), date(2032, 12, 31), 0.75),
]


def ucc_transitional_factor(reporting_date: date) -> float:
    """Return the CRR3 Art. 495d multiplier for UCC at ``reporting_date``.

    The factor is 0% until end-2029, then 25%, 50%, 75% and 100% from 2033.
    """
    for start, end, factor in UCC_TRANSITIONAL_SCHEDULE:
        if start <= reporting_date <= end:
            return factor
    if reporting_date < UCC_TRANSITIONAL_SCHEDULE[0][0]:
        return 0.0
    return 1.0


# ---------------------------------------------------------------------------
# 2. SA-CCR - CRR Art. 274-280f / BCBS 279
# ---------------------------------------------------------------------------

#: Regulatory multiplier (CRR Art. 274(2)).
ALPHA_DEFAULT: float = 1.4

#: Floor of the PFE multiplier (CRR Art. 278(3)).
MULTIPLIER_FLOOR: float = 0.05

#: Interest-rate maturity buckets (end date E, in years) - CRR Art. 280a.
IR_BUCKET_BOUNDS: tuple[float, float] = (1.0, 5.0)

#: Correlation used across IR maturity buckets (Basel formula).
IR_BUCKET_CORRELATION: dict[tuple[int, int], float] = {
    (1, 2): 0.70,
    (2, 3): 0.70,
    (1, 3): 0.30,
}

#: Supervisory factors, correlations and option volatilities (CRR Art. 280a-280f,
#: Table 2 of Art. 280c-280e; BCBS 279 Table 2).
SUPERVISORY_FACTOR = {
    "IR": 0.005,
    "FX": 0.04,
    "CREDIT_SINGLE": {
        "AAA": 0.0038, "AA": 0.0038, "A": 0.0042, "BBB": 0.0054,
        "BB": 0.0106, "B": 0.016, "CCC": 0.06,
    },
    "CREDIT_INDEX": {"IG": 0.0038, "SG": 0.0106},
    "EQUITY_SINGLE": 0.32,
    "EQUITY_INDEX": 0.20,
    "COMMODITY_ELECTRICITY": 0.40,
    "COMMODITY_OTHER": 0.18,
}

CORRELATION = {
    "CREDIT_SINGLE": 0.50,
    "CREDIT_INDEX": 0.80,
    "EQUITY_SINGLE": 0.50,
    "EQUITY_INDEX": 0.80,
    "COMMODITY": 0.40,
}

SUPERVISORY_OPTION_VOL = {
    "IR": 0.50,
    "FX": 0.15,
    "CREDIT_SINGLE": 1.00,
    "CREDIT_INDEX": 0.80,
    "EQUITY_SINGLE": 1.20,
    "EQUITY_INDEX": 0.75,
    "COMMODITY_ELECTRICITY": 1.50,
    "COMMODITY_OTHER": 0.70,
}

#: Commodity hedging sets (CRR Art. 277(3)).
COMMODITY_HEDGING_SETS = ("ENERGY", "METALS", "AGRICULTURAL", "OTHER")

ASSET_CLASSES = ("IR", "FX", "CREDIT", "EQUITY", "COMMODITY")

CREDIT_RATINGS = tuple(SUPERVISORY_FACTOR["CREDIT_SINGLE"].keys()) + ("IG", "SG")

#: Business days per year used to convert MPOR and the 10-day floor.
BUSINESS_DAYS_PER_YEAR: int = 250
