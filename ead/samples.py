"""Sample portfolios used to pre-fill the application and in the tests."""

from __future__ import annotations

import pandas as pd

from .sa_ccr import TRADE_COLUMNS, NettingSet


def _trades(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in TRADE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    text_cols = ["netting_set", "trade_id", "asset_class", "hedging_set", "risk_entity",
                 "rating", "instrument", "direction", "option_type", "option_position"]
    df[text_cols] = df[text_cols].fillna("")
    df["is_index"] = df["is_index"].fillna(False).astype(bool)
    return df[TRADE_COLUMNS]


def bcbs_example_1() -> tuple[pd.DataFrame, list[NettingSet]]:
    """BCBS 279, Annex 4a, Example 1 (amounts in USD thousands).

    Two USD interest rate swaps and one purchased EUR receiver swaption,
    unmargined. Published result: EAD = 1.4 x (60 + 347) = 569.
    """
    trades = _trades([
        dict(netting_set="BCBS-EX1", trade_id="IRS-10Y-PAYFIX", asset_class="IR",
             hedging_set="USD", instrument="LINEAR", direction="LONG",
             notional=10_000, mtm=30, start_years=0, end_years=10, maturity_years=10),
        dict(netting_set="BCBS-EX1", trade_id="IRS-4Y-RECFIX", asset_class="IR",
             hedging_set="USD", instrument="LINEAR", direction="SHORT",
             notional=10_000, mtm=-20, start_years=0, end_years=4, maturity_years=4),
        dict(netting_set="BCBS-EX1", trade_id="SWPN-1Y10Y-REC", asset_class="IR",
             hedging_set="EUR", instrument="OPTION", option_type="PUT",
             option_position="BOUGHT", underlying_price=0.06, strike=0.05,
             lambda_shift=0.0, notional=5_000, mtm=50, start_years=1, end_years=11,
             maturity_years=11, exercise_years=1),
    ])
    return trades, [NettingSet(name="BCBS-EX1", margined=False)]


def multi_asset_example() -> tuple[pd.DataFrame, list[NettingSet]]:
    """Illustrative multi-asset portfolio over two netting sets (EUR thousands)."""
    trades = _trades([
        # Netting set A - corporate client, unmargined
        dict(netting_set="CORP-A", trade_id="A-IRS-7Y", asset_class="IR", hedging_set="EUR",
             instrument="LINEAR", direction="LONG", notional=25_000, mtm=410,
             start_years=0, end_years=7, maturity_years=7),
        dict(netting_set="CORP-A", trade_id="A-IRS-2Y", asset_class="IR", hedging_set="EUR",
             instrument="LINEAR", direction="SHORT", notional=15_000, mtm=-95,
             start_years=0, end_years=2, maturity_years=2),
        dict(netting_set="CORP-A", trade_id="A-FXFWD-USD", asset_class="FX", hedging_set="EUR/USD",
             instrument="LINEAR", direction="SHORT", notional=12_000, mtm=-180,
             end_years=0.75, maturity_years=0.75),
        dict(netting_set="CORP-A", trade_id="A-CAP-5Y", asset_class="IR", hedging_set="EUR",
             instrument="OPTION", option_type="CALL", option_position="SOLD",
             underlying_price=0.024, strike=0.03, lambda_shift=0.01, notional=10_000, mtm=-60,
             start_years=0, end_years=5, maturity_years=5, exercise_years=4.75),
        # Netting set B - bank counterparty, margined (CSA)
        dict(netting_set="BANK-B", trade_id="B-IRS-10Y", asset_class="IR", hedging_set="EUR",
             instrument="LINEAR", direction="SHORT", notional=50_000, mtm=-1_200,
             start_years=0, end_years=10, maturity_years=10),
        dict(netting_set="BANK-B", trade_id="B-IRS-USD-3Y", asset_class="IR", hedging_set="USD",
             instrument="LINEAR", direction="LONG", notional=30_000, mtm=350,
             start_years=0, end_years=3, maturity_years=3),
        dict(netting_set="BANK-B", trade_id="B-CDS-FIRM-X", asset_class="CREDIT",
             risk_entity="Firm X", rating="BBB", is_index=False, instrument="LINEAR",
             direction="LONG", notional=10_000, mtm=120, start_years=0, end_years=5,
             maturity_years=5),
        dict(netting_set="BANK-B", trade_id="B-CDX-IG", asset_class="CREDIT",
             risk_entity="CDX IG", rating="IG", is_index=True, instrument="LINEAR",
             direction="SHORT", notional=20_000, mtm=-40, start_years=0, end_years=5,
             maturity_years=5),
        dict(netting_set="BANK-B", trade_id="B-EQ-OPT", asset_class="EQUITY",
             risk_entity="EURO STOXX 50", is_index=True, instrument="OPTION",
             option_type="PUT", option_position="BOUGHT", underlying_price=5_000,
             strike=4_600, notional=8_000, mtm=210, maturity_years=1, exercise_years=1),
        dict(netting_set="BANK-B", trade_id="B-BRENT-SWAP", asset_class="COMMODITY",
             hedging_set="ENERGY", risk_entity="Crude oil", instrument="LINEAR",
             direction="LONG", notional=6_000, mtm=90, maturity_years=1.5),
        dict(netting_set="BANK-B", trade_id="B-POWER-FWD", asset_class="COMMODITY",
             hedging_set="ENERGY", risk_entity="Electricity", instrument="LINEAR",
             direction="SHORT", notional=3_000, mtm=-25, maturity_years=0.5),
    ])
    netting_sets = [
        NettingSet(name="CORP-A", margined=False, collateral=0.0),
        NettingSet(name="BANK-B", margined=True, collateral=-400.0, threshold=0.0,
                   mta=50.0, nica=0.0, mpor_days=10),
    ]
    return trades, netting_sets


def sample_facilities() -> pd.DataFrame:
    """Illustrative loans and off-balance-sheet items (EUR thousands)."""
    return pd.DataFrame([
        dict(facility_id="F001", counterparty="Corporate A", product_type="Undrawn committed credit facility",
             drawn=12_000, undrawn=8_000, provisions=0, original_maturity_years=3, risk_weight=1.00),
        dict(facility_id="F002", counterparty="Corporate A", product_type="Undrawn committed credit facility",
             drawn=0, undrawn=5_000, provisions=0, original_maturity_years=1, risk_weight=1.00),
        dict(facility_id="F003", counterparty="SME B", product_type="Unconditionally cancellable commitment (UCC)",
             drawn=1_500, undrawn=2_500, provisions=0, original_maturity_years=1, risk_weight=0.85),
        dict(facility_id="F004", counterparty="Corporate C", product_type="Performance bond / bid bond / warranty",
             drawn=0, undrawn=4_000, provisions=0, original_maturity_years=2, risk_weight=1.00),
        dict(facility_id="F005", counterparty="Corporate C", product_type="Credit substitute guarantee",
             drawn=0, undrawn=3_000, provisions=0, original_maturity_years=5, risk_weight=1.00),
        dict(facility_id="F006", counterparty="Importer D", product_type="Trade finance documentary credit",
             drawn=0, undrawn=2_000, provisions=0, original_maturity_years=0.5, risk_weight=1.00),
        dict(facility_id="F007", counterparty="Retail portfolio", product_type="Unconditionally cancellable commitment (UCC)",
             drawn=6_000, undrawn=14_000, provisions=150, original_maturity_years=1, risk_weight=0.75),
        dict(facility_id="F008", counterparty="Corporate E", product_type="Undrawn committed credit facility",
             drawn=4_000, undrawn=1_000, provisions=4_500, original_maturity_years=4, risk_weight=1.50),
    ])
