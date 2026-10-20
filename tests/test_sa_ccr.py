import math

import numpy as np
import pandas as pd
import pytest

from ead.sa_ccr import (
    NettingSet,
    compute_netting_set,
    ir_bucket,
    maturity_factor,
    option_delta,
    pfe_multiplier,
    supervisory_duration,
)
from ead.samples import bcbs_example_1, multi_asset_example


# --- Benchmark against the published BCBS 279 worked example ---------------

def test_bcbs_example_1_reproduces_published_ead():
    trades, netting_sets = bcbs_example_1()
    res = compute_netting_set(trades, netting_sets[0])
    assert res.replacement_cost == pytest.approx(60.0)
    assert res.multiplier == pytest.approx(1.0)
    assert res.addon_aggregate == pytest.approx(347, abs=0.5)
    assert res.ead == pytest.approx(569, abs=1.0)


def test_bcbs_example_1_intermediate_values():
    trades, netting_sets = bcbs_example_1()
    res = compute_netting_set(trades, netting_sets[0])
    tl = res.trades.set_index("trade_id")
    # Adjusted notionals published in BCBS 279 Annex 4a (USD thousands)
    assert tl.loc["IRS-10Y-PAYFIX", "adjusted_notional"] == pytest.approx(78_694, abs=1)
    assert tl.loc["IRS-4Y-RECFIX", "adjusted_notional"] == pytest.approx(36_254, abs=1)
    assert tl.loc["SWPN-1Y10Y-REC", "adjusted_notional"] == pytest.approx(37_428, abs=1)
    assert tl.loc["SWPN-1Y10Y-REC", "delta"] == pytest.approx(-0.27, abs=0.005)


# --- Building blocks -------------------------------------------------------

def test_supervisory_duration():
    assert supervisory_duration(0, 10) == pytest.approx((1 - math.exp(-0.5)) / 0.05)
    assert supervisory_duration(0, 0) == 0.0
    assert supervisory_duration(-1, 2) == pytest.approx(supervisory_duration(0, 2))


def test_maturity_factor():
    assert maturity_factor(5, margined=False) == 1.0
    assert maturity_factor(0.25, margined=False) == pytest.approx(0.5)
    assert maturity_factor(0.001, margined=False) == pytest.approx(math.sqrt(10 / 250))
    assert maturity_factor(5, margined=True, mpor_days=10) == pytest.approx(1.5 * math.sqrt(10 / 250))


def test_option_delta_signs_and_parity():
    args = dict(price=100, strike=100, vol=0.2, expiry=1)
    call_b = option_delta("CALL", "BOUGHT", **args)
    put_b = option_delta("PUT", "BOUGHT", **args)
    assert 0 < call_b < 1 and -1 < put_b < 0
    assert option_delta("CALL", "SOLD", **args) == pytest.approx(-call_b)
    assert call_b - put_b == pytest.approx(1.0)  # N(d1) + N(-d1) = 1


def test_option_delta_shift_allows_negative_rates():
    delta = option_delta("CALL", "BOUGHT", price=-0.002, strike=0.001, vol=0.5,
                         expiry=2, shift=0.01)
    assert 0 < delta < 1
    with pytest.raises(ValueError):
        option_delta("CALL", "BOUGHT", price=-0.002, strike=0.001, vol=0.5, expiry=2)


def test_ir_bucket_boundaries():
    assert ir_bucket(0.5) == 1
    assert ir_bucket(1.0) == 2
    assert ir_bucket(5.0) == 2
    assert ir_bucket(5.01) == 3


def test_multiplier_floor_and_cap():
    assert pfe_multiplier(10, 0, 100) == 1.0
    assert pfe_multiplier(-1e9, 0, 100) == pytest.approx(0.05)
    assert 0.05 < pfe_multiplier(-50, 0, 100) < 1.0


# --- Netting-set logic -----------------------------------------------------

def _single_trade(**kw):
    base = dict(netting_set="NS", trade_id="T1", asset_class="IR", hedging_set="EUR",
                instrument="LINEAR", direction="LONG", notional=1_000_000, mtm=0.0,
                start_years=0, end_years=5, maturity_years=5)
    base.update(kw)
    return pd.DataFrame([base])


def test_offsetting_trades_in_same_bucket_cancel():
    trades = pd.concat([_single_trade(trade_id="T1"),
                        _single_trade(trade_id="T2", direction="SHORT")])
    res = compute_netting_set(trades, NettingSet(name="NS"))
    assert res.addon_aggregate == pytest.approx(0.0, abs=1e-9)


def test_margined_rc_floor_and_cap():
    trades = _single_trade(mtm=0.0)
    ns = NettingSet(name="NS", margined=True, collateral=0.0, threshold=1e9, mta=0, nica=0)
    res = compute_netting_set(trades, ns)
    # A huge threshold makes the margined EAD exceed the unmargined one: cap applies
    assert res.capped_at_unmargined
    assert res.ead == pytest.approx(res.ead_unmargined)


def test_fx_hedging_sets_do_not_offset_across_pairs():
    trades = pd.concat([
        _single_trade(trade_id="F1", asset_class="FX", hedging_set="EUR/USD",
                      direction="LONG", maturity_years=1, end_years=1),
        _single_trade(trade_id="F2", asset_class="FX", hedging_set="EUR/GBP",
                      direction="SHORT", maturity_years=1, end_years=1),
    ])
    res = compute_netting_set(trades, NettingSet(name="NS"))
    assert res.addon_aggregate == pytest.approx(2 * 0.04 * 1_000_000)


def test_credit_aggregation_single_entity():
    trades = _single_trade(asset_class="CREDIT", hedging_set="", risk_entity="X",
                           rating="BBB", is_index=False)
    res = compute_netting_set(trades, NettingSet(name="NS"))
    expected = 0.0054 * 1_000_000 * supervisory_duration(0, 5)
    # one entity: sqrt((rho A)^2 + (1-rho^2) A^2) = A
    assert res.addon_aggregate == pytest.approx(expected)


def test_commodity_electricity_factor():
    trades = _single_trade(asset_class="COMMODITY", hedging_set="ENERGY",
                           risk_entity="Electricity", maturity_years=1, end_years=1)
    res = compute_netting_set(trades, NettingSet(name="NS"))
    assert res.addon_aggregate == pytest.approx(0.40 * 1_000_000)


def test_invalid_input_is_reported():
    with pytest.raises(ValueError, match="direction must be LONG or SHORT"):
        compute_netting_set(_single_trade(direction="UP"), NettingSet(name="NS"))


def test_multi_asset_example_runs():
    trades, netting_sets = multi_asset_example()
    for ns in netting_sets:
        res = compute_netting_set(trades[trades["netting_set"] == ns.name], ns)
        assert res.ead > 0
        assert np.isfinite(res.ead)
