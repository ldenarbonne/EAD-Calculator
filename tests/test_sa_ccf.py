from datetime import date

import pandas as pd
import pytest

from ead.parameters import ucc_transitional_factor
from ead.sa_ccf import compute_sa_ccf, crr2_ccf, crr3_ccf
from ead.samples import sample_facilities


@pytest.mark.parametrize("d, expected", [
    (date(2026, 10, 20), 0.0),
    (date(2029, 12, 31), 0.0),
    (date(2030, 6, 30), 0.25),
    (date(2031, 6, 30), 0.50),
    (date(2032, 6, 30), 0.75),
    (date(2033, 1, 1), 1.0),
])
def test_ucc_transitional_schedule(d, expected):
    assert ucc_transitional_factor(d) == expected


def test_crr3_buckets():
    d = date(2034, 1, 1)
    assert crr3_ccf("Credit substitute guarantee", d) == (1, 1.0)
    assert crr3_ccf("Performance bond / bid bond / warranty", d) == (2, 0.5)
    assert crr3_ccf("Undrawn committed credit facility", d) == (3, 0.4)
    assert crr3_ccf("Trade finance documentary credit", d) == (4, 0.2)
    assert crr3_ccf("Unconditionally cancellable commitment (UCC)", d) == (5, pytest.approx(0.1))


def test_crr2_commitment_depends_on_maturity():
    assert crr2_ccf("Undrawn committed credit facility", 1.0) == 0.20
    assert crr2_ccf("Undrawn committed credit facility", 1.5) == 0.50


def test_ead_formula_and_provisions():
    df = pd.DataFrame([
        dict(facility_id="A", product_type="Undrawn committed credit facility",
             drawn=100, undrawn=50, provisions=0),
        dict(facility_id="B", product_type="Undrawn committed credit facility",
             drawn=100, undrawn=50, provisions=120),  # 20 deducted from undrawn
    ])
    res = compute_sa_ccf(df, date(2026, 1, 1)).set_index("facility_id")
    assert res.loc["A", "ead"] == pytest.approx(100 + 0.4 * 50)
    assert res.loc["B", "ead"] == pytest.approx(0 + 0.4 * 30)


def test_unknown_product_raises():
    df = pd.DataFrame([dict(facility_id="X", product_type="Mystery", drawn=1, undrawn=1)])
    with pytest.raises(ValueError, match="Unknown product type"):
        compute_sa_ccf(df, date(2026, 1, 1))


def test_sample_runs_and_rwa():
    res = compute_sa_ccf(sample_facilities(), date(2026, 10, 20))
    assert (res["ead"] >= 0).all()
    assert res["rwa"].sum() == pytest.approx((res["ead"] * res["risk_weight"]).sum())
