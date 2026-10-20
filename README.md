# EAD Calculator – SA-CCF (CRR3) & SA-CCR

An interactive Streamlit application that computes **regulatory Exposure at Default (EAD)** under the EU Capital Requirements Regulation:

| Module | Scope | Legal basis |
|---|---|---|
| **SA-CCF** | Loans and off-balance-sheet items: standardised credit conversion factors, provisions, RWA, CRR2 vs CRR3 impact, UCC phase-in | CRR3 Art. 111, Annex I, Art. 495d |
| **SA-CCR** | Derivatives: replacement cost, PFE add-ons for IR, FX, credit, equity and commodity, supervisory delta for options, margined and unmargined netting sets | CRR Art. 274–280f, BCBS 279 |

Every intermediate quantity (supervisory duration, adjusted notional, delta, maturity factor, hedging-set add-ons, multiplier, RC, PFE) is displayed so that the result can be traced back to the regulation.

**Live app:** _add your Streamlit Community Cloud URL here once deployed_

---

## Validation

The SA-CCR engine reproduces the published **BCBS 279 Annex 4a – Example 1** (two USD swaps and a EUR receiver swaption, unmargined):

| | BCBS 279 | This tool |
|---|---|---|
| Replacement cost | 60 | 60 |
| Aggregate add-on | 347 | 346.8 |
| Swaption supervisory delta | −0.27 | −0.269 |
| **EAD** | **569** | **569.5** |

The test suite (`tests/`) also checks the CRR3 buckets, the UCC transitional schedule, provision handling, IR maturity buckets, the multiplier floor, the margined EAD cap, option deltas (put-call parity, negative-rate shift) and input validation.

```bash
pip install -r requirements-dev.txt
pytest
```

## Methodology in brief

**Off-balance sheet (SA-CCF)**

```
EAD = max(Drawn − Provisions, 0) + CCF × max(Undrawn − residual Provisions, 0)
```

CRR3 buckets: 100% / 50% / 40% / 20% / 10%. Unconditionally cancellable commitments are phased in from 0% (until end-2029) to 10% (from 2033).

**Derivatives (SA-CCR)**

```
EAD = α × (RC + PFE),   α = 1.4
RC  = max(V − C, 0)                         (unmargined)
RC  = max(V − C, TH + MTA − NICA, 0)        (margined, EAD capped at the unmargined EAD)
PFE = multiplier × AddOn_aggregate
multiplier = min{1 ; 0.05 + 0.95 × exp[(V − C) / (2 × 0.95 × AddOn_aggregate)]}
```

Add-ons are computed per hedging set (IR by currency with three maturity buckets, FX by currency pair, single-factor model for credit, equity and commodity) and summed across asset classes. Full formulas and parameter tables are in the app's *Methodology & references* tab.

## Project structure

```
├── app.py                  # Streamlit interface
├── ead/
│   ├── parameters.py       # All regulatory parameters in one place
│   ├── sa_ccf.py           # SA-CCF engine
│   ├── sa_ccr.py           # SA-CCR engine
│   └── samples.py          # Sample portfolios (incl. BCBS 279 Example 1)
├── data/                   # CSV templates / sample inputs
├── tests/                  # pytest suite
└── requirements.txt
```

## Run locally

```bash
git clone https://github.com/ldenarbonne/ead-calculator.git
cd ead-calculator
pip install -r requirements.txt
streamlit run app.py
```

## Input format

Data can be typed directly in the app or imported from CSV. The files in `data/` show the expected columns:

- `sample_facilities.csv` – `facility_id, counterparty, product_type, drawn, undrawn, provisions, original_maturity_years, risk_weight`
- `sample_trades.csv` – one row per derivative: `netting_set, trade_id, asset_class, hedging_set, risk_entity, rating, is_index, instrument, direction, option_type, option_position, underlying_price, strike, lambda_shift, notional, mtm, start_years, end_years, maturity_years, exercise_years`

Netting-set collateral terms (C, TH, MTA, NICA, MPOR, α) are entered in the app.

## Limitations

Educational and demonstration tool, not a production capital engine. Not implemented in this version: CDO tranches, basis and volatility transactions, the simplified SA-CCR and the original exposure method, collateral haircut calculation (C must be entered net of haircuts), and IRB own estimates of CCF (planned for a next version, with synthetic default data).

## References

- Regulation (EU) 2024/1623 (CRR3) – [EUR-Lex](https://eur-lex.europa.eu/eli/reg/2024/1623/oj/eng)
- BCBS 279, *The standardised approach for measuring counterparty credit risk exposures* (2014) – [BIS](https://www.bis.org/publ/bcbs279.pdf)
- Basel Framework, CRE52 – [BIS](https://www.bis.org/basel_framework/chapter/CRE/52.htm)
- EBA final draft RTS on off-balance-sheet items under CRR3 Art. 111(8) (EBA/RTS/2025/06)
- EBA final draft RTS amending the SA-CCR RTS (EBA/RTS/2024/16)

## Author

**Laurent Motais de Narbonne** – Senior Risk Manager
[LinkedIn](https://www.linkedin.com/in/laurent-motais-de-narbonne/) · [Publications](https://ldenarbonne.github.io/) · [GitHub](https://github.com/ldenarbonne)

Licensed under the MIT License.
