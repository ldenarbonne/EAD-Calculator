"""EAD Calculator - Streamlit application.

Exposure at Default under the EU Capital Requirements Regulation:
  * Loans and off-balance-sheet items - Standardised CCFs (CRR3, Art. 111)
  * Derivatives - Standardised Approach for Counterparty Credit Risk (SA-CCR)

Run locally with:  streamlit run app.py
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ead import __version__
from ead.parameters import (
    ALPHA_DEFAULT,
    COMMODITY_HEDGING_SETS,
    CREDIT_RATINGS,
    CRR3_BUCKET_CCF,
    PRODUCT_CATALOGUE,
    ucc_transitional_factor,
)
from ead.sa_ccf import compute_sa_ccf, summarise_by_bucket
from ead.sa_ccr import (
    TRADE_COLUMNS,
    NettingSet,
    compute_portfolio,
    ead_collateral_sensitivity,
)
from ead.samples import bcbs_example_1, multi_asset_example, sample_facilities

# ---------------------------------------------------------------------------
# Page set-up
# ---------------------------------------------------------------------------

st.set_page_config(page_title="EAD Calculator", page_icon="📐", layout="wide")

SERIES_1 = "#2a78d6"   # categorical slot 1 (blue)
SERIES_2 = "#eb6834"   # categorical slot 2 (orange)
GRID = "rgba(128,128,128,0.18)"

FACILITY_COLUMNS = ["facility_id", "counterparty", "product_type", "drawn", "undrawn",
                    "provisions", "original_maturity_years", "risk_weight"]
NETTING_COLUMNS = ["name", "margined", "collateral", "threshold", "mta", "nica",
                   "mpor_days", "alpha"]


def _style_figure(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0),
        hoverlabel=dict(namelength=-1), font=dict(size=13),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickformat=",.0f")
    return fig


def _fmt(x: float | None, decimals: int = 0) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:,.{decimals}f}"


def _netting_sets_to_df(netting_sets: list[NettingSet]) -> pd.DataFrame:
    return pd.DataFrame([{c: getattr(ns, c) for c in NETTING_COLUMNS} for ns in netting_sets])


def _df_to_netting_sets(df: pd.DataFrame) -> list[NettingSet]:
    out = []
    for _, r in df.dropna(subset=["name"]).iterrows():
        name = str(r["name"]).strip()
        if not name:
            continue
        out.append(NettingSet(
            name=name,
            margined=bool(r.get("margined", False)),
            collateral=float(np.nan_to_num(r.get("collateral", 0.0))),
            threshold=float(np.nan_to_num(r.get("threshold", 0.0))),
            mta=float(np.nan_to_num(r.get("mta", 0.0))),
            nica=float(np.nan_to_num(r.get("nica", 0.0))),
            mpor_days=float(np.nan_to_num(r.get("mpor_days", 10.0), nan=10.0)) or 10.0,
            alpha=float(np.nan_to_num(r.get("alpha", ALPHA_DEFAULT), nan=ALPHA_DEFAULT)) or ALPHA_DEFAULT,
        ))
    return out


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    if "facilities" not in st.session_state:
        st.session_state.facilities = sample_facilities()[FACILITY_COLUMNS]
    if "trades" not in st.session_state:
        trades, netting_sets = bcbs_example_1()
        st.session_state.trades = trades
        st.session_state.netting_sets = _netting_sets_to_df(netting_sets)
    st.session_state.setdefault("ccf_editor_version", 0)
    st.session_state.setdefault("ccr_editor_version", 0)


_init_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    reporting_date = st.date_input("Reporting date", value=date.today(),
                                   min_value=date(2025, 1, 1), max_value=date(2040, 12, 31),
                                   help="Drives the CRR3 transitional factor for "
                                        "unconditionally cancellable commitments (Art. 495d).")
    unit = st.text_input("Amount unit (display only)", value="EUR thousands")
    st.caption(f"UCC transitional factor at this date: "
               f"**{ucc_transitional_factor(reporting_date):.0%}** of the 10% CCF.")
    st.divider()
    st.markdown(
        "**Author:** Laurent Motais de Narbonne  \n"
        "[LinkedIn](https://www.linkedin.com/in/laurent-motais-de-narbonne/) · "
        "[Publications](https://ldenarbonne.github.io/) · "
        "[GitHub](https://github.com/ldenarbonne)"
    )
    st.caption(f"Version {__version__}. Educational tool – not a substitute for "
               "the legal text or for an institution's validated implementation.")

st.title("Exposure at Default (EAD) Calculator")
st.markdown(
    "Regulatory EAD under the EU Capital Requirements Regulation: **standardised credit "
    "conversion factors (CRR3)** for loans and off-balance-sheet items, and the **SA-CCR** "
    "for derivatives. Every intermediate step is displayed so that results can be traced "
    "back to the regulation."
)

tab_ccf, tab_ccr, tab_method = st.tabs(
    ["Loans & off-balance sheet (SA-CCF)", "Derivatives (SA-CCR)", "Methodology & references"]
)

# ---------------------------------------------------------------------------
# Tab 1 - SA-CCF
# ---------------------------------------------------------------------------

with tab_ccf:
    st.subheader("Standardised credit conversion factors – CRR3")
    st.markdown(
        "EAD = (drawn − provisions) + CCF × (undrawn − remaining provisions). "
        "Provisions (specific credit risk adjustments) are deducted from the drawn amount "
        "first, then from the off-balance-sheet nominal (Art. 111(1)). "
        "The CRR2 column shows the pre-reform CCF for an impact comparison."
    )

    c1, c2, c3 = st.columns([1, 1, 2])
    if c1.button("Load sample portfolio", key="ccf_sample"):
        st.session_state.facilities = sample_facilities()[FACILITY_COLUMNS]
        st.session_state.ccf_editor_version += 1
    if c2.button("Clear table", key="ccf_clear"):
        st.session_state.facilities = pd.DataFrame(columns=FACILITY_COLUMNS)
        st.session_state.ccf_editor_version += 1
    upload = c3.file_uploader("Import CSV", type="csv", key="ccf_upload",
                              label_visibility="collapsed")
    if upload is not None and st.session_state.get("ccf_last_upload") != upload.file_id:
        try:
            imported = pd.read_csv(upload)
            st.session_state.facilities = imported.reindex(columns=FACILITY_COLUMNS)
            st.session_state.ccf_last_upload = upload.file_id
            st.session_state.ccf_editor_version += 1
        except Exception as exc:  # noqa: BLE001 - surface any parsing problem
            st.error(f"Could not read the CSV file: {exc}")

    facilities = st.data_editor(
        st.session_state.facilities,
        key=f"ccf_editor_{st.session_state.ccf_editor_version}",
        num_rows="dynamic",
        width="stretch",
        column_config={
            "facility_id": st.column_config.TextColumn("Facility ID", required=True),
            "counterparty": st.column_config.TextColumn("Counterparty"),
            "product_type": st.column_config.SelectboxColumn(
                "Product type (CRR3 Annex I)", options=list(PRODUCT_CATALOGUE), required=True,
                width="large"),
            "drawn": st.column_config.NumberColumn("Drawn", min_value=0.0, format="localized"),
            "undrawn": st.column_config.NumberColumn("Undrawn / nominal", min_value=0.0,
                                                     format="localized"),
            "provisions": st.column_config.NumberColumn("Provisions", min_value=0.0,
                                                        format="localized"),
            "original_maturity_years": st.column_config.NumberColumn(
                "Orig. maturity (y)", min_value=0.0, format="%.2f",
                help="Used only for the CRR2 comparison (20% if ≤ 1 year, 50% otherwise)."),
            "risk_weight": st.column_config.NumberColumn(
                "Risk weight", min_value=0.0, max_value=15.0, format="percent",
                help="Decimal (1.00 = 100%). Used to derive RWA."),
        },
    )

    try:
        res = compute_sa_ccf(facilities.dropna(how="all"), reporting_date)
    except ValueError as exc:
        st.error(str(exc))
        res = None

    if res is not None and not res.empty:
        total_ead, total_crr2 = res["ead"].sum(), res["ead_crr2"].sum()
        impact = (total_ead / total_crr2 - 1) if total_crr2 else np.nan
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"EAD – CRR3 ({unit})", _fmt(total_ead))
        m2.metric(f"EAD – CRR2 ({unit})", _fmt(total_crr2))
        m3.metric("CRR3 vs CRR2", "–" if np.isnan(impact) else f"{impact:+.1%}")
        m4.metric(f"RWA – CRR3 ({unit})", _fmt(res["rwa"].sum()))

        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Results by facility**")
            st.dataframe(
                res[["facility_id", "counterparty", "product_type", "bucket", "ccf_crr3",
                     "ead_on_balance", "ead_off_balance", "ead", "rwa", "ccf_crr2", "ead_crr2"]],
                hide_index=True, width="stretch",
                column_config={
                    "facility_id": "Facility", "counterparty": "Counterparty",
                    "product_type": "Product", "bucket": "Bucket",
                    "ccf_crr3": st.column_config.NumberColumn("CCF CRR3", format="percent"),
                    "ead_on_balance": st.column_config.NumberColumn("On-balance", format="localized"),
                    "ead_off_balance": st.column_config.NumberColumn("CCF × undrawn", format="localized"),
                    "ead": st.column_config.NumberColumn("EAD", format="localized"),
                    "rwa": st.column_config.NumberColumn("RWA", format="localized"),
                    "ccf_crr2": st.column_config.NumberColumn("CCF CRR2", format="percent"),
                    "ead_crr2": st.column_config.NumberColumn("EAD CRR2", format="localized"),
                },
            )
        with right:
            summary = summarise_by_bucket(res)
            labels = [f"B{b} · {CRR3_BUCKET_CCF[b]:.0%}" for b in summary["bucket"]]
            fig = go.Figure()
            fig.add_bar(name="CRR2", x=labels, y=summary["ead_crr2"], marker_color=SERIES_2,
                        hovertemplate="%{x}<br>CRR2 EAD: %{y:,.0f}<extra></extra>")
            fig.add_bar(name="CRR3", x=labels, y=summary["ead"], marker_color=SERIES_1,
                        hovertemplate="%{x}<br>CRR3 EAD: %{y:,.0f}<extra></extra>")
            fig.update_traces(marker_line_width=0)
            fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08,
                              title=dict(text=f"EAD by CRR3 bucket ({unit})", x=0, font=dict(size=14)))
            st.plotly_chart(_style_figure(fig), width="stretch")

        if ucc_transitional_factor(reporting_date) < 1 and (res["bucket"] == 5).any():
            st.info(
                f"Unconditionally cancellable commitments receive "
                f"{CRR3_BUCKET_CCF[5] * ucc_transitional_factor(reporting_date):.1%} at this "
                "reporting date (CRR3 Art. 495d phase-in: 0% to end-2029, then 2.5%, 5%, 7.5%, "
                "and 10% from 2033). Change the reporting date in the sidebar to see the "
                "fully phased-in impact."
            )

        st.download_button("Download results (CSV)", res.to_csv(index=False).encode(),
                           file_name="sa_ccf_results.csv", mime="text/csv")
    elif res is not None:
        st.info("Add at least one facility to compute the EAD.")

    with st.expander("CRR3 product catalogue used for bucket mapping"):
        st.dataframe(
            pd.DataFrame([
                {"Product type": k, "Bucket": v[0], "CCF CRR3": CRR3_BUCKET_CCF[v[0]],
                 "CCF CRR2": "20% / 50%" if v[1] is None else f"{v[1]:.0%}", "Description": v[2]}
                for k, v in PRODUCT_CATALOGUE.items()
            ]),
            hide_index=True, width="stretch",
            column_config={"CCF CRR3": st.column_config.NumberColumn(format="percent")},
        )
        st.caption("The mapping is a simplified reading of CRR3 Annex I and of the EBA RTS on "
                   "off-balance-sheet items. Always check the legal text for borderline products.")

# ---------------------------------------------------------------------------
# Tab 2 - SA-CCR
# ---------------------------------------------------------------------------

with tab_ccr:
    st.subheader("Standardised Approach for Counterparty Credit Risk")
    st.markdown("EAD = α × (RC + PFE), with PFE = multiplier × AddOn<sup>aggregate</sup> "
                "and α = 1.4 (CRR Art. 274).", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([1.3, 1.3, 0.9, 2])
    if c1.button("BCBS 279 – Example 1", key="ccr_bcbs",
                 help="Published benchmark: EAD = 1.4 × (60 + 347) = 569 (USD thousands)."):
        t, ns = bcbs_example_1()
        st.session_state.trades, st.session_state.netting_sets = t, _netting_sets_to_df(ns)
        st.session_state.ccr_editor_version += 1
    if c2.button("Multi-asset sample", key="ccr_multi"):
        t, ns = multi_asset_example()
        st.session_state.trades, st.session_state.netting_sets = t, _netting_sets_to_df(ns)
        st.session_state.ccr_editor_version += 1
    if c3.button("Clear", key="ccr_clear"):
        st.session_state.trades = pd.DataFrame(columns=TRADE_COLUMNS)
        st.session_state.netting_sets = pd.DataFrame(columns=NETTING_COLUMNS)
        st.session_state.ccr_editor_version += 1
    upload_t = c4.file_uploader("Import trades CSV", type="csv", key="ccr_upload",
                                label_visibility="collapsed")
    if upload_t is not None and st.session_state.get("ccr_last_upload") != upload_t.file_id:
        try:
            st.session_state.trades = pd.read_csv(upload_t).reindex(columns=TRADE_COLUMNS)
            st.session_state.ccr_last_upload = upload_t.file_id
            st.session_state.ccr_editor_version += 1
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read the CSV file: {exc}")

    st.markdown("**Trades** – one row per trade. Use *direction* (LONG/SHORT in the primary "
                "risk factor) for linear trades, e.g. pay-fixed swap = LONG; use *option_type* "
                "and *option_position* for options. Times are in years.")
    v = st.session_state.ccr_editor_version
    trades = st.data_editor(
        st.session_state.trades,
        key=f"ccr_trades_{v}",
        num_rows="dynamic",
        width="stretch",
        column_config={
            "netting_set": st.column_config.TextColumn("Netting set", required=True),
            "trade_id": st.column_config.TextColumn("Trade ID", required=True),
            "asset_class": st.column_config.SelectboxColumn(
                "Asset class", options=["IR", "FX", "CREDIT", "EQUITY", "COMMODITY"], required=True),
            "hedging_set": st.column_config.TextColumn(
                "Hedging set", help="IR: currency · FX: currency pair · COMMODITY: "
                + ", ".join(COMMODITY_HEDGING_SETS) + " · not used for CREDIT / EQUITY."),
            "risk_entity": st.column_config.TextColumn(
                "Risk entity", help="CREDIT/EQUITY: reference entity or index · COMMODITY: "
                                    "commodity type (write 'Electricity' for the 40% factor)."),
            "rating": st.column_config.SelectboxColumn(
                "Rating", options=list(CREDIT_RATINGS),
                help="Single-name credit: AAA…CCC · credit index: IG or SG."),
            "is_index": st.column_config.CheckboxColumn("Index?"),
            "instrument": st.column_config.SelectboxColumn("Instrument", options=["LINEAR", "OPTION"]),
            "direction": st.column_config.SelectboxColumn("Direction", options=["LONG", "SHORT"]),
            "option_type": st.column_config.SelectboxColumn("Option type", options=["CALL", "PUT"]),
            "option_position": st.column_config.SelectboxColumn("Position", options=["BOUGHT", "SOLD"]),
            "underlying_price": st.column_config.NumberColumn("Underlying P", format="%.4f"),
            "strike": st.column_config.NumberColumn("Strike K", format="%.4f"),
            "lambda_shift": st.column_config.NumberColumn(
                "λ shift", format="%.4f", help="Shift for negative rates / prices (EBA RTS)."),
            "notional": st.column_config.NumberColumn("Notional", min_value=0.0, format="localized"),
            "mtm": st.column_config.NumberColumn("MtM", format="localized"),
            "start_years": st.column_config.NumberColumn("S (y)", format="%.2f"),
            "end_years": st.column_config.NumberColumn("E (y)", format="%.2f"),
            "maturity_years": st.column_config.NumberColumn("M (y)", format="%.2f"),
            "exercise_years": st.column_config.NumberColumn("T (y)", format="%.2f"),
        },
    )

    st.markdown("**Netting sets** – collateral terms. C > 0 means net collateral received.")
    ns_df = st.data_editor(
        st.session_state.netting_sets,
        key=f"ccr_ns_{v}",
        num_rows="dynamic",
        width="stretch",
        column_config={
            "name": st.column_config.TextColumn("Netting set", required=True),
            "margined": st.column_config.CheckboxColumn("Margined (VM)?"),
            "collateral": st.column_config.NumberColumn("C – net collateral", format="localized"),
            "threshold": st.column_config.NumberColumn("TH – threshold", format="localized"),
            "mta": st.column_config.NumberColumn("MTA", format="localized"),
            "nica": st.column_config.NumberColumn("NICA", format="localized"),
            "mpor_days": st.column_config.NumberColumn("MPOR (days)", min_value=1, format="%.0f"),
            "alpha": st.column_config.NumberColumn("α", min_value=0.0, format="%.2f"),
        },
    )

    trades_clean = trades.dropna(how="all")
    results = []
    if not trades_clean.empty:
        try:
            results = compute_portfolio(trades_clean, _df_to_netting_sets(ns_df))
        except ValueError as exc:
            st.error(str(exc))

    if results:
        summary = pd.DataFrame([r.summary() for r in results])
        num_cols = ["V (net MtM)", "C (collateral)", "RC", "AddOn aggregate", "PFE", "EAD",
                    "EAD unmargined"]
        summary[num_cols] = summary[num_cols].astype(float).round(0)
        total = summary["EAD"].sum()
        st.markdown(f"#### Portfolio EAD: {_fmt(total)} ({unit})")
        st.dataframe(
            summary, hide_index=True, width="stretch",
            column_config={
                c: st.column_config.NumberColumn(format="localized")
                for c in ["V (net MtM)", "C (collateral)", "RC", "AddOn aggregate", "PFE",
                          "EAD", "EAD unmargined"]
            } | {"multiplier": st.column_config.NumberColumn(format="%.4f")},
        )

        names = [r.netting_set.name for r in results]
        chosen = st.selectbox("Detail for netting set", names)
        r = results[names.index(chosen)]

        m = st.columns(5)
        m[0].metric("Replacement cost (RC)", _fmt(r.replacement_cost))
        m[1].metric("AddOn aggregate", _fmt(r.addon_aggregate))
        m[2].metric("Multiplier", f"{r.multiplier:.4f}")
        m[3].metric("PFE", _fmt(r.pfe))
        m[4].metric("EAD", _fmt(r.ead))
        if r.netting_set.margined:
            st.caption(
                f"Margined netting set – MF = 1.5 × √(MPOR/250). Unmargined EAD = "
                f"{_fmt(r.ead_unmargined)}; the margined EAD is capped at this amount "
                f"({'cap applied' if r.capped_at_unmargined else 'cap not binding'})."
            )

        left, right = st.columns(2)
        with left:
            abc = r.addon_by_class.sort_values("addon", ascending=True)
            fig = go.Figure(go.Bar(
                x=abc["addon"], y=abc["asset_class"], orientation="h", marker_color=SERIES_1,
                text=[_fmt(x) for x in abc["addon"]], textposition="outside",
                hovertemplate="%{y}: %{x:,.0f}<extra></extra>"))
            fig.update_layout(title=dict(text="Add-on by asset class (no diversification "
                                              "across classes)", x=0, font=dict(size=14)))
            fig = _style_figure(fig, height=320)
            fig.update_xaxes(showgrid=True, gridcolor=GRID, tickformat=",.0f",
                             range=[0, float(abc["addon"].max()) * 1.18 if len(abc) else 1])
            fig.update_yaxes(showgrid=False)
            st.plotly_chart(fig, width="stretch")
        with right:
            span = max(abs(r.mtm), r.addon_aggregate, 1.0)
            grid = np.linspace(r.netting_set.collateral - 2 * span,
                               r.netting_set.collateral + 2 * span, 61)
            sens = ead_collateral_sensitivity(
                trades_clean[trades_clean["netting_set"].astype(str).str.strip() == chosen],
                r.netting_set, grid)
            fig2 = go.Figure()
            fig2.add_scatter(x=sens["collateral"], y=sens["ead"], mode="lines", name="EAD",
                             line=dict(color=SERIES_1, width=2),
                             customdata=np.stack([sens["rc"], sens["pfe"], sens["multiplier"]], axis=1),
                             hovertemplate="C = %{x:,.0f}<br>EAD = %{y:,.0f}<br>RC = %{customdata[0]:,.0f}"
                                           "<br>PFE = %{customdata[1]:,.0f}<br>multiplier = "
                                           "%{customdata[2]:.3f}<extra></extra>")
            fig2.add_scatter(x=[r.netting_set.collateral], y=[r.ead], mode="markers",
                             name="Current", marker=dict(size=10, color=SERIES_1,
                                                         line=dict(width=2, color="white")),
                             hovertemplate="Current C = %{x:,.0f}<br>EAD = %{y:,.0f}<extra></extra>",
                             showlegend=False)
            fig2.update_layout(title=dict(text="EAD sensitivity to collateral held (C)", x=0,
                                          font=dict(size=14)), showlegend=False)
            fig2 = _style_figure(fig2, height=320)
            fig2.update_xaxes(title_text="Net collateral C", tickformat=",.0f")
            st.plotly_chart(fig2, width="stretch")

        st.markdown("**Hedging-set add-ons**")
        st.dataframe(r.hedging_sets.round(2), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="localized")
                                    for c in r.hedging_sets.columns
                                    if c not in ("asset_class", "hedging_set", "trades", "entities")})
        st.markdown("**Trade-level calculation**")
        trade_view = r.trades[["trade_id", "asset_class", "hedging_set", "risk_entity",
                               "notional", "supervisory_duration", "adjusted_notional",
                               "delta", "maturity_factor", "ir_bucket", "effective_notional",
                               "supervisory_factor", "trade_addon_signed", "mtm"]]
        st.dataframe(
            trade_view, hide_index=True, width="stretch",
            column_config={
                "notional": st.column_config.NumberColumn("Notional", format="localized"),
                "supervisory_duration": st.column_config.NumberColumn("SD", format="%.4f"),
                "adjusted_notional": st.column_config.NumberColumn("d = N × SD", format="localized"),
                "delta": st.column_config.NumberColumn("δ", format="%.4f"),
                "maturity_factor": st.column_config.NumberColumn("MF", format="%.4f"),
                "ir_bucket": st.column_config.NumberColumn("IR bucket", format="%.0f"),
                "effective_notional": st.column_config.NumberColumn("δ × d × MF", format="localized"),
                "supervisory_factor": st.column_config.NumberColumn("SF", format="percent"),
                "trade_addon_signed": st.column_config.NumberColumn("SF × δ × d × MF", format="localized"),
                "mtm": st.column_config.NumberColumn("MtM", format="localized"),
            },
        )
        all_trades = pd.concat([x.trades for x in results], ignore_index=True)
        d1, d2 = st.columns(2)
        d1.download_button("Download trade-level results (CSV)",
                           all_trades.to_csv(index=False).encode(),
                           file_name="sa_ccr_trades.csv", mime="text/csv")
        d2.download_button("Download netting-set summary (CSV)",
                           summary.to_csv(index=False).encode(),
                           file_name="sa_ccr_summary.csv", mime="text/csv")
    elif trades_clean.empty:
        st.info("Load an example or add trades to compute the EAD.")

# ---------------------------------------------------------------------------
# Tab 3 - Methodology
# ---------------------------------------------------------------------------

with tab_method:
    st.subheader("Methodology")
    st.markdown(r"""
### 1. Loans and off-balance-sheet items – standardised CCFs (CRR3)

$$\text{EAD} = \max(\text{Drawn} - \text{SCRA}, 0) + \text{CCF} \times \max(\text{Undrawn} - \text{SCRA}_{\text{residual}}, 0)$$

| CRR3 bucket (Annex I) | CCF | Typical items |
|---|---|---|
| Bucket 1 | 100% | Credit substitutes, acceptances, forward purchases, commitments the client must draw |
| Bucket 2 | 50% | Transaction-related contingencies (performance bonds, bid bonds, warranties), NIFs/RUFs |
| Bucket 3 | 40% | Undrawn commitments that are not unconditionally cancellable, regardless of maturity |
| Bucket 4 | 20% | Short-term self-liquidating trade finance items |
| Bucket 5 | 10% | Unconditionally cancellable commitments (UCC) – phased in under Art. 495d |

**UCC phase-in (Art. 495d):** 0% of the 10% until 31/12/2029, then 25% (2030), 50% (2031),
75% (2032) and 100% from 2033.

**CRR2 comparison:** 100% / 50% / 20% / 0%, where undrawn commitments were 20% (original
maturity ≤ 1 year) or 50% (> 1 year), and UCC were 0%.

### 2. Derivatives – SA-CCR (CRR Art. 274–280f, BCBS 279)

$$\text{EAD} = \alpha \,(\text{RC} + \text{PFE}), \qquad \alpha = 1.4$$

**Replacement cost**
- Unmargined: $\text{RC} = \max(V - C, 0)$
- Margined: $\text{RC} = \max(V - C,\ TH + MTA - NICA,\ 0)$; the margined EAD is capped at the unmargined EAD.

**Potential future exposure**

$$\text{PFE} = \text{multiplier} \times \text{AddOn}^{\text{aggregate}}, \quad
\text{multiplier} = \min\left\{1;\ 0.05 + 0.95\, e^{\frac{V - C}{2 \times 0.95 \times \text{AddOn}^{\text{aggregate}}}}\right\}$$

**Trade-level inputs**
- Adjusted notional: $d = N \times SD$ for IR and credit, with $SD = \frac{e^{-0.05S} - e^{-0.05E}}{0.05}$; notional for FX, equity and commodity.
- Maturity factor: $MF = \sqrt{\min(M, 1)}$ (floor 10 business days) if unmargined; $MF = 1.5\sqrt{MPOR/250}$ if margined.
- Supervisory delta: ±1 for linear trades; for options $\pm\Phi(\pm d_1)$ with
  $d_1 = \frac{\ln\left(\frac{P+\lambda}{K+\lambda}\right) + \tfrac12\sigma^2 T}{\sigma\sqrt{T}}$.

**Aggregation by asset class**
- **Interest rates:** hedging set per currency, three maturity buckets (< 1y, 1–5y, > 5y):
  $EN = \sqrt{D_1^2 + D_2^2 + D_3^2 + 1.4 D_1D_2 + 1.4 D_2D_3 + 0.6 D_1D_3}$, AddOn = 0.5% × EN.
- **FX:** hedging set per currency pair, AddOn = 4% × |Σ δ d MF|.
- **Credit / equity:** single-factor model,
  $\text{AddOn} = \sqrt{\left(\sum_k \rho_k A_k\right)^2 + \sum_k (1-\rho_k^2) A_k^2}$ with ρ = 50% (single names) or 80% (indices).
- **Commodity:** hedging sets energy / metals / agricultural / other; same formula across commodity types with ρ = 40%.
- **No diversification across asset classes:** add-ons are summed.

| Asset class | Supervisory factor | Correlation | Option volatility |
|---|---|---|---|
| Interest rate | 0.50% | – | 50% |
| FX | 4.0% | – | 15% |
| Credit single name | AAA–AA 0.38%, A 0.42%, BBB 0.54%, BB 1.06%, B 1.6%, CCC 6.0% | 50% | 100% |
| Credit index | IG 0.38%, SG 1.06% | 80% | 80% |
| Equity single name / index | 32% / 20% | 50% / 80% | 120% / 75% |
| Commodity electricity / other | 40% / 18% | 40% | 150% / 70% |

**Validation:** the engine reproduces the BCBS 279 Annex 4a Example 1 result
(EAD = 1.4 × (60 + 347) = 569); see `tests/test_sa_ccr.py`.

**Scope limits of this version:** CDO tranches, basis and volatility transactions, the simplified
SA-CCR and the original exposure method are not implemented. Collateral haircuts are not computed:
C must be entered net of haircuts. In the EU, CRR3 includes transitional provisions that allow
α = 1 in specific output-floor cases, and the UK PRA applies α = 1 to non-financial corporates and
pension funds; α can be overridden per netting set for such analyses.

### References
- Regulation (EU) 2024/1623 (CRR3) – Art. 111, Annex I, Art. 495d. [EUR-Lex](https://eur-lex.europa.eu/eli/reg/2024/1623/oj/eng)
- Regulation (EU) No 575/2013 as amended by Regulation (EU) 2019/876 (CRR2) – Art. 274–280f.
- BCBS 279, *The standardised approach for measuring counterparty credit risk exposures* (2014). [BIS](https://www.bis.org/publ/bcbs279.pdf)
- Basel Framework, CRE52. [BIS](https://www.bis.org/basel_framework/chapter/CRE/52.htm)
- EBA, Final draft RTS on off-balance-sheet items under CRR3 Art. 111(8) (EBA/RTS/2025/06).
- EBA, Final draft RTS amending the SA-CCR RTS (EBA/RTS/2024/16).
""")
