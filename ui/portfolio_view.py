"""
ui/portfolio_view.py
====================
Sección "Cartera": lista de tickers con pesos editables y su análisis conjunto
(correlaciones, diversificación, contribución al riesgo, métricas clave).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from portfolio import MAX_TICKERS, PortfolioResult, analyze_portfolio, resolve_weights
from ui import charts
from ui import components as ui
from ui.components import Markup, esc, fnum, signed, tone

DEFAULT_TICKERS = ["VIST", "YPF", "AAPL", "MSFT", "GLD"]
TREND_CLS = {"alcista": "up", "bajista": "down"}
TREND_ARROW = {"alcista": "▲ ", "bajista": "▼ ", "mixta": ""}


@st.cache_data(ttl=600, show_spinner=False, max_entries=24)
def _analyze(tickers: tuple[str, ...], weights: tuple[float, ...], window: str) -> PortfolioResult:
    return analyze_portfolio(list(tickers), list(weights), window)


def _base_rows(tickers: list[str], weights: list[float | None] | None = None) -> pd.DataFrame:
    return pd.DataFrame({"Ticker": tickers,
                         "Peso %": pd.Series(weights or [None] * len(tickers), dtype="float64")})


def _equal_weights() -> None:
    """Borra los pesos cargados (vuelve a 1/n) conservando los tickers."""
    last = st.session_state.get("pf_last")
    tickers = [t for t in (last["Ticker"].tolist() if last is not None else DEFAULT_TICKERS) if isinstance(t, str) and t.strip()]
    st.session_state["pf_rows"] = _base_rows(tickers)
    st.session_state["pf_ver"] += 1


def _open_analysis(ticker: str) -> None:
    st.session_state["ticker_in"] = ticker
    st.session_state["mode"] = "Análisis"
    st.session_state["auto_run"] = True


def _cell(v: Any, fmt: str) -> str:
    return "—" if v is None else fmt.format(v)


def render(window: str, panel, plotly_config: dict) -> None:
    ss = st.session_state
    ss.setdefault("pf_rows", _base_rows(DEFAULT_TICKERS))
    ss.setdefault("pf_ver", 0)

    left, right = st.columns([1, 1.5], gap="medium")
    with left:
        with panel("Composición", f"hasta {MAX_TICKERS} tickers", key="pf-editor"):
            edited = st.data_editor(
                ss["pf_rows"], key=f"pf_editor_{ss['pf_ver']}", num_rows="dynamic", hide_index=True, width="stretch",
                placeholder="—",
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker", max_chars=15,
                                                          help="Símbolo de Yahoo Finance (ej. AAPL, GGAL, YPFD.BA)"),
                    "Peso %": st.column_config.NumberColumn(
                        "Peso %", min_value=0.0, max_value=100.0, step=0.5, format="%.1f",
                        help="Vacío = reparte en partes iguales lo que falta para llegar a 100%"),
                })
            ss["pf_last"] = edited
            st.button("Volver a pesos iguales (1/n)", on_click=_equal_weights, width="stretch")
            tickers, weights, notes = resolve_weights(list(zip(edited["Ticker"], edited["Peso %"])))
            if tickers:
                custom = edited["Peso %"].notna().any()
                ui.render("<div class='asof' style='margin-top:0.6rem'>Pesos usados"
                          f"{' (cargados por vos)' if custom else ' (iguales)'}: "
                          + " · ".join(f"<b>{esc(t)}</b> {w:.1f}%" for t, w in zip(tickers, weights)) + "</div>")
            for n in notes:
                ui.render(ui.note(n))

    if len(tickers) < 2:
        with right:
            st.info("Cargá al menos dos tickers en la tabla para analizar la cartera.")
        return
    try:
        with st.spinner("Descargando precios y calculando la cartera…"):
            r = _analyze(tuple(tickers), tuple(round(w, 4) for w in weights), window)
    except Exception as exc:
        with right:
            st.error(str(exc))
        return
    s = r.summary

    with right:
        with panel("Peso vs contribución al riesgo", "cuánto pesa cada activo y cuánto riesgo aporta", key="pf-risk"):
            st.plotly_chart(charts.risk_vs_weight(r.table), width="stretch", config=plotly_config)

    for n in r.notes:
        ui.render(ui.note(n))
    ui.render(ui.kpi_grid([
        (f"Retorno {window}", signed(s["return_pct"], "{:+.1f}"), f"{s['start']} a {s['end']}", tone(s["return_pct"])),
        (f"vs {s['benchmark']}", signed(s.get("vs_benchmark_pp"), "{:+.1f}", " pp"),
         f"{s['benchmark']} {fnum(s.get('benchmark_return_pct'), '{:+.1f}', '%')}", tone(s.get("vs_benchmark_pp"))),
        ("Volatilidad anual", fnum(s["volatility_pct"], "{:.1f}", "%"),
         f"Promedio ponderado {fnum(s['weighted_avg_volatility_pct'], '{:.1f}', '%')}", ""),
        ("Máximo drawdown", fnum(s["max_drawdown_pct"], "{:.1f}", "%"), f"Actual {fnum(s['current_drawdown_pct'], '{:.1f}', '%')}", ""),
        ("Sharpe", fnum(s["sharpe"], "{:.2f}"), f"Tasa libre de riesgo {fnum(s['risk_free_pct'], '{:.2f}', '%')}", ""),
        (f"Beta vs {s['benchmark']}", fnum(s["beta"], "{:.2f}"), "retornos diarios", ""),
        ("Ratio de diversificación", fnum(s["diversification_ratio"], "{:.2f}"), "1.0 = sin diversificación", ""),
        ("Correlación promedio", fnum(s["avg_correlation"], "{:.2f}"), f"1er factor explica {fnum(s['first_factor_share_pct'], '{:.0f}', '%')}", ""),
        ("Activos efectivos", fnum(s["effective_assets"], "{:.1f}"), f"de {s['n_assets']} · por peso", ""),
        ("Apuestas efectivas", fnum(s["effective_bets"], "{:.1f}"), "por contribución al riesgo", ""),
        ("VaR 95% · 1 día", fnum(s["var95_1d_pct"], "{:.2f}", "%"), f"CVaR {fnum(s['cvar95_1d_pct'], '{:.2f}', '%')}", ""),
        ("Sector principal", f"{fnum(s['max_sector_weight_pct'], '{:.0f}', '%')}", s["max_sector"], ""),
    ]))

    with panel("Evolución base 100", f"ventana {window} · pesos constantes (rebalanceo diario)", key="pf-lines"):
        st.plotly_chart(charts.portfolio_lines(r.normalized, r.value, r.bench_value, r.benchmark),
                        width="stretch", config=plotly_config)

    c1, c2 = st.columns([1.15, 1], gap="medium")
    with c1:
        with panel("Matriz de correlaciones", "retornos diarios de la ventana", key="pf-corr"):
            st.plotly_chart(charts.correlation_heatmap(r.corr), width="stretch", config=plotly_config)
    with c2:
        with panel("Drawdown de la cartera", "caída desde el máximo previo", key="pf-dd"):
            st.plotly_chart(charts.portfolio_drawdown(r.drawdown), width="stretch", config=plotly_config)
        with panel("Concentración por sector", "% del peso", key="pf-sector"):
            st.plotly_chart(charts.sector_bars(r.sectors), width="stretch", config=plotly_config)

    with panel("Métricas por activo", f"ventana {window}", key="pf-table"):
        rows, cls = [], []
        for x in r.table:
            trend = x["trend"] or "—"
            rows.append([Markup(f"<b>{esc(x['ticker'])}</b>"), (x["name"] or "")[:26], x["sector"],
                         _cell(x["weight_pct"], "{:.1f}"), _cell(x["risk_contribution_pct"], "{:.1f}"),
                         signed(x["return_pct"], "{:+.1f}"), _cell(x["volatility_pct"], "{:.1f}"),
                         _cell(x["max_drawdown_pct"], "{:.1f}"), _cell(x["sharpe"], "{:.2f}"), _cell(x["beta"], "{:.2f}"),
                         _cell(x["corr_with_portfolio"], "{:.2f}"), TREND_ARROW.get(trend, "") + trend,
                         _cell(x["rsi14"], "{:.1f}"), _cell(x["pe_forward"], "{:.1f}"),
                         signed(x["upside_pct"], "{:+.1f}") if x["upside_pct"] is not None else "—"])
            cls.append([""] * 5 + [tone(x["return_pct"])] + [""] * 5 + [TREND_CLS.get(trend, "")] + [""] * 2
                       + [tone(x["upside_pct"])])
        ui.render(ui.table(["Ticker", "Nombre", "Sector", "Peso %", "Riesgo %", "Retorno %", "Vol %", "Max DD %",
                            "Sharpe", "Beta", "Corr. cartera", "Tendencia", "RSI 14", "P/E fwd", "Upside"],
                           rows, cell_classes=cls, text_cols=3))
        ui.render("<div class='asof' style='margin-top:0.5rem'>Riesgo %: parte de la varianza de la cartera que "
                  "explica cada activo (suma 100). Tendencia: precio vs SMA 50 y SMA 200, mismo criterio que el "
                  "análisis individual. Upside: al precio objetivo medio del consenso.</div>")

    c1, c2 = st.columns([1.6, 1], gap="medium")
    with c1:
        with panel("Lecturas de la cartera", "calculadas, no generadas por IA", key="pf-facts"):
            ui.render(ui.facts_list(r.facts))
    with c2:
        with panel("Análisis individual", "abre el informe completo de un activo", key="pf-open"):
            pick = st.selectbox("Activo", [x["ticker"] for x in r.table], key="pf_pick")
            st.button(f"Analizar {pick}", type="primary", width="stretch", on_click=_open_analysis, args=(pick,))
            ui.render("<div class='asof' style='margin-top:0.5rem'>Corre el agente con los parámetros de la barra "
                      "lateral de Análisis.</div>")
