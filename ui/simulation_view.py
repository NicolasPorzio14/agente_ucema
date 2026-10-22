"""
ui/simulation_view.py
=====================
Pestaña "Simulación": Monte Carlo del precio del activo analizado.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from agent import AnalysisResult, fetch_history, fetch_risk_free
from simulation import DRIFTS, HORIZONS, METHODS, SAMPLES, SimulationResult, simulate_price
from ui import charts
from ui import components as ui
from ui.components import fnum, signed


@st.cache_data(ttl=900, show_spinner=False, max_entries=32)
def _simulate(ticker: str, horizon: str, method: str, drift: str, sample: str, target: float | None,
              support: tuple[float, str] | None, currency: str | None) -> SimulationResult:
    close = fetch_history(ticker, "5Y")["Close"]
    rf = fetch_risk_free()
    sup = {"level": support[0], "source": support[1]} if support else None
    return simulate_price(close, ticker, horizon, method, drift, sample, rf, target, sup, currency)


def _default_support(supports: list[dict[str, Any]]) -> int:
    """Primer soporte al menos 5% por debajo; los muy cercanos se perforan casi seguro."""
    for i, s in enumerate(supports):
        if s["distance_pct"] <= -5:
            return i
    return 0


def render(result: AnalysisResult, panel, plotly_config: dict) -> SimulationResult | None:
    """Dibuja la pestaña y devuelve la simulación (la usan el asistente y el PDF)."""
    m, t = result.market or {}, result.technicals or {}
    target = m.get("target_mean")
    supports = t.get("supports") or []

    c1, c2, c3, c4, c5 = st.columns([1.05, 1.25, 1.1, 0.9, 1.5], gap="small")
    horizon = c1.segmented_control("Horizonte", list(HORIZONS), default="12M", key="sim_h") or "12M"
    method = c2.segmented_control("Método", list(METHODS), default="bootstrap", key="sim_m",
                                  format_func=lambda k: {"bootstrap": "Bootstrap", "gbm": "Lognormal"}[k]) or "bootstrap"
    drift = c3.segmented_control("Deriva", list(DRIFTS), default="neutral", key="sim_d",
                                 format_func=lambda k: {"neutral": "Neutral", "historica": "Histórica"}[k]) or "neutral"
    sample = c4.segmented_control("Muestra", list(SAMPLES), default="3Y", key="sim_s") or "3Y"
    support = None
    if supports:
        idx = c5.selectbox("Soporte a evaluar", range(len(supports)), index=_default_support(supports), key="sim_sup",
                           format_func=lambda i: f"{supports[i]['level']:,.2f} ({supports[i]['distance_pct']:+.1f}%) · "
                                                 f"{supports[i]['source']}")
        support = (supports[idx]["level"], supports[idx]["source"])

    try:
        with st.spinner("Simulando 10.000 trayectorias…"):
            sim = _simulate(result.ticker, horizon, method, drift, sample, target, support, m.get("currency"))
    except Exception as exc:
        st.warning(f"No se pudo simular: {exc}")
        return None
    s = sim.summary
    tg, sp = s.get("target") or {}, s.get("support") or {}
    pc = s["final_price_percentiles"]

    ui.render(ui.kpi_grid([
        ("Prob. de tocar el objetivo", fnum(tg.get("prob_touch_pct"), "{:.1f}", "%"),
         f"Objetivo {fnum(tg.get('price'))} ({fnum(tg.get('distance_pct'), '{:+.1f}', '%')})" if tg else "sin consenso", ""),
        ("Prob. de terminar más allá", fnum(tg.get("prob_end_beyond_pct"), "{:.1f}", "%"),
         f"al cierre de {horizon}" if tg else "sin consenso", ""),
        (f"Mediana a {horizon}", fnum(pc["p50"]), signed(s["median_return_pct"], "{:+.1f}") + " vs hoy", ""),
        ("Rango 90%", f"{fnum(pc['p5'])} – {fnum(pc['p95'])}", "percentiles 5 y 95", ""),
        ("Prob. de ganancia", fnum(s["prob_gain_pct"], "{:.1f}", "%"),
         f"Pérdida > 20%: {fnum(s['prob_loss_over_20_pct'], '{:.1f}', '%')}", ""),
        (f"VaR 95% · {horizon}", fnum(s["var95_pct"], "{:.1f}", "%"), f"CVaR 95% {fnum(s['cvar95_pct'], '{:.1f}', '%')}", ""),
    ]))
    for n in sim.notes:
        ui.render(ui.note(n))

    sub = f"{METHODS[method]} · {DRIFTS[drift].lower()} · {s['paths']:,} trayectorias".replace(",", ".")
    with panel(f"Abanico de precios a {horizon}", sub, key="sim-fan"):
        hist = fetch_history(result.ticker, "1Y")["Close"].tail(126)
        st.plotly_chart(charts.simulation_fan(sim, hist, tg.get("price"), sp.get("price")),
                        width="stretch", config=plotly_config)
    left, right = st.columns([1.3, 1], gap="medium")
    with left:
        with panel("Distribución del precio final", f"{s['paths']:,} escenarios".replace(",", "."), key="sim-hist"):
            st.plotly_chart(charts.simulation_histogram(sim, tg.get("price"), sp.get("price")),
                            width="stretch", config=plotly_config)
    with right:
        with panel("Lecturas de la simulación", "lo que recibe el asistente", key="sim-facts"):
            ui.render(ui.facts_list(sim.facts))
    with panel("Cómo leer esta simulación", key="sim-help"):
        ui.render(
            "<ul class='facts'>"
            "<li><b>Bootstrap</b>: arma cada trayectoria pegando bloques de 10 ruedas de retornos reales de la muestra. "
            "Respeta las colas gordas y las rachas de volatilidad. <b>Lognormal</b> supone retornos normales "
            "(el modelo de Black-Scholes).</li>"
            "<li><b>Deriva neutral</b>: quita la tendencia pasada y hace crecer el precio, en promedio, a la tasa "
            "libre de riesgo. <b>Histórica</b>: proyecta la tendencia de la muestra (optimista si la acción viene "
            "de subir mucho).</li>"
            "<li><b>Tocar</b> el objetivo es llegar en algún momento del horizonte; <b>terminar más allá</b> es estar "
            "por encima (o por debajo) al final. La primera es siempre mayor o igual.</li>"
            "<li>No es un pronóstico: mide qué tan exigente es un nivel de precio dada la volatilidad de la acción. "
            "Semilla fija: los números no cambian al recargar.</li></ul>")
    return sim
