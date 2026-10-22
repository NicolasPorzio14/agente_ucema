"""
app.py
======
Interfaz Streamlit del Agente de Análisis Financiero, con tema oscuro estilo
terminal financiera (ver ui/theme.py, ui/components.py y ui/charts.py).

Ejecutar con:  python -m streamlit run app.py
"""

from __future__ import annotations

import dataclasses
import itertools
import os
import queue
import threading
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Carga OPENAI_API_KEY (y opcionalmente OPENAI_MODEL o EXTRA_CA_CERTS) desde el
# archivo .env ubicado junto a app.py, si existe. Va antes de importar agent
# porque agent configura los certificados al importarse.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from agent import (DEFAULT_OPENAI_MODEL, PERIOD_DAYS, AgentStep, AnalysisResult,  # noqa: E402
                   DataSourceError, FinancialAgent, TickerNotFoundError)
from portfolio import WINDOWS  # noqa: E402
from report_pdf import build_pdf  # noqa: E402
from ui import assistant_view, portfolio_view, simulation_view  # noqa: E402
from ui import charts  # noqa: E402
from ui import components as ui  # noqa: E402
from ui.components import Markup, big, esc, fnum, signed, tone  # noqa: E402
from ui.theme import inject_css  # noqa: E402

st.set_page_config(page_title="AGF · Research financiero", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")
inject_css()

PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}

_panel_ids = itertools.count()


def panel(title: str, sub: str = "", key: str | None = None):
    """Contenedor con borde. La key "panel-…" es la que toma el CSS del tema."""
    box = st.container(border=True, key=f"panel-{key or next(_panel_ids)}")
    with box:
        ui.render(ui.panel_header(title, sub))
    return box


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------

st.session_state.setdefault("mode", "Análisis")
st.session_state.setdefault("ticker_in", "VIST")

with st.sidebar:
    ui.render("<div class='side-brand'><span class='logo'>AGF</span>Agente de análisis financiero</div>")
    mode = st.segmented_control("Sección", ["Análisis", "Cartera"], key="mode", required=True, width="stretch",
                                label_visibility="collapsed")
    if mode == "Cartera":
        pf_window = st.radio("Ventana de la cartera", list(WINDOWS), index=1, horizontal=True,
                             help="Período sobre el que se miden retornos, volatilidad y correlaciones.")
        ui.render("<div class='side-foot'>Cargá los tickers y sus pesos en la tabla. Las filas sin peso reparten "
                  "en partes iguales lo que falta para 100%.<br><br>Datos: Yahoo Finance. Fines educativos. "
                  "No es asesoramiento de inversión.</div>")

if mode == "Cartera":
    ui.render(ui.topbar(None, {"Sección": "Cartera", "Ventana": pf_window, "Fuente": "Yahoo Finance"}))
    portfolio_view.render(pf_window, panel, PLOTLY_CONFIG)
    st.stop()

with st.sidebar:
    ticker = st.text_input("Ticker", key="ticker_in",
                           help="Símbolo de Yahoo Finance. Ej: VIST, GGAL, AAPL, SPY, YPFD.BA")
    period = st.radio("Temporalidad", list(PERIOD_DAYS), index=2, horizontal=True)
    peers = st.text_input(
        "Comparables (opcional)", placeholder="YPF, PAM, TGS",
        help="Tickers separados por coma. Con CrewAI, si lo dejás vacío el agente elige los "
             "comparables. Con el motor de reglas, sin comparables se omite esa sección.")
    engine_label = st.selectbox(
        "Motor del agente", ["CrewAI + OpenAI", "Reglas cuantitativas"],
        help="CrewAI + OpenAI usa tus créditos de OpenAI. El motor de reglas no usa LLM ni consume créditos.")
    use_crewai = engine_label.startswith("CrewAI")
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    if use_crewai:
        with st.expander("OpenAI", expanded=not api_key):
            typed_key = st.text_input(
                "OPENAI_API_KEY", type="password",
                placeholder="Cargada desde .env" if api_key else "sk-...",
                help="Si ya la cargaste en el archivo .env podés dejar este campo vacío. "
                     "La clave solo vive en esta sesión del navegador.")
            api_key = typed_key.strip() or api_key
            model = st.text_input("Modelo", value=model,
                                  help="Cualquier modelo de chat de OpenAI con tool calling, ej. gpt-4.1-mini "
                                       "o gpt-4.1. Un modelo más grande redacta un informe de mejor calidad.")
        if not api_key:
            st.warning("Falta OPENAI_API_KEY: se usará el motor de reglas.")
    # "auto_run" lo activa el botón "Analizar" de la sección Cartera.
    run = st.button("Ejecutar análisis", type="primary", width="stretch") or st.session_state.pop("auto_run", False)
    ui.render("<div class='side-foot'>Datos: Yahoo Finance · Google News<br>"
              "Fines educativos. No es asesoramiento de inversión.</div>")


# ---------------------------------------------------------------------------
# Ejecución del agente
# ---------------------------------------------------------------------------

def render_step(step: AgentStep) -> None:
    ui.render(ui.step_html(step.kind, step.title, step.detail))


def run_agent_live(agent: FinancialAgent, ticker: str, period: str, peers: str) -> AnalysisResult:
    """Corre el agente en segundo plano y muestra sus pasos desde el hilo principal.

    CrewAI ejecuta herramientas en otros hilos (a veces en paralelo). Si esos
    hilos escriben en la página, Streamlit recibe actualizaciones desordenadas y
    el navegador puede quedar en blanco. Por eso el agente solo encola pasos y
    este hilo, el único que toca la UI, los dibuja.
    """
    steps: queue.Queue[AgentStep] = queue.Queue()
    agent.on_step = steps.put
    outcome: dict[str, Any] = {}

    def worker() -> None:
        try:
            outcome["result"] = agent.run(ticker, period, peers)
        except Exception as exc:  # se relanza en el hilo principal
            outcome["error"] = exc

    thread = threading.Thread(target=worker, name="financial-agent", daemon=True)
    thread.start()
    while thread.is_alive() or not steps.empty():
        try:
            render_step(steps.get(timeout=0.25))
        except queue.Empty:
            continue
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


result: AnalysisResult | None = st.session_state.get("result")
# Un resultado guardado por una versión anterior de la app rompería la interfaz nueva.
if result is not None and not all(hasattr(result, f.name) for f in dataclasses.fields(AnalysisResult)):
    st.session_state.pop("result", None)
    result = None

if run:
    ui.render(ui.topbar(ticker.strip().upper() or None, {"Ventana": period, "Estado": "analizando"}))
    engine = "crewai" if use_crewai and api_key else "reglas"
    st.session_state.pop("result", None)
    result = None
    with st.status(f"Analizando {ticker.upper()}…", expanded=True) as status:
        agent = FinancialAgent(engine=engine, api_key=api_key or None, model=model.strip() or None)
        try:
            result = run_agent_live(agent, ticker, period, peers)
            st.session_state["result"] = result
            status.update(label="Análisis completo · ver razonamiento", state="complete", expanded=False)
        except TickerNotFoundError as exc:
            status.update(label="Ticker no encontrado", state="error")
            st.error(str(exc))
        except DataSourceError as exc:
            status.update(label="Falla de la fuente de datos", state="error")
            st.error(f"{exc}. Probá de nuevo en unos minutos (Yahoo limita la cantidad de pedidos). "
                     "Si el error menciona 'certificate', revisá EXTRA_CA_CERTS en el README.")
        except Exception as exc:  # cualquier otro error inesperado
            status.update(label="Error inesperado", state="error")
            st.exception(exc)

if result is None:
    if not run:
        ui.render(ui.topbar(None, {"Fuente": "Yahoo Finance", "Noticias": "Google News"}))
        ui.render(ui.empty_state())
    st.stop()


# ---------------------------------------------------------------------------
# Cabecera, indicadores y señales
# ---------------------------------------------------------------------------

m = result.market or {}
f = result.financials or {}
t = result.technicals or {}
p = result.peers or {}
news = result.news or {}
cur = m.get("currency", "")
engine_short = "CrewAI · " + result.engine.split("(")[-1].rstrip(")") if result.engine.startswith("CrewAI") \
    else f"Reglas{' (fallback)' if 'fallback' in result.engine else ''}"

if not run:
    ui.render(ui.topbar(result.ticker, {"Ventana": result.period, "Motor": engine_short,
                                        "Último dato": m.get("last_date", "N/D")}))
ui.render(ui.quote_header(m, result.ticker, result.period, engine_short))
for n in m.get("notes") or []:
    ui.render(ui.note(n))

rets = t.get("returns_pct") or {}
bench = t.get("vs_benchmark") or {}
up = m.get("upside_to_target_mean_pct")
ui.render(ui.kpi_grid([
    ("Market cap", f"{big(m.get('market_cap'))} {cur}", f"EV {big(m.get('enterprise_value'))}", ""),
    ("P/E trailing", fnum(m.get("pe_trailing"), "{:.1f}", "x"), f"Forward {fnum(m.get('pe_forward'), '{:.1f}', 'x')}", ""),
    ("EV/EBITDA TTM", fnum(m.get("ev_ebitda"), "{:.1f}", "x"), f"P/S {fnum(m.get('price_to_sales'), '{:.2f}', 'x')}", ""),
    ("FCF yield TTM", fnum(m.get("fcf_yield_pct"), "{:.2f}", "%"), f"UST 10Y {fnum(m.get('us10y_yield_pct'), '{:.2f}', '%')}", ""),
    ("Dividend yield", fnum(m.get("dividend_yield_pct"), "{:.2f}", "%"), f"Payout {fnum(m.get('payout_ratio_pct'), '{:.0f}', '%')}", ""),
    ("Upside obj. medio", signed(up, "{:+.1f}"), f"Obj. {fnum(m.get('target_mean'))} · {m.get('analyst_count') or 0} analistas", tone(up)),
    ("ROE TTM", fnum(m.get("roe_ttm_pct"), "{:.1f}", "%"), f"Mg. neto {fnum(m.get('net_margin_ttm_pct'), '{:.1f}', '%')}", ""),
    ("Mg. operativo TTM", fnum(m.get("operating_margin_ttm_pct"), "{:.1f}", "%"), m.get("fundamentals_basis") or "", ""),
    ("Deuda neta/EBITDA", fnum(m.get("net_debt_to_ebitda_ttm"), "{:.2f}", "x"), f"Deuda neta {big(m.get('net_debt_mrq'))}", ""),
    ("Beta 5Y (Yahoo)", fnum(m.get("beta_5y_monthly"), "{:.2f}"), f"Beta 1Y diaria {fnum(bench.get('beta_1y_daily'), '{:.2f}')}", ""),
    ("Retorno 1Y", signed(rets.get("1Y"), "{:+.1f}"), f"YTD {fnum(rets.get('YTD'), '{:+.1f}', '%')}", tone(rets.get("1Y"))),
    ("Próximo balance", m.get("next_earnings_date") or "N/D", f"Short {fnum(m.get('short_pct_float'), '{:.1f}', '%')} del float", ""),
]))
ui.render(ui.signals(t, m))
ui.render("<div style='height:0.6rem'></div>")


def cell(v: Any, fmt: str) -> str:
    return "—" if v is None or (isinstance(v, float) and pd.isna(v)) else fmt.format(v)


tab_report, tab_chat, tab_tech, tab_fin, tab_val, tab_sim, tab_news, tab_log = st.tabs(
    ["Informe", "Asistente", "Técnico", "Financieros", "Valuación y comparables", "Simulación", "Noticias", "Agente"])

# ---------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------
with tab_report:
    left, right = st.columns([2.4, 1], gap="medium")
    with left:
        with st.container(border=True, key="report"):
            st.markdown(result.diagnosis)
    with right:
        with panel("Control de calidad", "borrador vs. versión final"):
            ui.render(ui.qc_card(result.review))
        # El PDF incluye la simulación, que se calcula más abajo: el botón se completa después.
        pdf_slot = st.empty()
        st.download_button("Descargar Markdown (.md)", data=result.diagnosis, width="stretch",
                           file_name=f"informe_{result.ticker}_{m.get('last_date', '')}.md", mime="text/markdown")
        if m.get("facts"):
            with panel("Lecturas de valuación", "precalculadas"):
                ui.render(ui.facts_list(m["facts"]))
        if m.get("summary"):
            with panel("Perfil de la empresa"):
                ui.render(f"<p style='font-size:0.84rem;line-height:1.55'>{esc(m['summary'])}</p>")

# ---------------------------------------------------------------------------
# Técnico
# ---------------------------------------------------------------------------
with tab_tech:
    if result.chart_df is not None and not result.chart_df.empty:
        with panel("Precio · medias móviles · Bollinger · volumen · RSI · MACD",
                   f"ventana {result.period} desde {t.get('window_start', '')} · velas huecas = alcistas"):
            st.plotly_chart(charts.technical_chart(result.chart_df), width="stretch", config=PLOTLY_CONFIG)
            if result.chart_df["SMA200"].isna().all():
                ui.render(ui.note("La SMA 200 no está disponible: el activo tiene menos de 200 ruedas de historia."))
    else:
        st.warning("No hay datos suficientes para graficar.")
    if t:
        left, right = st.columns([1.6, 1], gap="medium")
        with left:
            if result.benchmark_df is not None and not result.benchmark_df.empty:
                with panel(f"Desempeño relativo vs {bench.get('benchmark', '')}", "base 100 al inicio de la ventana"):
                    st.plotly_chart(charts.relative_chart(result.benchmark_df), width="stretch", config=PLOTLY_CONFIG)
            with panel("Lecturas técnicas", "lo que recibe el agente"):
                ui.render(ui.facts_list(t.get("facts") or []))
        with right:
            with panel("Retornos y riesgo"):
                risk_rows = [[k, signed(v, "{:+.2f}")] for k, v in rets.items()]
                risk_cls = [["", tone(v)] for v in rets.values()]
                risk_rows += [
                    ["Volatilidad anual (ventana)", fnum(t.get("volatility_window_annual_pct"), "{:.1f}", "%")],
                    ["Máximo drawdown (ventana)", fnum(t.get("max_drawdown_window_pct"), "{:.1f}", "%")],
                    ["Sharpe / Sortino", f"{fnum(t.get('sharpe_window'), '{:.2f}')} / {fnum(t.get('sortino_window'), '{:.2f}')}"],
                    ["ATR 14", f"{fnum(t.get('atr14'))} ({fnum(t.get('atr14_pct_price'), '{:.1f}', '%')})"],
                    [f"Relativo vs {bench.get('benchmark', '')}", signed(bench.get("relative_performance_pp"), "{:+.1f}", " pp")],
                    ["Correlación 1Y", fnum(bench.get("correlation_1y_daily"), "{:.2f}")],
                    ["Volumen 20d / 90d", fnum(t.get("volume_20d_vs_90d"), "{:.2f}", "x")],
                ]
                risk_cls += [["", ""]] * 4 + [["", tone(bench.get("relative_performance_pp"))]] + [["", ""]] * 2
                ui.render(ui.table(["Métrica", "Valor"], risk_rows, cell_classes=risk_cls))
                if t.get("risk_ratios_note"):
                    ui.render(ui.note(t["risk_ratios_note"]))
            with panel("Soportes y resistencias", "origen de cada nivel"):
                lv_rows, lv_cls = [], []
                for kind, key in (("Resistencia", "resistances"), ("Soporte", "supports")):
                    for s in t.get(key) or []:
                        lv_rows.append([kind, fnum(s["level"]), fnum(s["distance_pct"], "{:+.1f}", "%"), s["source"]])
                        lv_cls.append(["", "", "", "wrap"])
                ui.render(ui.table(["Tipo", "Nivel", "Dist.", "Origen"], lv_rows, cell_classes=lv_cls))

# ---------------------------------------------------------------------------
# Financieros
# ---------------------------------------------------------------------------
with tab_fin:
    if f.get("applicable") and result.annual_df is not None:
        fcur = f.get("currency") or cur
        s = f.get("summary") or {}
        ui.render(ui.kpi_grid([
            ("CAGR ventas", fnum(s.get("revenue_cagr_pct"), "{:.1f}", "%"), s.get("period") or "", ""),
            ("CAGR EPS", fnum(s.get("eps_cagr_pct"), "{:.1f}", "%"), "diluido", ""),
            ("CAGR FCF", fnum(s.get("fcf_cagr_pct"), "{:.1f}", "%"), f"Años con FCF < 0: {s.get('years_with_negative_fcf', 'N/D')}", ""),
            ("ROIC promedio", fnum(s.get("avg_roic_pct"), "{:.1f}", "%"), "años fiscales", ""),
            ("FCF / utilidad neta", fnum(s.get("avg_fcf_to_net_income_pct"), "{:.0f}", "%"), "promedio", ""),
            ("Acciones diluidas", fnum(s.get("diluted_shares_change_pct"), "{:+.1f}", "%"),
             "recompras" if (s.get("diluted_shares_change_pct") or 0) < -1 else "dilución" if (s.get("diluted_shares_change_pct") or 0) > 1 else "estables", ""),
        ]))
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            with panel("Ventas, utilidad y free cash flow", f"millones de {fcur} · año fiscal"):
                st.plotly_chart(charts.financials_chart(result.annual_df), width="stretch", config=PLOTLY_CONFIG)
        with c2:
            with panel("Márgenes y ROIC", "% · año fiscal"):
                st.plotly_chart(charts.margins_chart(result.annual_df), width="stretch", config=PLOTLY_CONFIG)
        if result.quarterly_df is not None and not result.quarterly_df.empty:
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                with panel("Ventas trimestrales", f"millones de {fcur}"):
                    st.plotly_chart(charts.quarterly_revenue_chart(result.quarterly_df), width="stretch", config=PLOTLY_CONFIG)
            with c2:
                with panel("Margen operativo trimestral", "%"):
                    st.plotly_chart(charts.quarterly_margin_chart(result.quarterly_df), width="stretch", config=PLOTLY_CONFIG)
        rows_def = [("revenue", "Ventas", 1e6, "{:,.1f}"), ("gross_profit", "Ganancia bruta", 1e6, "{:,.1f}"),
                    ("operating_income", "Resultado operativo", 1e6, "{:,.1f}"), ("ebitda", "EBITDA", 1e6, "{:,.1f}"),
                    ("net_income", "Utilidad neta", 1e6, "{:,.1f}"), ("eps_diluted", "EPS diluido", 1, "{:,.2f}"),
                    ("operating_cf", "Flujo operativo", 1e6, "{:,.1f}"), ("capex", "Capex", 1e6, "{:,.1f}"),
                    ("fcf", "Free cash flow", 1e6, "{:,.1f}"), ("net_debt", "Deuda neta", 1e6, "{:,.1f}"),
                    ("revenue_growth", "Crecimiento ventas %", 1, "{:+.1f}"), ("gross_margin", "Margen bruto %", 1, "{:.1f}"),
                    ("operating_margin", "Margen operativo %", 1, "{:.1f}"), ("net_margin", "Margen neto %", 1, "{:.1f}"),
                    ("fcf_margin", "Margen FCF %", 1, "{:.1f}"), ("roe", "ROE %", 1, "{:.1f}"), ("roic", "ROIC %", 1, "{:.1f}"),
                    ("net_debt_to_ebitda", "Deuda neta / EBITDA", 1, "{:.2f}"),
                    ("interest_coverage", "Cobertura de intereses (x)", 1, "{:.1f}")]
        adf = result.annual_df
        years = [f"FY{d.year}" for d in adf.index]
        rows = [[label] + [cell(v / div if v is not None and not pd.isna(v) else None, fmt) for v in adf[col]]
                for col, label, div, fmt in rows_def if col in adf and adf[col].notna().any()]
        with panel("Estados contables", f"millones de {fcur} salvo indicación"):
            ui.render(ui.table(["Concepto"] + years, rows))
        if f.get("facts"):
            with panel("Lecturas de los estados contables", "lo que recibe el agente"):
                ui.render(ui.facts_list(f["facts"]))
    else:
        st.info(f.get("note") or "Sin estados contables para este activo.")

# ---------------------------------------------------------------------------
# Valuación y comparables
# ---------------------------------------------------------------------------
with tab_val:
    c1, c2 = st.columns([1, 1.2], gap="medium")
    with c1:
        with panel("Múltiplos y rendimientos"):
            val_rows = [
                ["P/E trailing", fnum(m.get("pe_trailing"), "{:.2f}", "x")], ["P/E forward", fnum(m.get("pe_forward"), "{:.2f}", "x")],
                ["PEG", fnum(m.get("peg"), "{:.2f}")], ["EV/EBITDA (TTM)", fnum(m.get("ev_ebitda"), "{:.2f}", "x")],
                ["EV/Ventas (TTM)", fnum(m.get("ev_revenue"), "{:.2f}", "x")], ["P/S (TTM)", fnum(m.get("price_to_sales"), "{:.2f}", "x")],
                ["P/B", fnum(m.get("price_to_book"), "{:.2f}", "x")], ["Earnings yield", fnum(m.get("earnings_yield_pct"), "{:.2f}", "%")],
                ["FCF yield (TTM)", fnum(m.get("fcf_yield_pct"), "{:.2f}", "%")], ["Treasury 10 años", fnum(m.get("us10y_yield_pct"), "{:.2f}", "%")],
                ["Insiders / Institucionales", f"{fnum(m.get('insiders_pct'), '{:.1f}', '%')} / {fnum(m.get('institutions_pct'), '{:.1f}', '%')}"],
                ["Short interest", f"{fnum(m.get('short_pct_float'), '{:.1f}', '%')} · {fnum(m.get('short_ratio_days'), '{:.1f}')} días"],
            ]
            ui.render(ui.table(["Métrica", "Valor"], val_rows))
    with c2:
        with panel("Consenso de analistas", f"{m.get('analyst_count') or 0} analistas"):
            rt = m.get("recommendation_trend") or {}
            now = rt.get("now") or {}
            if now:
                labels = [("strongBuy", "Compra fuerte"), ("buy", "Compra"), ("hold", "Mantener"),
                          ("sell", "Venta"), ("strongSell", "Venta fuerte")]
                ui.render(ui.table([lbl for _, lbl in labels], [[now.get(k, 0) for k, _ in labels]], text_cols=0))
                ui.render(f"<div class='asof' style='margin:0.4rem 0 0.8rem 0'>Recomiendan comprar "
                          f"{fnum(rt.get('pct_buy_now'), '{:.0f}', '%')} hoy vs "
                          f"{fnum(rt.get('pct_buy_3m_ago'), '{:.0f}', '%')} hace 3 meses</div>")
            if m.get("target_mean"):
                ui.render(ui.table(["Objetivo bajo", "Objetivo medio", "Objetivo alto", "Upside al medio"],
                                   [[fnum(m.get("target_low")), fnum(m.get("target_mean")), fnum(m.get("target_high")),
                                     signed(up, "{:+.1f}")]], cell_classes=[["", "", "", tone(up)]], text_cols=0))
            if m.get("recent_rating_changes"):
                ui.render("<div style='height:0.6rem'></div>")
                ui.render(ui.table(["Fecha", "Firma", "Acción", "De", "A", "Objetivo"],
                                   [[r.get("date"), r.get("firm"), r.get("action"), r.get("from"), r.get("to"),
                                     fnum(r.get("price_target"))] for r in m["recent_rating_changes"]],
                                   text_cols=5))

    peer_cols = [("market_cap", "Mkt cap", None), ("pe_trailing", "P/E", "{:.1f}"), ("pe_forward", "P/E fwd", "{:.1f}"),
                 ("ev_ebitda", "EV/EBITDA", "{:.1f}"), ("price_to_sales", "P/S", "{:.2f}"), ("price_to_book", "P/B", "{:.2f}"),
                 ("fcf_yield_pct", "FCF yld %", "{:.2f}"), ("gross_margin_pct", "Mg bruto %", "{:.1f}"),
                 ("operating_margin_pct", "Mg op %", "{:.1f}"), ("net_margin_pct", "Mg neto %", "{:.1f}"),
                 ("roe_pct", "ROE %", "{:.1f}"), ("revenue_growth_yoy_pct", "Crec. vtas %", "{:+.1f}"),
                 ("net_debt_to_ebitda", "DN/EBITDA", "{:.2f}"), ("dividend_yield_pct", "Div %", "{:.2f}"),
                 ("return_52w_pct", "Ret 52s %", "{:+.1f}")]
    if p.get("table"):
        sub = "mediana de pares del mismo sector · TTM con el mismo método para todos"
        with panel("Comparables", sub):
            rows, rcls = [], []
            for r in p["table"]:
                rows.append([Markup(f"<b>{esc(r['ticker'])}</b>"), (r.get("name") or "")[:28], r.get("industry") or ""]
                            + [big(r.get(k)) if k == "market_cap" else cell(r.get(k), fmt) for k, _, fmt in peer_cols])
                rcls.append("hl" if r["ticker"] == result.ticker else "" if r.get("in_median", True) else "dim")
            med = p.get("peer_median") or {}
            rows.append(["Mediana", "", ""] + ["—" if k == "market_cap" else cell(med.get(k), fmt)
                                               for k, _, fmt in peer_cols])
            rcls.append("med")
            ui.render(ui.table(["Ticker", "Nombre", "Industria"] + [lbl for _, lbl, _ in peer_cols], rows,
                               row_classes=rcls, text_cols=3))
            ui.render("<div class='asof' style='margin-top:0.5rem'>Resaltado: empresa analizada · atenuado: otro sector "
                      "(fuera de la mediana)</div>")
        with panel("Lecturas de la comparación", "rankings precalculados"):
            ui.render(ui.facts_list(p.get("facts") or []))
    else:
        st.info("Sin comparables. Cargalos en la barra lateral o usá el motor CrewAI para que el agente los elija.")

# ---------------------------------------------------------------------------
# Simulación Monte Carlo y asistente (la simulación va primero en el código
# porque el asistente usa su resultado)
# ---------------------------------------------------------------------------
with tab_sim:
    simulation = simulation_view.render(result, panel, PLOTLY_CONFIG)

with pdf_slot:
    # Se genera recién al hacer clic (callable), no en cada recarga de la página.
    st.download_button("Descargar informe en PDF", data=lambda: build_pdf(result, simulation), type="primary",
                       width="stretch", icon=":material/picture_as_pdf:", mime="application/pdf",
                       file_name=f"informe_{result.ticker}_{m.get('last_date', '')}.pdf")

with tab_chat:
    assistant_view.render(result, panel, api_key, model.strip() or DEFAULT_OPENAI_MODEL,
                          dict(simulation.summary, facts=simulation.facts) if simulation else None)

# ---------------------------------------------------------------------------
# Noticias
# ---------------------------------------------------------------------------
with tab_news:
    if news.get("headlines"):
        sub = (f"{news['count']} titulares · 30 días · sentimiento léxico {news['sentiment_label']} "
               f"({news['sentiment_score']:+.2f}, orientativo) · búsqueda \"{news.get('query_name', '')}\"")
        with panel("Noticias", sub):
            ui.render(ui.news_list(news["headlines"]))
    else:
        st.info(news.get("error") or "Sin titulares recientes para este activo.")
    if news.get("source_errors"):
        ui.render(ui.note("Fuentes con error: " + "; ".join(news["source_errors"])))

# ---------------------------------------------------------------------------
# Razonamiento del agente
# ---------------------------------------------------------------------------
with tab_log:
    with panel("Razonamiento del agente", f"{len(result.steps)} pasos · motor {result.engine}"):
        ui.render("".join(ui.step_html(s.kind, s.title, s.detail) for s in result.steps))
