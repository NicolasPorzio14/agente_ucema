"""
ui/theme.py
===========
Tema claro "fintech": tokens de color, CSS global y la plantilla de Plotly.

Criterios:
* Superficies claras (fondo gris azulado, paneles blancos) y un azul de marca
  con celeste como acento. El azul es interfaz (botones, pestaña activa,
  encabezados); en los gráficos solo aparece como la primera serie.
* Los colores de series tienen contraste >= 3:1 contra el blanco y se
  distinguen entre sí también con daltonismo (azul, naranja, verde azulado,
  ámbar, magenta, en ese orden fijo).
* Verde y rojo son colores de estado (suba / baja) y siempre van con ▲ ▼.
* Tipografía IBM Plex Sans con cifras tabulares; la monoespaciada queda solo
  para el registro del agente.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# -- tokens -------------------------------------------------------------------
PAGE = "#f3f6fa"
PANEL = "#ffffff"          # superficie de paneles y gráficos
PANEL_2 = "#f6f8fb"        # encabezados de tabla, inputs
LINE = "#e4e9f0"           # hairline: grilla, bordes
LINE_STRONG = "#cdd6e1"
INK = "#0f1d33"            # texto principal (azul noche)
INK_2 = "#3d4b60"          # texto secundario
MUTED = "#6a788c"          # ejes, etiquetas
ACCENT = "#1463d9"         # marca: azul
ACCENT_2 = "#0ea5e9"       # acento: celeste
NAVY = "#0b2552"

UP_TEXT = "#0d8a4f"        # estado: suba (siempre con ▲)
DOWN_TEXT = "#c8362d"      # estado: baja (siempre con ▼)
WARN = "#9a5b00"

# Series categóricas (orden fijo; nunca ciclar).
S1 = "#1f6fd6"   # azul
S2 = "#d4590f"   # naranja
S3 = "#0f9a72"   # verde azulado
S4 = "#b07f00"   # ámbar
S5 = "#c63d78"   # magenta
NEUTRAL_REF = "#8a96a8"    # referencias sin identidad (Bollinger, SMA20, benchmark)
CANDLE_UP = "#2c3e57"      # vela alcista: hueca, borde oscuro
CANDLE_DOWN = "#7d8ba0"    # vela bajista: rellena, gris azulado
VOLUME = "#cfd8e4"

FONT_SANS = "'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif"
FONT_MONO = "'IBM Plex Mono', ui-monospace, 'Cascadia Mono', Consolas, monospace"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {{
  --page: {PAGE}; --panel: {PANEL}; --panel-2: {PANEL_2};
  --line: {LINE}; --line-strong: {LINE_STRONG};
  --ink: {INK}; --ink-2: {INK_2}; --muted: {MUTED};
  --accent: {ACCENT}; --accent-2: {ACCENT_2}; --navy: {NAVY};
  --accent-soft: rgba(20,99,217,0.07);
  --up: {UP_TEXT}; --down: {DOWN_TEXT}; --warn: {WARN};
  --up-soft: rgba(13,138,79,0.09); --down-soft: rgba(200,54,45,0.08);
  --sans: {FONT_SANS}; --mono: {FONT_MONO};
  --radius: 12px;
  --shadow: 0 1px 2px rgba(15,29,51,0.04), 0 4px 14px rgba(15,29,51,0.05);
}}

/* -- base ------------------------------------------------------------------ */
html, body, .stApp, [data-testid="stAppViewContainer"] {{
  background: var(--page) !important; color: var(--ink); font-family: var(--sans);
}}
[data-testid="stHeader"] {{ background: transparent; height: 0; }}
[data-testid="stToolbar"], [data-testid="stDecoration"], footer, #MainMenu {{ display: none !important; }}
.block-container {{ padding: 1.1rem 1.8rem 3rem 1.8rem !important; max-width: 1560px !important; }}
h1, h2, h3, h4 {{ font-family: var(--sans); color: var(--ink); letter-spacing: -0.01em; }}
p, li {{ color: var(--ink-2); }}
a {{ color: var(--accent); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-thumb {{ background: var(--line-strong); border-radius: 4px; }}
::-webkit-scrollbar-track {{ background: transparent; }}

/* -- sidebar --------------------------------------------------------------- */
[data-testid="stSidebar"] {{ background: var(--panel) !important; border-right: 1px solid var(--line); }}
[data-testid="stSidebar"] .block-container {{ padding-top: 1rem !important; }}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
  font-size: 0.8rem !important; font-weight: 500 !important; color: var(--ink-2) !important;
}}
[data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="select"] > div {{
  background: var(--panel-2) !important; border-color: var(--line-strong) !important; color: var(--ink) !important;
}}

/* -- botones ---------------------------------------------------------------- */
.stButton > button, .stDownloadButton > button {{ border-radius: 8px !important; font-weight: 500 !important; }}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{
  background: linear-gradient(135deg, var(--accent), #2f86f0) !important; color: #fff !important; border: none !important;
  box-shadow: 0 2px 6px rgba(20,99,217,0.25);
}}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {{ filter: brightness(1.06); }}
.stButton > button[kind="primary"] p, .stDownloadButton > button[kind="primary"] p,
.stDownloadButton > button[kind="primary"] span {{ color: #fff !important; font-weight: 600 !important; }}
.stButton > button[kind="secondary"], .stDownloadButton > button[kind="secondary"] {{
  background: var(--panel) !important; color: var(--accent) !important; border: 1px solid rgba(20,99,217,0.35) !important;
}}
.stButton > button[kind="secondary"] p, .stDownloadButton > button[kind="secondary"] p {{ color: var(--accent) !important; }}
.stButton > button[kind="secondary"]:hover, .stDownloadButton > button[kind="secondary"]:hover {{ background: var(--accent-soft) !important; }}
.stDownloadButton {{ margin-bottom: 0.2rem; }}

/* -- pestañas ---------------------------------------------------------------- */
[role="tablist"] {{ border-bottom: 1px solid var(--line-strong); gap: 0.2rem !important; background: transparent; }}
[role="tab"] {{ padding: 0.55rem 0.95rem !important; }}
[role="tab"] p {{ font-size: 0.9rem !important; font-weight: 500 !important; color: var(--muted) !important; }}
[role="tab"]:hover p {{ color: var(--ink) !important; }}
[role="tab"][aria-selected="true"] p {{ color: var(--accent) !important; font-weight: 600 !important; }}
[role="tab"] .react-aria-SelectionIndicator {{ background: var(--accent) !important; height: 3px !important;
  border-radius: 3px 3px 0 0; }}
[data-baseweb="tab-highlight"] {{ background: var(--accent) !important; height: 3px !important; }}
[data-baseweb="tab-border"] {{ display: none; }}
[data-testid="stTabs"] [data-baseweb="tab-panel"] {{ padding-top: 1rem; }}

/* -- paneles (contenedores con borde) --------------------------------------- */
/* En Streamlit 1.66 el contenedor con borde es el propio stVerticalBlock; los
   paneles llevan key "panel-…" (o "report") y Streamlit agrega la clase st-key-… */
[class*="st-key-panel-"], .st-key-report {{
  background: var(--panel); border: 1px solid var(--line) !important; border-radius: var(--radius) !important;
  box-shadow: var(--shadow); padding: 1rem 1.1rem !important;
}}
[data-testid="stExpander"] details {{ background: var(--panel); border: 1px solid var(--line) !important;
  border-radius: 10px; }}
[data-testid="stExpander"] summary p {{ font-size: 0.85rem; font-weight: 500; color: var(--ink-2) !important; }}
[data-testid="stStatusWidget"], [data-testid="stStatus"] details {{ background: var(--panel) !important; }}
[data-testid="stAlert"] {{ border-radius: 10px; }}
[data-testid="stElementContainer"]:has(> .stHtml) {{ margin-bottom: 0; }}

/* -- informe (markdown del agente) ------------------------------------------ */
.st-key-report h3 {{
  font-size: 1.04rem; font-weight: 600; color: var(--navy); border-bottom: 1px solid var(--line);
  padding: 1.2rem 0 0.4rem 0; margin-bottom: 0.6rem;
}}
.st-key-report h3::before {{ content: ''; display: inline-block; width: 4px; height: 0.95em; margin-right: 0.55rem;
  border-radius: 2px; background: linear-gradient(180deg, var(--accent-2), var(--accent)); vertical-align: -0.1em; }}
.st-key-report h3:first-child {{ padding-top: 0.2rem; }}
.st-key-report p, .st-key-report li {{ color: var(--ink-2); font-size: 0.95rem; line-height: 1.65; }}
.st-key-report strong {{ color: var(--ink); }}
.st-key-report table {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; margin: 0.4rem 0 0.9rem 0;
  font-variant-numeric: tabular-nums; }}
.st-key-report th {{ font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;
  color: var(--muted); border-bottom: 1px solid var(--line-strong); text-align: left; padding: 7px 9px; background: var(--panel-2); }}
.st-key-report td {{ border-bottom: 1px solid var(--line); padding: 7px 9px; color: var(--ink-2); vertical-align: top; }}
.st-key-report hr {{ border-color: var(--line); }}

/* -- componentes HTML propios ------------------------------------------------ */
.topbar {{ display: flex; align-items: center; justify-content: space-between; gap: 1rem;
  background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); box-shadow: var(--shadow);
  padding: 0.6rem 1rem; margin-bottom: 1rem; }}
.topbar .brand {{ display: flex; align-items: center; gap: 0.7rem; }}
.logo {{ display: inline-grid; place-items: center; width: 34px; height: 34px; border-radius: 9px; color: #fff;
  font-weight: 700; font-size: 0.78rem; letter-spacing: 0.02em;
  background: linear-gradient(135deg, var(--navy), var(--accent) 60%, var(--accent-2)); }}
.topbar .title {{ color: var(--ink); font-size: 1rem; font-weight: 600; }}
.topbar .title small {{ display: block; color: var(--muted); font-size: 0.74rem; font-weight: 400; }}
.topbar .meta {{ display: flex; gap: 0.5rem; flex-wrap: wrap; }}
.topbar .meta span {{ color: var(--muted); font-size: 0.76rem; background: var(--panel-2); border: 1px solid var(--line);
  border-radius: 999px; padding: 3px 10px; }}
.topbar .meta b {{ color: var(--ink); font-weight: 600; }}

/* Cabecera de cotización: tarjeta destacada con degradé azul. */
.quote {{ display: grid; grid-template-columns: minmax(280px, 1.35fr) minmax(240px, 1fr) minmax(260px, 1.1fr);
  gap: 1.6rem; align-items: center; border-radius: 16px; padding: 1.3rem 1.5rem; margin-bottom: 1rem; color: #fff;
  background: radial-gradient(120% 140% at 100% 0%, rgba(14,165,233,0.55) 0%, rgba(14,165,233,0) 55%),
              linear-gradient(135deg, var(--navy) 0%, #123d86 55%, var(--accent) 100%);
  box-shadow: 0 10px 30px rgba(11,37,82,0.22); }}
.quote .tk {{ color: #bfe3ff; font-size: 0.82rem; font-weight: 600; letter-spacing: 0.06em; }}
.quote .name {{ font-size: 1.55rem; font-weight: 600; color: #fff; line-height: 1.2; margin: 0.15rem 0 0.5rem 0; }}
.quote .chips {{ display: flex; flex-wrap: wrap; gap: 0.35rem; }}
.chip {{ font-size: 0.72rem; color: #e4f1ff; background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.18);
  border-radius: 999px; padding: 2px 9px; }}
.quote .px {{ font-size: 2.5rem; font-weight: 600; color: #fff; line-height: 1; font-variant-numeric: tabular-nums; }}
.quote .px small {{ font-size: 0.95rem; color: #bfe3ff; font-weight: 500; margin-left: 0.35rem; }}
.delta {{ display: inline-block; font-size: 0.92rem; font-weight: 600; margin-top: 0.5rem; padding: 2px 10px;
  border-radius: 999px; font-variant-numeric: tabular-nums; }}
.delta.up {{ color: #b6f5d3; background: rgba(13,138,79,0.35); }}
.delta.down {{ color: #ffd2cc; background: rgba(200,54,45,0.35); }}
.delta.flat {{ color: #e4f1ff; background: rgba(255,255,255,0.12); }}
.asof {{ font-size: 0.74rem; color: var(--muted); margin-top: 0.35rem; }}
.quote .asof {{ color: #bcd3f0; }}

.range .lbl {{ font-size: 0.74rem; color: #cfe2fa; margin-bottom: 0.45rem; }}
.range .track {{ position: relative; height: 6px; background: rgba(255,255,255,0.18); border-radius: 3px; margin: 0.8rem 0 0.4rem 0; }}
.range .fill {{ position: absolute; left: 0; top: 0; bottom: 0; background: linear-gradient(90deg, rgba(255,255,255,0.35), #7cc8ff);
  border-radius: 3px; }}
.range .mark {{ position: absolute; top: -5px; width: 3px; height: 16px; background: #fff; border-radius: 2px; }}
.range .mark.target {{ background: #ffd166; }}
.range .ends {{ display: flex; justify-content: space-between; font-size: 0.76rem; color: #e4f1ff;
  font-variant-numeric: tabular-nums; }}
.range + .range {{ margin-top: 0.9rem; }}

.kpis {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 0.7rem; margin-bottom: 1rem; }}
.kpi {{ background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); box-shadow: var(--shadow);
  padding: 0.75rem 0.9rem; min-width: 0; }}
.kpi .k {{ font-size: 0.76rem; font-weight: 500; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.kpi .v {{ font-size: 1.3rem; font-weight: 600; color: var(--ink); margin-top: 0.2rem; white-space: nowrap;
  font-variant-numeric: tabular-nums; }}
.kpi .s {{ font-size: 0.74rem; color: var(--muted); margin-top: 0.15rem; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }}
.kpi .v.up {{ color: var(--up); }} .kpi .v.down {{ color: var(--down); }}

.signals {{ display: flex; flex-wrap: wrap; gap: 0.45rem; }}
.sig {{ font-size: 0.8rem; border: 1px solid var(--line); border-radius: 999px; padding: 4px 12px; color: var(--muted);
  background: var(--panel); white-space: nowrap; box-shadow: 0 1px 2px rgba(15,29,51,0.04); }}
.sig b {{ font-weight: 600; margin-left: 0.35rem; }}
.sig.pos {{ background: var(--up-soft); border-color: rgba(13,138,79,0.2); }} .sig.pos b {{ color: var(--up); }}
.sig.neg {{ background: var(--down-soft); border-color: rgba(200,54,45,0.2); }} .sig.neg b {{ color: var(--down); }}
.sig.neu b {{ color: var(--ink); }}

.panel-h {{ display: flex; align-items: baseline; justify-content: space-between; gap: 0.6rem; flex-wrap: wrap;
  font-size: 0.98rem; font-weight: 600; color: var(--ink); padding-bottom: 0.55rem; margin-bottom: 0.7rem;
  border-bottom: 1px solid var(--line); }}
.panel-h span {{ color: var(--muted); font-size: 0.76rem; font-weight: 400; }}

.tbl {{ width: 100%; border-collapse: collapse; font-size: 0.84rem; font-variant-numeric: tabular-nums; }}
.tbl th {{ color: var(--muted); font-weight: 600; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.04em;
  text-align: right; padding: 7px 9px; border-bottom: 1px solid var(--line-strong); background: var(--panel-2); white-space: nowrap; }}
.tbl th:first-child {{ border-top-left-radius: 8px; }} .tbl th:last-child {{ border-top-right-radius: 8px; }}
.tbl th:first-child, .tbl td:first-child, .tbl th.l, .tbl td.l {{ text-align: left; }}
.tbl td {{ color: var(--ink-2); text-align: right; padding: 6px 9px; border-bottom: 1px solid var(--line); white-space: nowrap; }}
.tbl tr:hover td {{ background: rgba(20,99,217,0.035); }}
.tbl tr.hl td {{ color: var(--ink); background: var(--accent-soft); font-weight: 500; }}
.tbl tr.hl td:first-child {{ box-shadow: inset 3px 0 0 var(--accent); color: var(--accent); }}
.tbl tr.med td {{ color: var(--ink); border-top: 1px solid var(--line-strong); font-weight: 600; background: var(--panel-2); }}
.tbl tr.dim td {{ opacity: 0.5; }}
.tbl td.up {{ color: var(--up); }} .tbl td.down {{ color: var(--down); }}
.tbl td.wrap {{ white-space: normal; text-align: left; min-width: 150px; color: var(--muted); font-size: 0.78rem; }}
.tbl-wrap {{ overflow-x: auto; }}

ul.facts {{ margin: 0 !important; padding: 0 !important; list-style: none; }}
.facts li {{ margin: 0 !important; font-size: 0.86rem; color: var(--ink-2); padding: 7px 0 7px 16px; border-bottom: 1px solid var(--line);
  position: relative; line-height: 1.5; }}
.facts li:last-child {{ border-bottom: none; }}
.facts li::before {{ content: ''; position: absolute; left: 2px; top: 0.95em; width: 6px; height: 6px; border-radius: 50%;
  background: var(--accent-2); }}

.news {{ display: flex; flex-direction: column; }}
.news .item {{ display: grid; grid-template-columns: 92px 1fr auto; gap: 0.9rem; align-items: start;
  padding: 0.75rem 0.2rem; border-bottom: 1px solid var(--line); }}
.news .item:last-child {{ border-bottom: none; }}
.news .d {{ font-size: 0.78rem; color: var(--muted); padding-top: 2px; font-variant-numeric: tabular-nums; }}
.news .t a, .news .t {{ color: var(--ink); font-size: 0.93rem; line-height: 1.45; font-weight: 500; }}
.news .src {{ font-size: 0.74rem; color: var(--muted); margin-top: 0.2rem; }}
.tag {{ font-size: 0.74rem; padding: 2px 9px; border-radius: 999px; white-space: nowrap; font-weight: 500;
  background: var(--panel-2); color: var(--ink-2); }}
.tag.pos {{ color: var(--up); background: var(--up-soft); }}
.tag.neg {{ color: var(--down); background: var(--down-soft); }}

.log {{ font-family: var(--mono); font-size: 0.76rem; }}
.log .row {{ display: grid; grid-template-columns: 22px 1fr; gap: 0.4rem; padding: 5px 0; border-bottom: 1px solid var(--line); }}
.log .ic {{ color: var(--muted); }}
.log .row.tool_call .ic {{ color: var(--accent); }} .log .row.tool_result .ic {{ color: var(--up); }}
.log .row.error .ic, .log .row.error .tt {{ color: var(--down); }}
.log .tt {{ color: var(--ink-2); }}
.log details summary {{ cursor: pointer; color: var(--accent); font-size: 0.72rem; list-style: none; }}
.log pre {{ background: var(--panel-2); border: 1px solid var(--line); border-radius: 8px; padding: 0.6rem; margin: 0.3rem 0;
  color: var(--ink-2); white-space: pre-wrap; word-break: break-word; font-size: 0.72rem; max-height: 260px; overflow: auto; }}

.qc {{ font-size: 0.82rem; color: var(--ink-2); }}
.qc .st {{ display: inline-block; padding: 3px 10px; border-radius: 999px; margin-bottom: 0.5rem; font-weight: 600;
  background: var(--panel-2); }}
.qc .st.ok {{ color: var(--up); background: var(--up-soft); }}
.qc .st.bad {{ color: var(--down); background: var(--down-soft); }}
.qc .chg {{ border-left: 3px solid var(--line-strong); padding: 4px 0 4px 9px; margin: 6px 0; line-height: 1.45; }}

/* -- asistente --------------------------------------------------------------- */
[data-testid="stChatMessage"] {{ background: transparent; padding: 0.6rem 0.2rem; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{ background: var(--panel-2); border-radius: 12px;
  padding: 0.6rem 0.8rem; }}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {{ font-size: 0.93rem; line-height: 1.6; color: var(--ink-2); }}
[data-testid="stChatInput"] {{ border-radius: 12px; }}
.chat-meta {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem; margin-top: 0.4rem; font-size: 0.76rem; }}
.chat-meta .st {{ padding: 2px 9px; border-radius: 999px; font-weight: 600; background: var(--panel-2); color: var(--ink-2); }}
.chat-meta .st.ok {{ color: var(--up); background: var(--up-soft); }}
.chat-meta .st.bad {{ color: var(--down); background: var(--down-soft); }}
.chat-meta .st.warn {{ color: var(--warn); background: #fff4dc; }}
.chat-meta .lbl {{ color: var(--muted); margin-left: 0.4rem; }}
.chat-meta .src {{ color: var(--accent); background: var(--accent-soft); border-radius: 999px; padding: 2px 9px; }}
.chat-meta details {{ flex-basis: 100%; color: var(--ink-2); }}
.chat-meta summary {{ cursor: pointer; color: var(--accent); }}
.chat-meta .chg {{ border-left: 3px solid var(--line-strong); padding: 3px 0 3px 9px; margin: 5px 0; }}
.st-key-suggest .stButton > button {{ width: 100%; justify-content: flex-start; text-align: left; border-radius: 10px !important;
  background: var(--panel-2) !important; border: 1px solid var(--line) !important; }}
.st-key-suggest .stButton > button > div {{ justify-content: flex-start; width: 100%; }}
.st-key-suggest .stButton > button p {{ color: var(--ink-2) !important; font-size: 0.85rem; text-align: left; }}
[data-testid="stChatMessageAvatarAssistant"] {{ background: linear-gradient(135deg, var(--accent), var(--accent-2)); color: #fff; }}
[data-testid="stChatMessageAvatarUser"] {{ background: var(--navy); color: #fff; }}
.st-key-suggest .stButton > button:hover {{ border-color: rgba(20,99,217,0.4) !important; background: var(--accent-soft) !important; }}

.note {{ font-size: 0.8rem; color: var(--warn); border: 1px solid #f1d8a8; background: #fff8ea; border-radius: 10px;
  padding: 7px 12px; margin-bottom: 0.5rem; }}
.empty {{ text-align: center; padding: 3.5rem 1rem; border: 1px dashed var(--line-strong); border-radius: 16px;
  background: var(--panel); }}
.empty .big {{ font-size: 1.2rem; font-weight: 600; color: var(--ink); }}
.empty .big::before {{ content: ''; display: block; width: 48px; height: 48px; margin: 0 auto 0.9rem auto; border-radius: 14px;
  background: linear-gradient(135deg, var(--navy), var(--accent) 60%, var(--accent-2)); }}
.empty p {{ color: var(--muted); font-size: 0.93rem; max-width: 580px; margin: 0.7rem auto 0 auto; line-height: 1.55; }}
.side-brand {{ display: flex; align-items: center; gap: 0.6rem; font-size: 0.95rem; font-weight: 600; color: var(--ink);
  border-bottom: 1px solid var(--line); padding-bottom: 0.7rem; margin-bottom: 0.7rem; }}
.side-brand .logo {{ width: 28px; height: 28px; border-radius: 8px; font-size: 0.66rem; }}
.side-foot {{ font-size: 0.74rem; color: var(--muted); line-height: 1.5; margin-top: 0.9rem; }}

@media (max-width: 1100px) {{
  .kpis {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
  .quote {{ grid-template-columns: 1fr; }}
}}
@media (max-width: 640px) {{
  .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .news .item {{ grid-template-columns: 1fr; gap: 0.3rem; }}
  .topbar {{ flex-direction: column; align-items: flex-start; }}
}}
</style>
"""


def inject_css() -> None:
    st.html(CSS)


# -- Plotly -------------------------------------------------------------------
def style_figure(fig: go.Figure, height: int = 360, legend: bool = True) -> go.Figure:
    """Plantilla común: superficie blanca, grilla hairline sólida, tipografía del tema."""
    fig.update_layout(
        height=height,
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(family=FONT_SANS, color=INK_2, size=12),
        margin=dict(l=8, r=8, t=30 if legend else 10, b=8),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=PANEL, bordercolor=LINE_STRONG, font=dict(family=FONT_SANS, color=INK, size=12)),
        legend=dict(orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom",
                    font=dict(family=FONT_SANS, size=11, color=INK_2), bgcolor="rgba(0,0,0,0)"),
        showlegend=legend,
        barcornerradius=4,
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE_STRONG, tickcolor=LINE_STRONG, zeroline=False,
                     tickfont=dict(family=FONT_SANS, size=11, color=MUTED),
                     showspikes=True, spikemode="across", spikethickness=1, spikecolor=LINE_STRONG,
                     spikedash="solid")
    fig.update_yaxes(showgrid=True, gridcolor=LINE, gridwidth=1, linecolor=LINE, zeroline=False,
                     tickfont=dict(family=FONT_SANS, size=11, color=MUTED), side="right")
    return fig
