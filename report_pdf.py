"""
report_pdf.py
=============
Exporta el informe a PDF con la estética de la app (fpdf2, Python puro).

Contenido: portada con cotización y métricas clave, gráfico de precio con
medias móviles, el informe del agente (Markdown -> PDF), la simulación Monte
Carlo si se corrió, el control de calidad y las fuentes. Los gráficos se
dibujan como vectores del PDF: no hacen falta matplotlib ni un navegador.

Uso:  pdf_bytes = build_pdf(result, simulation)
"""

from __future__ import annotations

import os
import re
from datetime import date
from typing import Any

import pandas as pd
from fontTools.ttLib import TTFont
from fpdf import FPDF
from fpdf.enums import TableCellFillMode, XPos, YPos

from agent import AnalysisResult

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fonts")
FONT = "Plex"

# Paleta del tema claro (ui/theme.py), en RGB.
NAVY = (11, 37, 82)
ACCENT = (20, 99, 217)
ACCENT_2 = (14, 165, 233)
INK = (15, 29, 51)
INK_2 = (61, 75, 96)
MUTED = (106, 120, 140)
LINE = (228, 233, 240)
PANEL_2 = (246, 248, 251)
UP = (13, 138, 79)
DOWN = (200, 54, 45)
S1, S2, S3 = (31, 111, 214), (212, 89, 15), (15, 154, 114)
NEUTRAL = (138, 150, 168)
BAND_OUT, BAND_IN = (221, 233, 249), (184, 208, 242)

# Símbolos de la interfaz que la tipografía no trae; el signo del número ya informa.
_SYMBOLS = str.maketrans({"▲": "", "▼": "", "■": "", "●": "•", "◆": "•", "✕": "x", "→": "->", " ": " "})


class _ReportPDF(FPDF):
    def __init__(self, ticker: str):
        super().__init__(format="A4", unit="mm")
        self.ticker = ticker
        self.set_margins(16, 16, 16)
        self.set_auto_page_break(True, margin=18)
        self.add_font(FONT, "", os.path.join(FONT_DIR, "IBMPlexSans-Regular.ttf"))
        self.add_font(FONT, "B", os.path.join(FONT_DIR, "IBMPlexSans-Bold.ttf"))
        self.add_font(FONT, "I", os.path.join(FONT_DIR, "IBMPlexSans-Italic.ttf"))
        self.add_font(FONT, "BI", os.path.join(FONT_DIR, "IBMPlexSans-Bold.ttf"))
        self._cmap = set(TTFont(os.path.join(FONT_DIR, "IBMPlexSans-Regular.ttf")).getBestCmap())
        self.set_title(f"Informe de research {ticker}")
        self.set_author("Agente de análisis financiero")

    def clean(self, text: Any) -> str:
        s = str(text if text is not None else "").translate(_SYMBOLS)
        return "".join(ch if ord(ch) in self._cmap or ch in "\n\t" else "" for ch in s).strip()

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font(FONT, "B", 8)
        self.set_text_color(*ACCENT)
        self.cell(0, 5, f"AGF · {self.ticker}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*LINE)
        self.line(self.l_margin, self.get_y() + 1, self.w - self.r_margin, self.get_y() + 1)
        self.ln(5)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font(FONT, "", 7)
        self.set_text_color(*MUTED)
        self.cell(0, 4, "Fines educativos. No es asesoramiento de inversión. Datos: Yahoo Finance y Google News.")
        self.set_x(self.l_margin)
        self.cell(0, 4, f"Página {self.page_no()}/{{nb}}", align="R")


# ---------------------------------------------------------------------------
# Piezas
# ---------------------------------------------------------------------------

def _fmt(v: Any, fmt: str = "{:,.2f}", suffix: str = "") -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "N/D"
    try:
        return fmt.format(v) + suffix
    except (TypeError, ValueError):
        return str(v)


def _big(n: float | None) -> str:
    if n is None:
        return "N/D"
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(n) >= div:
            return f"{n / div:,.2f}{suf}"
    return f"{n:,.0f}"


def _cover(pdf: _ReportPDF, r: AnalysisResult) -> None:
    m = r.market or {}
    cur = m.get("currency", "")
    # Banda superior con degradé azul (franjas finas: fpdf2 no necesita patrones para esto).
    top_h = 46
    steps = 60
    for i in range(steps):
        t = i / (steps - 1)
        col = tuple(int(NAVY[k] + (ACCENT[k] - NAVY[k]) * t) for k in range(3))
        pdf.set_fill_color(*col)
        pdf.rect(pdf.w * i / steps, 0, pdf.w / steps + 0.3, top_h, style="F")
    pdf.set_xy(16, 10)
    pdf.set_font(FONT, "B", 9)
    pdf.set_text_color(191, 227, 255)
    pdf.cell(0, 5, f"INFORME DE RESEARCH · {r.ticker} · {cur}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(FONT, "B", 19)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, pdf.clean(m.get("name") or r.ticker)[:60], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(FONT, "", 9)
    pdf.set_text_color(228, 241, 255)
    chips = " · ".join(str(x) for x in (m.get("sector"), m.get("industry"), m.get("country")) if x)
    pdf.cell(0, 5, pdf.clean(chips), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 5, pdf.clean(f"Cierre {m.get('last_date', 'N/D')} · ventana {r.period} · motor {r.engine} · "
                             f"emitido {date.today():%Y-%m-%d}"))
    # Precio a la derecha
    price, chg = m.get("price"), m.get("change_pct")
    pdf.set_xy(pdf.w - 76, 12)
    pdf.set_font(FONT, "B", 24)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(60, 11, f"{_fmt(price)} {cur}", align="R", new_x=XPos.LEFT, new_y=YPos.NEXT)
    if chg is not None:
        pdf.set_font(FONT, "B", 10)
        pdf.set_text_color(*((182, 245, 211) if chg >= 0 else (255, 210, 204)))
        pdf.cell(60, 6, f"{chg:+.2f}% en el día", align="R")
    pdf.set_y(top_h + 6)


def _kpis(pdf: _ReportPDF, r: AnalysisResult) -> None:
    m, t = r.market or {}, r.technicals or {}
    rets = t.get("returns_pct") or {}
    items = [
        ("Market cap", _big(m.get("market_cap"))), ("P/E trailing", _fmt(m.get("pe_trailing"), "{:.1f}", "x")),
        ("P/E forward", _fmt(m.get("pe_forward"), "{:.1f}", "x")), ("EV/EBITDA TTM", _fmt(m.get("ev_ebitda"), "{:.1f}", "x")),
        ("FCF yield TTM", _fmt(m.get("fcf_yield_pct"), "{:.2f}", "%")), ("ROE TTM", _fmt(m.get("roe_ttm_pct"), "{:.1f}", "%")),
        ("Mg. operativo TTM", _fmt(m.get("operating_margin_ttm_pct"), "{:.1f}", "%")),
        ("Deuda neta/EBITDA", _fmt(m.get("net_debt_to_ebitda_ttm"), "{:.2f}", "x")),
        ("Upside obj. medio", _fmt(m.get("upside_to_target_mean_pct"), "{:+.1f}", "%")),
        ("Retorno 1Y", _fmt(rets.get("1Y"), "{:+.1f}", "%")), ("Tendencia", (t.get("trend") or "N/D").capitalize()),
        ("RSI 14", _fmt(t.get("rsi14"), "{:.1f}")),
    ]
    cols, gap = 6, 2.5
    w = (pdf.w - pdf.l_margin - pdf.r_margin - gap * (cols - 1)) / cols
    h = 14
    y0 = pdf.get_y()
    for i, (k, v) in enumerate(items):
        x = pdf.l_margin + (i % cols) * (w + gap)
        y = y0 + (i // cols) * (h + gap)
        pdf.set_fill_color(*PANEL_2)
        pdf.set_draw_color(*LINE)
        pdf.rect(x, y, w, h, style="DF", round_corners=True, corner_radius=1.6)
        pdf.set_xy(x + 2.5, y + 2)
        pdf.set_font(FONT, "", 6.8)
        pdf.set_text_color(*MUTED)
        pdf.cell(w - 4, 3.5, k)
        pdf.set_xy(x + 2.5, y + 6.3)
        pdf.set_font(FONT, "B", 11)
        sign_col = UP if v.startswith("+") else DOWN if v.startswith("-") and k in ("Upside obj. medio", "Retorno 1Y") else INK
        pdf.set_text_color(*sign_col)
        pdf.cell(w - 4, 6, pdf.clean(v))
    pdf.set_y(y0 + 2 * (h + gap) + 2)


def _axes(pdf: _ReportPDF, x: float, y: float, w: float, h: float, lo: float, hi: float) -> None:
    pdf.set_draw_color(*LINE)
    pdf.set_line_width(0.15)
    pdf.set_font(FONT, "", 6.5)
    pdf.set_text_color(*MUTED)
    for i in range(5):
        v = lo + (hi - lo) * i / 4
        yy = y + h - h * i / 4
        pdf.line(x, yy, x + w, yy)
        pdf.set_xy(x + w + 1, yy - 1.5)
        pdf.cell(12, 3, f"{v:,.2f}" if hi - lo < 50 else f"{v:,.0f}")


def _scale(values, lo, hi, y, h):
    return [y + h - (v - lo) / (hi - lo) * h for v in values]


def _legend(pdf: _ReportPDF, x: float, y: float, items: list[tuple[str, tuple]]) -> None:
    pdf.set_font(FONT, "", 7)
    for label, col in items:
        pdf.set_fill_color(*col)
        pdf.rect(x, y + 1, 4, 1.4, style="F")
        pdf.set_xy(x + 5, y)
        pdf.set_text_color(*INK_2)
        pdf.cell(pdf.get_string_width(label) + 2, 3.5, label)
        x += pdf.get_string_width(label) + 10


def _section_title(pdf: _ReportPDF, text: str) -> None:
    if pdf.get_y() > pdf.h - 40:
        pdf.add_page()
    pdf.ln(2)
    y = pdf.get_y()
    pdf.set_fill_color(*ACCENT)
    pdf.rect(pdf.l_margin, y + 1, 1.2, 4.6, style="F")
    pdf.set_x(pdf.l_margin + 3.5)
    pdf.set_font(FONT, "B", 11.5)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 6.5, pdf.clean(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(*LINE)
    pdf.line(pdf.l_margin, pdf.get_y() + 0.5, pdf.w - pdf.r_margin, pdf.get_y() + 0.5)
    pdf.ln(2.5)


def _price_chart(pdf: _ReportPDF, r: AnalysisResult) -> None:
    df = r.chart_df
    if df is None or df.empty:
        return
    _section_title(pdf, f"Precio y medias móviles · ventana {r.period}")
    x, y = pdf.l_margin, pdf.get_y() + 5
    w, h = pdf.w - pdf.l_margin - pdf.r_margin - 14, 58
    series = [("Cierre", df["Close"], INK, 0.45), ("SMA 50", df["SMA50"], S1, 0.5), ("SMA 200", df["SMA200"], S2, 0.5)]
    vals = pd.concat([s for _, s, _, _ in series], axis=1)
    lo, hi = float(vals.min().min()), float(vals.max().max())
    pad = (hi - lo) * 0.05 or 1
    lo, hi = lo - pad, hi + pad
    _legend(pdf, x, y - 5, [(n, c) for n, s, c, _ in series if s.notna().any()])
    _axes(pdf, x, y, w, h, lo, hi)
    n = len(df)
    for _, s, col, lw in series:
        pts = [(x + w * i / max(n - 1, 1), yy) for i, yy in enumerate(_scale(s.to_numpy(), lo, hi, y, h))
               if pd.notna(s.iloc[i])]
        if len(pts) > 1:
            pdf.set_draw_color(*col)
            pdf.set_line_width(lw)
            pdf.polyline(pts)
    pdf.set_font(FONT, "", 6.5)
    pdf.set_text_color(*MUTED)
    pdf.set_xy(x, y + h + 1)
    pdf.cell(30, 3, f"{df.index[0]:%Y-%m-%d}")
    pdf.set_xy(x + w - 30, y + h + 1)
    pdf.cell(30, 3, f"{df.index[-1]:%Y-%m-%d}", align="R")
    pdf.set_y(y + h + 6)


def _fan_chart(pdf: _ReportPDF, sim) -> None:
    x, y = pdf.l_margin, pdf.get_y() + 5
    w, h = pdf.w - pdf.l_margin - pdf.r_margin - 14, 52
    b = sim.bands
    s = sim.summary
    levels = [lvl for lvl in ((s.get("target") or {}).get("price"), (s.get("support") or {}).get("price")) if lvl]
    lo = min([float(b[5].min())] + levels)
    hi = max([float(b[95].max())] + levels)
    pad = (hi - lo) * 0.05
    lo, hi = lo - pad, hi + pad
    _legend(pdf, x, y - 5, [("Percentil 5 a 95", BAND_OUT), ("Percentil 25 a 75", BAND_IN), ("Mediana", S1),
                            ("Objetivo", S3), ("Soporte", S2)])
    _axes(pdf, x, y, w, h, lo, hi)
    n = len(b[50])
    xs = [x + w * i / (n - 1) for i in range(n)]
    for lo_q, hi_q, col in ((5, 95, BAND_OUT), (25, 75, BAND_IN)):
        top = list(zip(xs, _scale(b[hi_q], lo, hi, y, h)))
        bot = list(zip(xs, _scale(b[lo_q], lo, hi, y, h)))[::-1]
        pdf.set_fill_color(*col)
        pdf.polygon(top + bot, style="F")
    pdf.set_draw_color(*S1)
    pdf.set_line_width(0.6)
    pdf.polyline(list(zip(xs, _scale(b[50], lo, hi, y, h))))
    for key, col in (("target", S3), ("support", S2)):
        lvl = (s.get(key) or {}).get("price")
        if lvl:
            yy = _scale([lvl], lo, hi, y, h)[0]
            pdf.set_draw_color(*col)
            pdf.set_line_width(0.4)
            pdf.set_dash_pattern(dash=1, gap=1)
            pdf.line(x, yy, x + w, yy)
            pdf.set_dash_pattern()
    pdf.set_font(FONT, "", 6.5)
    pdf.set_text_color(*MUTED)
    pdf.set_xy(x, y + h + 1)
    pdf.cell(30, 3, f"Hoy {sim.dates[0]:%Y-%m-%d}")
    pdf.set_xy(x + w - 30, y + h + 1)
    pdf.cell(30, 3, f"{sim.dates[-1]:%Y-%m-%d}", align="R")
    pdf.set_y(y + h + 6)


# ---------------------------------------------------------------------------
# Markdown del informe
# ---------------------------------------------------------------------------

_ITALIC = re.compile(r"(?<![*\w])\*(?!\*)([^*\n]+?)(?<!\*)\*(?![*\w])")
_ITALIC_US = re.compile(r"(?<!\w)_([^_\n]+?)_(?!\w)")


def _inline(pdf: _ReportPDF, text: str) -> str:
    """Markdown inline -> marcado de fpdf2 (**negrita**, __itálica__)."""
    s = pdf.clean(text).replace("--", "–").replace("__", "_")
    s = _ITALIC_US.sub(r"__\1__", _ITALIC.sub(r"__\1__", s))
    return re.sub(r"`([^`]+)`", r"\1", s)


def _paragraph(pdf: _ReportPDF, text: str, size: float = 9.3) -> None:
    pdf.set_font(FONT, "", size)
    pdf.set_text_color(*INK_2)
    pdf.multi_cell(0, 4.9, _inline(pdf, text), markdown=True, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1.2)


def _bullet(pdf: _ReportPDF, text: str, marker: str = "•", indent: float = 0) -> None:
    pdf.set_font(FONT, "", 9.3)
    pdf.set_text_color(*ACCENT)
    pdf.set_x(pdf.l_margin + indent)
    pdf.cell(5, 4.9, marker)
    pdf.set_text_color(*INK_2)
    pdf.multi_cell(0, 4.9, _inline(pdf, text), markdown=True, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(0.6)


def _table(pdf: _ReportPDF, rows: list[list[str]]) -> None:
    if not rows:
        return
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]
    numeric = [all(re.fullmatch(r"[-+−]?[\d.,]+\s*[%x]?|N/D|—|-", c.strip() or "-") for c in col[1:])
               for col in zip(*rows)]
    align = tuple("RIGHT" if num and j > 0 else "LEFT" for j, num in enumerate(numeric))
    pdf.set_font(FONT, "", 8)
    pdf.set_text_color(*INK_2)
    pdf.set_draw_color(*LINE)
    with pdf.table(text_align=align, line_height=4.6, padding=1.4, markdown=True, borders_layout="HORIZONTAL_LINES",
                   cell_fill_color=PANEL_2, cell_fill_mode=TableCellFillMode.ROWS,
                   headings_style=_headings_style(), first_row_as_headings=True) as table:
        for row in rows:
            tr = table.row()
            for c in row:
                tr.cell(_inline(pdf, c))
    pdf.ln(2)


def _headings_style():
    from fpdf.fonts import FontFace
    return FontFace(family=FONT, emphasis="BOLD", color=MUTED, fill_color=(236, 241, 248), size_pt=7.5)


def _markdown(pdf: _ReportPDF, md: str) -> None:
    lines = md.splitlines()
    i = 0
    para: list[str] = []

    def flush():
        if para:
            _paragraph(pdf, " ".join(para))
            para.clear()

    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if not line or re.fullmatch(r"-{3,}|\*{3,}", line):
            flush()
        elif line.startswith("#"):
            flush()
            level = len(line) - len(line.lstrip("#"))
            title = line.lstrip("#").strip().strip("*")
            if level <= 3:
                _section_title(pdf, title)
            else:
                pdf.set_font(FONT, "B", 9.8)
                pdf.set_text_color(*INK)
                pdf.multi_cell(0, 5.2, pdf.clean(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        elif line.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    rows.append(cells)
                i += 1
            _table(pdf, rows)
            continue
        elif re.match(r"^[-*+]\s+", line):
            flush()
            indent = 5 if len(raw) - len(raw.lstrip()) >= 2 else 0
            _bullet(pdf, re.sub(r"^[-*+]\s+", "", line), "•" if not indent else "–", indent)
        elif re.match(r"^\d+[.)]\s+", line):
            flush()
            num = re.match(r"^(\d+)[.)]", line).group(1)
            _bullet(pdf, re.sub(r"^\d+[.)]\s+", "", line), f"{num}.")
        else:
            para.append(line)
        i += 1
    flush()


# ---------------------------------------------------------------------------
# Documento
# ---------------------------------------------------------------------------

def build_pdf(result: AnalysisResult, sim=None) -> bytes:
    """Arma el PDF. sim: SimulationResult (opcional) de la pestaña Simulación."""
    pdf = _ReportPDF(result.ticker)
    pdf.add_page()
    _cover(pdf, result)
    _kpis(pdf, result)
    _price_chart(pdf, result)
    _markdown(pdf, result.diagnosis)

    if sim is not None:
        s = sim.summary
        _section_title(pdf, f"Simulación Monte Carlo · horizonte {s['horizon']}")
        _paragraph(pdf, f"{s['method']}, {s['drift'].lower()}, {s['paths']:_} trayectorias.".replace("_", "."), 8.5)
        if pdf.get_y() > pdf.h - 90:
            pdf.add_page()
        _fan_chart(pdf, sim)
        for f in sim.facts:
            _bullet(pdf, f)

    review = result.review
    _section_title(pdf, "Control de calidad")
    if not review:
        _paragraph(pdf, "Informe generado por el motor de reglas cuantitativas: todas las cifras salen directamente "
                        "de los datos, sin redacción de un LLM.")
    else:
        pending = review.get("contradictions_after") or []
        fixes = review.get("corrections") or []
        flagged = review.get("flagged_after") or []
        _paragraph(pdf, f"El borrador del LLM se verificó contra los datos de las herramientas: {len(fixes)} cambios "
                        f"aplicados, {len(pending)} contradicciones pendientes y {len(flagged)} cifras derivadas sin "
                        "coincidencia exacta (diferencias o porcentajes calculados).")
        for c in pending:
            _bullet(pdf, f"Contradicción pendiente: {c}")

    _section_title(pdf, "Fuentes y metodología")
    for txt in ("Precios, estados contables, consenso y perfil: Yahoo Finance (yfinance). Noticias: Yahoo Finance y "
                "Google News (español e inglés).",
                "TTM = suma de los últimos 4 trimestres reportados; FY = año fiscal; MRQ = último balance trimestral. "
                "Los múltiplos que mezclarían monedas (empresas que reportan en otra moneda) se omiten.",
                "Los indicadores técnicos, rankings y lecturas ('facts') se calculan en Python; el LLM solo redacta y un "
                "verificador determinístico más un revisor cotejan el texto contra los datos."):
        _bullet(pdf, txt)
    return bytes(pdf.output())
