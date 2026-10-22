"""
ui/components.py
================
Piezas de interfaz en HTML (se dibujan con st.html, sin pasar por Markdown).

Todo texto que viene de afuera (titulares, nombres de empresas, salida del
agente) pasa por esc() antes de entrar al HTML.
"""

from __future__ import annotations

import html
import json
from typing import Any, Iterable

import streamlit as st

STEP_ICONS = {"thought": "◆", "tool_call": "▸", "tool_result": "✓", "error": "✕", "final": "■"}
REC_LABELS = {"strong_buy": "Compra fuerte", "buy": "Compra", "hold": "Mantener",
              "underperform": "Bajo rendimiento", "sell": "Venta", "strong_sell": "Venta fuerte"}


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def fnum(value: Any, fmt: str = "{:,.2f}", suffix: str = "") -> str:
    if value is None:
        return "N/D"
    try:
        return fmt.format(value) + suffix
    except (TypeError, ValueError):
        return esc(value)


def big(n: float | None) -> str:
    if n is None:
        return "N/D"
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= div:
            return f"{n / div:,.2f}{suf}"
    return f"{n:,.0f}"


def tone(value: float | None) -> str:
    if value is None:
        return ""
    return "up" if value > 0 else "down" if value < 0 else ""


def signed(value: float | None, fmt: str = "{:+.2f}", suffix: str = "%") -> str:
    """Variación con flecha: el color de estado nunca va solo."""
    if value is None:
        return "N/D"
    arrow = "▲" if value > 0 else "▼" if value < 0 else "■"
    return f"{arrow} {fmt.format(value)}{suffix}"


def render(markup: str) -> None:
    st.html(markup)


# -- estructura ----------------------------------------------------------------
def topbar(ticker: str | None = None, meta: dict[str, str] | None = None) -> str:
    items = "".join(f"<span>{esc(k)} <b>{esc(v)}</b></span>" for k, v in (meta or {}).items())
    sec = (f"<span class='title'>{esc(ticker)}<small>Informe de research</small></span>" if ticker else
           "<span class='title'>Agente de análisis financiero<small>Research de acciones con IA</small></span>")
    return (f"<div class='topbar'><div class='brand'><span class='logo'>AGF</span>{sec}</div>"
            f"<div class='meta'>{items}</div></div>")


def empty_state() -> str:
    return ("<div class='empty'><div class='big'>Listo para analizar</div>"
            "<p>Ingresá un ticker de Yahoo Finance en el panel izquierdo (por ejemplo VIST, GGAL, AAPL o "
            "YPFD.BA), elegí la temporalidad y presioná <b>Ejecutar análisis</b>. El agente reúne precios, "
            "estados contables, comparables y noticias, redacta el informe y lo verifica contra los datos.</p></div>")


def panel_header(title: str, sub: str = "") -> str:
    return f"<div class='panel-h'>{esc(title)}<span>{esc(sub)}</span></div>"


def note(text: str) -> str:
    return f"<div class='note'>{esc(text)}</div>"


def _range_bar(label: str, low: float, high: float, value: float, target: float | None = None,
               fmt: str = "{:,.2f}") -> str:
    span = high - low
    if not span or span <= 0:
        return ""

    def pos(x: float) -> float:
        return max(0.0, min(100.0, (x - low) / span * 100))

    target_mark = f"<div class='mark target' style='left:{pos(target):.1f}%' title='Objetivo medio'></div>" \
        if target is not None else ""
    return (f"<div class='range'><div class='lbl'>{esc(label)}</div>"
            f"<div class='track'><div class='fill' style='width:{pos(value):.1f}%'></div>"
            f"{target_mark}<div class='mark' style='left:{pos(value):.1f}%' title='Precio'></div></div>"
            f"<div class='ends'><span>{fmt.format(low)}</span><span>{fmt.format(high)}</span></div></div>")


def quote_header(m: dict[str, Any], ticker: str, period: str, engine: str) -> str:
    cur = m.get("currency", "")
    chg, prev, price = m.get("change_pct"), m.get("previous_close"), m.get("price")
    abs_chg = price - prev if price is not None and prev else None
    cls = tone(chg) or "flat"
    chips = "".join(f"<span class='chip'>{esc(x)}</span>"
                    for x in (m.get("quote_type"), m.get("sector"), m.get("industry"), m.get("country")) if x)
    left = (f"<div><div class='tk'>{esc(ticker)} · {esc(cur)}</div>"
            f"<div class='name'>{esc(m.get('name', ticker))}</div><div class='chips'>{chips}</div></div>")
    mid = (f"<div><div class='px'>{fnum(price)}<small>{esc(cur)}</small></div>"
           f"<div class='delta {cls}'>{signed(abs_chg, '{:+,.2f}', '')} ({fnum(chg, '{:+.2f}', '%')})</div>"
           f"<div class='asof'>Cierre {esc(m.get('last_date', 'N/D'))} · Ventana {esc(period)} · {esc(engine)}</div></div>")
    right = ""
    if m.get("low_52w_close") is not None and m.get("high_52w_close") is not None and price is not None:
        right += _range_bar("Rango 52 semanas (cierre)", m["low_52w_close"], m["high_52w_close"], price)
    if m.get("target_low") and m.get("target_high") and price is not None:
        lo, hi = min(m["target_low"], price), max(m["target_high"], price)
        right += _range_bar(f"Objetivos de analistas · medio {fnum(m.get('target_mean'))}", lo, hi, price,
                            m.get("target_mean"))
    return f"<div class='quote'>{left}{mid}<div>{right}</div></div>"


def kpi_grid(items: Iterable[tuple[str, str, str, str]]) -> str:
    """items: (etiqueta, valor, subtexto, tono 'up'|'down'|'')."""
    cells = "".join(f"<div class='kpi'><div class='k' title='{esc(k)}'>{esc(k)}</div>"
                    f"<div class='v {esc(cls)}'>{esc(v)}</div><div class='s' title='{esc(s)}'>{esc(s)}</div></div>"
                    for k, v, s, cls in items)
    return f"<div class='kpis'>{cells}</div>"


def signals(t: dict[str, Any], m: dict[str, Any]) -> str:
    pills: list[tuple[str, str, str]] = []
    trend = t.get("trend")
    if trend:
        pills.append(("Tendencia", trend.capitalize(),
                      "pos" if trend == "alcista" else "neg" if trend == "bajista" else "neu"))
    cross = (t.get("last_sma50_sma200_cross") or {})
    if cross.get("date"):
        golden = cross.get("direction") == "arriba"
        pills.append(("Último cruce", f"{'Golden' if golden else 'Death'} {cross['date']}", "pos" if golden else "neg"))
    if t.get("rsi14") is not None:
        zone = t.get("rsi_zone") or ""
        pills.append(("RSI 14", f"{t['rsi14']:.1f} {zone}", "neg" if zone == "sobrecompra" else
                      "pos" if zone == "sobreventa" else "neu"))
    if t.get("macd_state"):
        pills.append(("MACD", t["macd_state"].split(" ")[0], "pos" if t["macd_state"].startswith("alcista") else "neg"))
    if t.get("bollinger_position"):
        pills.append(("Bollinger", f"%B {t.get('bollinger_pct_b')}", "neu"))
    rec = m.get("recommendation")
    if rec:
        pills.append(("Consenso", REC_LABELS.get(rec, rec), "pos" if "buy" in rec else "neg" if "sell" in rec else "neu"))
    arrows = {"pos": "▲ ", "neg": "▼ ", "neu": ""}
    return "<div class='signals'>" + "".join(
        f"<span class='sig {c}'>{esc(k)}<b>{arrows[c]}{esc(v)}</b></span>" for k, v, c in pills) + "</div>"


def table(headers: list[str], rows: list[list[Any]], row_classes: list[str] | None = None,
          cell_classes: list[list[str]] | None = None, text_cols: int = 1) -> str:
    """Tabla HTML. Las primeras `text_cols` columnas son texto (izquierda); el resto, números."""
    head = "".join(f"<th class='{'l' if j < text_cols else ''}'>{esc(h)}</th>" for j, h in enumerate(headers))
    body = []
    for i, row in enumerate(rows):
        rc = row_classes[i] if row_classes else ""
        cells = "".join(
            f"<td class='{esc(' '.join(x for x in ('l' if j < text_cols else '', cell_classes[i][j] if cell_classes else '') if x))}'>"
            f"{c if isinstance(c, Markup) else esc(c)}</td>" for j, c in enumerate(row))
        body.append(f"<tr class='{esc(rc)}'>{cells}</tr>")
    return f"<div class='tbl-wrap'><table class='tbl'><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


class Markup(str):
    """Texto ya escapado que table() no vuelve a escapar."""


def facts_list(facts: Iterable[str]) -> str:
    return "<ul class='facts'>" + "".join(f"<li>{esc(f)}</li>" for f in facts) + "</ul>"


def news_list(headlines: list[dict[str, Any]]) -> str:
    items = []
    for h in headlines:
        title = esc(h.get("title"))
        if h.get("url"):
            title = f"<a href='{esc(h['url'])}' target='_blank' rel='noopener'>{title}</a>"
        score = h.get("score") or 0
        tag = ("<span class='tag pos'>▲ Positivo</span>" if score > 0 else
               "<span class='tag neg'>▼ Negativo</span>" if score < 0 else "<span class='tag'>● Neutro</span>")
        src = " · ".join(esc(x) for x in (h.get("publisher"), h.get("source")) if x)
        items.append(f"<div class='item'><div class='d'>{esc(h.get('date'))}</div>"
                     f"<div><div class='t'>{title}</div><div class='src'>{src}</div></div>{tag}</div>")
    return "<div class='news'>" + "".join(items) + "</div>"


def step_html(kind: str, title: str, detail: Any = None) -> str:
    det = ""
    if detail is not None:
        body = json.dumps(detail, ensure_ascii=False, indent=1, default=str) if isinstance(detail, (dict, list)) \
            else str(detail)
        det = f"<details><summary>ver detalle</summary><pre>{esc(body)}</pre></details>"
    # Los títulos traen `nombre_de_herramienta` en formato Markdown.
    tt = esc(title).replace("`", "")
    return (f"<div class='log'><div class='row {esc(kind)}'><span class='ic'>{STEP_ICONS.get(kind, '·')}</span>"
            f"<div><span class='tt'>{tt}</span>{det}</div></div></div>")


def qc_card(review: dict[str, Any] | None) -> str:
    if not review:
        return "<div class='qc'><span class='st'>Sin verificación (motor de reglas)</span></div>"
    fixes = review.get("corrections") or []
    pending = review.get("contradictions_after") or []
    if review.get("status") != "ok":
        head = "<span class='st bad'>✕ No se pudo verificar</span>"
    elif pending:
        head = f"<span class='st bad'>✕ {len(pending)} contradicciones sin resolver</span>"
    else:
        head = f"<span class='st ok'>✓ Verificado · {len(fixes)} cambios</span>"
    body = "".join(f"<div class='chg' style='border-color:var(--down)'>{esc(c)}</div>" for c in pending)
    body += "".join(f"<div class='chg'>{esc(c)}</div>" for c in fixes[:12])
    flagged = review.get("flagged_after") or []
    if flagged:
        body += ("<div style='margin-top:0.6rem;color:var(--muted)'>Cifras derivadas sin coincidencia exacta "
                 "(diferencias o porcentajes calculados):</div>")
        body += "".join(f"<div class='chg'>{esc(s)}</div>" for s in flagged[:8])
    return f"<div class='qc'>{head}{body}</div>"


def chat_meta(sources: list[str], labels: dict[str, str], flagged: list[str], contradictions: list[str]) -> str:
    """Pie de cada respuesta del asistente: fuentes citadas y resultado de la verificación."""
    chips = "".join(f"<span class='src'>{esc(labels.get(s, s))}</span>" for s in sources)
    if contradictions:
        badge = f"<span class='st bad'>✕ {len(contradictions)} contradicción(es) con los datos</span>"
        items = contradictions + flagged
    elif flagged:
        badge = f"<span class='st warn'>{len(flagged)} cifra(s) derivada(s) a revisar</span>"
        items = flagged
    else:
        badge = "<span class='st ok'>✓ Cifras verificadas contra los datos</span>"
        items = []
    det = ""
    if items:
        det = ("<details><summary>ver detalle</summary>" + "".join(f"<div class='chg'>{esc(i)}</div>" for i in items)
               + "</details>")
    src = f"<span class='lbl'>Fuentes</span>{chips}" if chips else ""
    return f"<div class='chat-meta'>{badge}{src}{det}</div>"
