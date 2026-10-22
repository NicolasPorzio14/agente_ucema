"""
ui/charts.py
============
Gráficos Plotly con el tema claro de la app.

Decisiones de la skill de data viz:
* Velas sin tono: alcista hueca con borde oscuro, bajista rellena en gris. El
  verde/rojo clásico no pasa la separación para daltonismo (ΔE 4.1) y además
  choca con el naranja de la SMA 200; el relleno distingue sin depender del color.
* Medias con identidad (SMA 50 azul, SMA 200 naranja) validadas todos-contra-
  todos; SMA 20 y Bollinger son referencia y van en gris neutro.
* Un solo eje Y por gráfico: ventas y margen trimestral van en gráficos separados.
* Grilla hairline sólida; los umbrales de RSI (30/70) son las únicas líneas
  punteadas, porque sí son umbrales.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui.theme import (CANDLE_DOWN, CANDLE_UP, FONT_SANS, INK, LINE, LINE_STRONG, MUTED, NEUTRAL_REF,
                      PANEL, S1, S2, S3, S4, S5, VOLUME, style_figure)


def _end_label(fig: go.Figure, x, y: float, text: str, color: str, row: int | None = None,
               yshift: int = 0) -> None:
    """Etiqueta directa al final de la serie: texto en tinta, borde con el color de la serie."""
    kwargs = dict(row=row, col=1) if row else {}
    fig.add_annotation(x=x, y=y, text=text, showarrow=False, xanchor="left", xshift=6, yshift=yshift,
                       font=dict(family=FONT_SANS, size=11, color=INK),
                       bgcolor=PANEL, bordercolor=color, borderwidth=1, borderpad=2, **kwargs)


def _end_labels(fig: go.Figure, x, labels: list[tuple[float, str, str]], span: float,
                plot_px: float, row: int | None = None) -> None:
    """Varias etiquetas de fin de serie sin superponerse.

    Convierte cada valor a píxeles aproximados (plot_px = alto del área del
    gráfico) y corre hacia abajo cada etiqueta lo justo para quedar al menos
    19 px (alto de una etiqueta) debajo de la anterior, ya corrida.
    """
    if not span:
        span = 1.0
    prev_px = None
    for y, text, color in sorted(labels, key=lambda l: l[0], reverse=True):
        natural = -y / span * plot_px            # más abajo = mayor valor en píxeles
        placed = natural if prev_px is None else max(natural, prev_px + 19)
        _end_label(fig, x, y, text, color, row=row, yshift=int(round(natural - placed)))
        prev_px = placed


def technical_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.035,
                        row_heights=[0.55, 0.12, 0.165, 0.165],
                        subplot_titles=("", "Volumen", "RSI 14", "MACD 12 · 26 · 9"))
    x = df.index
    # Bandas de Bollinger como área de referencia.
    fig.add_trace(go.Scatter(x=x, y=df["BBUpper"], line=dict(color=NEUTRAL_REF, width=0.8),
                             name="Bollinger 20·2", legendgroup="bb", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["BBLower"], line=dict(color=NEUTRAL_REF, width=0.8),
                             fill="tonexty", fillcolor="rgba(138,150,168,0.12)", name="Bollinger inf.",
                             legendgroup="bb", showlegend=False, hoverinfo="skip"), row=1, col=1)
    if {"Open", "High", "Low"} <= set(df.columns):
        fig.add_trace(go.Candlestick(
            x=x, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Precio",
            increasing=dict(line=dict(color=CANDLE_UP, width=1), fillcolor=PANEL),
            decreasing=dict(line=dict(color=CANDLE_DOWN, width=1), fillcolor=CANDLE_DOWN),
            whiskerwidth=0.3), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=x, y=df["Close"], name="Cierre", line=dict(color=INK, width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["SMA20"], name="SMA 20", line=dict(color=NEUTRAL_REF, width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["SMA50"], name="SMA 50", line=dict(color=S1, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["SMA200"], name="SMA 200", line=dict(color=S2, width=2)), row=1, col=1)
    last = df.iloc[-1]
    labels = [(float(last["Close"]), f"{last['Close']:,.2f}", INK)]
    for col, color, label in (("SMA50", S1, "SMA50"), ("SMA200", S2, "SMA200")):
        if pd.notna(last[col]):
            labels.append((float(last[col]), f"{label} {last[col]:,.2f}", color))
    low = df["Low"] if "Low" in df else df["Close"]
    high = df["High"] if "High" in df else df["Close"]
    _end_labels(fig, x[-1], labels, float(high.max() - low.min()), plot_px=0.55 * 700, row=1)

    # Volumen: una sola serie, un solo color.
    fig.add_trace(go.Bar(x=x, y=df["Volume"], marker_color=VOLUME, name="Volumen", showlegend=False), row=2, col=1)

    # RSI con zonas de sobrecompra / sobreventa.
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(15,29,51,0.035)", line_width=0, row=3, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(15,29,51,0.035)", line_width=0, row=3, col=1)
    for level in (30, 70):
        fig.add_hline(y=level, line=dict(color=LINE_STRONG, width=1, dash="dot"), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["RSI14"], name="RSI", line=dict(color=S1, width=1.6),
                             showlegend=False), row=3, col=1)
    fig.update_yaxes(range=[0, 100], tickvals=[30, 70], row=3, col=1)

    # MACD: histograma divergente (azul positivo / rojo negativo), MACD en tinta, señal en naranja.
    hist = df["MACDHist"].fillna(0)
    fig.add_trace(go.Bar(x=x, y=hist, marker_color=["rgba(31,111,214,0.5)" if v >= 0 else "rgba(200,54,45,0.45)"
                                                     for v in hist], name="Histograma", showlegend=False),
                  row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["MACD"], name="MACD", line=dict(color=INK, width=1.2), showlegend=False),
                  row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["MACDSignal"], name="Señal", line=dict(color=S2, width=1.2),
                             showlegend=False), row=4, col=1)

    style_figure(fig, height=780)
    fig.update_layout(bargap=0.15, margin=dict(l=8, r=90, t=30, b=8))
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_annotations(selector=dict(text="Volumen"), font=dict(family=FONT_SANS, size=11, color=MUTED))
    for name in ("RSI 14", "MACD 12 · 26 · 9"):
        fig.update_annotations(selector=dict(text=name), font=dict(family=FONT_SANS, size=11, color=MUTED))
    return fig


def relative_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    asset, bench = df.columns[0], df.columns[1]
    fig.add_hline(y=100, line=dict(color=LINE_STRONG, width=1))
    fig.add_trace(go.Scatter(x=df.index, y=df[bench], name=bench, line=dict(color=NEUTRAL_REF, width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df[asset], name=asset, line=dict(color=S1, width=2)))
    _end_labels(fig, df.index[-1], [(float(df[col].iloc[-1]), f"{col} {df[col].iloc[-1]:.1f}", color)
                                     for col, color in ((asset, S1), (bench, NEUTRAL_REF))],
                float(df.max().max() - df.min().min()), plot_px=270)
    style_figure(fig, height=320)
    fig.update_layout(margin=dict(l=8, r=110, t=30, b=8))
    return fig


def financials_chart(df: pd.DataFrame) -> go.Figure:
    years = [f"FY{d.year}" for d in df.index]
    fig = go.Figure()
    for col, name, color in (("revenue", "Ventas", S1), ("net_income", "Utilidad neta", S2),
                             ("fcf", "Free cash flow", S3)):
        if df[col].notna().any():
            fig.add_trace(go.Bar(x=years, y=df[col] / 1e6, name=name, marker_color=color,
                                 marker_line=dict(color=PANEL, width=2)))
    style_figure(fig, height=330)
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_xaxes(showspikes=False)
    return fig


def margins_chart(df: pd.DataFrame) -> go.Figure:
    years = [f"FY{d.year}" for d in df.index]
    fig = go.Figure()
    for col, name, color in (("gross_margin", "Margen bruto", S1), ("operating_margin", "Margen operativo", S2),
                             ("net_margin", "Margen neto", S3), ("fcf_margin", "Margen FCF", S4),
                             ("roic", "ROIC", S5)):
        if df[col].notna().any():
            fig.add_trace(go.Scatter(x=years, y=df[col], name=name, mode="lines+markers",
                                     line=dict(color=color, width=2),
                                     marker=dict(size=8, color=color, line=dict(color=PANEL, width=2))))
    style_figure(fig, height=330)
    fig.update_yaxes(ticksuffix="%")
    return fig


def quarterly_revenue_chart(df: pd.DataFrame) -> go.Figure:
    labels = [d.strftime("%Y-%m") for d in df.index]
    fig = go.Figure(go.Bar(x=labels, y=df["revenue"] / 1e6, name="Ventas", marker_color=S1))
    style_figure(fig, height=250, legend=False)
    fig.update_layout(bargap=0.35)
    fig.update_xaxes(showspikes=False, type="category")
    return fig


def quarterly_margin_chart(df: pd.DataFrame) -> go.Figure:
    labels = [d.strftime("%Y-%m") for d in df.index]
    fig = go.Figure(go.Scatter(x=labels, y=df["operating_margin"], name="Margen operativo", mode="lines+markers",
                               line=dict(color=S1, width=2),
                               marker=dict(size=8, color=S1, line=dict(color=PANEL, width=2))))
    style_figure(fig, height=250, legend=False)
    fig.update_xaxes(type="category")
    fig.update_yaxes(ticksuffix="%")
    return fig


def simulation_fan(sim, history: pd.Series, target: float | None = None,
                   support: float | None = None) -> go.Figure:
    """Abanico de percentiles de la simulación, precedido por el precio real reciente."""
    fig = go.Figure()
    x = sim.dates
    for path in sim.sample_paths[:25]:
        fig.add_trace(go.Scatter(x=x, y=path, line=dict(color="rgba(31,111,214,0.10)", width=0.8),
                                 hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=sim.bands[95], line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=sim.bands[5], line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(31,111,214,0.10)", name="Percentil 5 a 95",
                             hovertemplate="P5 %{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=sim.bands[75], line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=sim.bands[25], line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(31,111,214,0.22)", name="Percentil 25 a 75",
                             hovertemplate="P25 %{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=sim.bands[50], name="Mediana", line=dict(color=S1, width=2.2),
                             hovertemplate="Mediana %{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=history.index, y=history, name="Precio real", line=dict(color=INK, width=1.6),
                             hovertemplate="Cierre %{y:,.2f}<extra></extra>"))
    fig.add_vline(x=sim.start_date, line=dict(color=LINE_STRONG, width=1))
    labels = [(float(sim.bands[50][-1]), f"Mediana {sim.bands[50][-1]:,.2f}", S1),
              (float(sim.bands[95][-1]), f"P95 {sim.bands[95][-1]:,.2f}", NEUTRAL_REF),
              (float(sim.bands[5][-1]), f"P5 {sim.bands[5][-1]:,.2f}", NEUTRAL_REF)]
    for level, name, color in ((target, "Objetivo", S3), (support, "Soporte", S2)):
        if level:
            fig.add_hline(y=level, line=dict(color=color, width=1.4, dash="dot"))
            labels.append((float(level), f"{name} {level:,.2f}", color))
    # Eje fijo en el rango de los percentiles (y los niveles marcados): las trayectorias
    # de muestra extremas no deben aplastar el abanico.
    levels = [lvl for lvl in (target, support) if lvl]
    lo = min([float(sim.bands[5].min()), float(history.min())] + levels)
    hi = max([float(sim.bands[95].max()), float(history.max())] + levels)
    pad = (hi - lo) * 0.06
    _end_labels(fig, x[-1], labels, hi - lo + 2 * pad, plot_px=380)
    style_figure(fig, height=440)
    fig.update_layout(margin=dict(l=8, r=120, t=30, b=8), hovermode="x")
    fig.update_yaxes(range=[lo - pad, hi + pad])
    return fig


def simulation_histogram(sim, target: float | None = None, support: float | None = None) -> go.Figure:
    """Distribución del precio final; el eje se recorta a los percentiles 0.5 a 99.5."""
    final = sim.final_prices
    lo, hi = (float(v) for v in pd.Series(final).quantile([0.005, 0.995]))
    fig = go.Figure(go.Histogram(x=final[(final >= lo) & (final <= hi)], nbinsx=60, marker_color=S1,
                                 opacity=0.75, name="Escenarios",
                                 hovertemplate="%{x}<br>%{y} escenarios<extra></extra>"))
    for level, name, color in ((sim.start_price, "Hoy", INK), (target, "Objetivo", S3), (support, "Soporte", S2)):
        if level and lo <= level <= hi:
            fig.add_vline(x=level, line=dict(color=color, width=1.4, dash="dot" if name != "Hoy" else "solid"),
                          annotation=dict(text=f"{name} {level:,.2f}", font=dict(family=FONT_SANS, size=11, color=INK),
                                          bgcolor=PANEL, bordercolor=color, borderwidth=1),
                          annotation_position="top")
    style_figure(fig, height=300, legend=False)
    fig.update_layout(bargap=0.04, hovermode="closest", margin=dict(l=8, r=8, t=34, b=8))
    fig.update_xaxes(showspikes=False)
    return fig


# -- cartera ---------------------------------------------------------------------
def portfolio_lines(normalized: pd.DataFrame, value: pd.Series, bench: pd.Series | None, bench_name: str) -> go.Figure:
    """Base 100: cada activo en gris fino (con nombre en el hover), cartera y benchmark destacados."""
    fig = go.Figure()
    fig.add_hline(y=100, line=dict(color=LINE_STRONG, width=1))
    for col in normalized.columns:
        fig.add_trace(go.Scatter(x=normalized.index, y=normalized[col], name=col, showlegend=False,
                                 line=dict(color="rgba(106,120,140,0.45)", width=1),
                                 hovertemplate=f"{col} %{{y:.1f}}<extra></extra>"))
    labels = [(float(value.iloc[-1]), f"Cartera {value.iloc[-1]:.1f}", S1)]
    if bench is not None:
        fig.add_trace(go.Scatter(x=bench.index, y=bench, name=bench_name, line=dict(color=S2, width=1.8),
                                 hovertemplate=f"{bench_name} %{{y:.1f}}<extra></extra>"))
        labels.append((float(bench.dropna().iloc[-1]), f"{bench_name} {bench.dropna().iloc[-1]:.1f}", S2))
    fig.add_trace(go.Scatter(x=value.index, y=value, name="Cartera", line=dict(color=S1, width=2.6),
                             hovertemplate="Cartera %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[None], y=[None], name="Activos", line=dict(color="rgba(106,120,140,0.6)", width=1)))
    if len(normalized.columns) <= 6:  # con más activos las etiquetas se amontonan; queda el hover
        for col in normalized.columns:
            labels.append((float(normalized[col].iloc[-1]), col, NEUTRAL_REF))
    span = float(max(normalized.max().max(), value.max()) - min(normalized.min().min(), value.min()))
    _end_labels(fig, value.index[-1], labels, span, plot_px=330)
    style_figure(fig, height=400)
    fig.update_layout(margin=dict(l=8, r=110, t=30, b=8), hovermode="x unified")
    return fig


def portfolio_drawdown(dd: pd.Series) -> go.Figure:
    fig = go.Figure(go.Scatter(x=dd.index, y=dd * 100, fill="tozeroy", name="Drawdown",
                               line=dict(color=S1, width=1.4), fillcolor="rgba(31,111,214,0.15)",
                               hovertemplate="%{y:.1f}%<extra></extra>"))
    style_figure(fig, height=240, legend=False)
    fig.update_yaxes(ticksuffix="%")
    return fig


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    """Matriz divergente: azul = correlación negativa, blanco = 0, naranja = positiva."""
    n = len(corr)
    text = [[f"{v:.2f}" for v in row] for row in corr.to_numpy()]
    fig = go.Figure(go.Heatmap(
        z=corr.to_numpy(), x=list(corr.columns), y=list(corr.index), zmin=-1, zmax=1,
        colorscale=[[0, S1], [0.5, "#ffffff"], [1, S2]], text=text, texttemplate="%{text}",
        textfont=dict(family=FONT_SANS, size=11 if n <= 8 else 9), xgap=2, ygap=2,
        colorbar=dict(thickness=10, len=0.8, tickfont=dict(family=FONT_SANS, size=10, color=MUTED)),
        hovertemplate="%{y} · %{x}: %{z:.2f}<extra></extra>"))
    style_figure(fig, height=max(300, 46 * n + 80), legend=False)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(showspikes=False, side="bottom")
    fig.update_yaxes(showgrid=False, autorange="reversed", side="left")
    return fig


def risk_vs_weight(table: list[dict]) -> go.Figure:
    """Peso de cada activo contra su contribución al riesgo de la cartera."""
    rows = sorted(table, key=lambda r: r["risk_contribution_pct"] or 0)
    names = [r["ticker"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=names, x=[r["weight_pct"] for r in rows], name="Peso", orientation="h",
                         marker_color="rgba(31,111,214,0.35)", hovertemplate="Peso %{x:.1f}%<extra></extra>"))
    fig.add_trace(go.Bar(y=names, x=[r["risk_contribution_pct"] for r in rows], name="Contribución al riesgo",
                         orientation="h", marker_color=S2, hovertemplate="Riesgo %{x:.1f}%<extra></extra>"))
    style_figure(fig, height=max(260, 34 * len(rows) + 90))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05, hovermode="y unified")
    fig.update_xaxes(ticksuffix="%", showgrid=True, gridcolor=LINE, showspikes=False)
    fig.update_yaxes(showgrid=False, side="left")
    return fig


def sector_bars(sectors: dict[str, float]) -> go.Figure:
    items = sorted(sectors.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(y=[k for k, _ in items], x=[v for _, v in items], orientation="h", marker_color=S1,
                           text=[f"{v:.1f}%" for _, v in items], textposition="outside",
                           textfont=dict(family=FONT_SANS, size=11, color=INK),
                           hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    style_figure(fig, height=max(200, 34 * len(items) + 60), legend=False)
    fig.update_layout(hovermode="closest", margin=dict(l=8, r=40, t=10, b=8))
    fig.update_xaxes(ticksuffix="%", showgrid=True, gridcolor=LINE, showspikes=False,
                     range=[0, max(v for _, v in items) * 1.2])
    fig.update_yaxes(showgrid=False, side="left")
    return fig
