"""
portfolio.py
============
Análisis de una cartera (o lista de seguimiento) de varios tickers.

Con los pesos que indica el usuario (por defecto 1/n) calcula:
  * por activo: retorno, volatilidad, drawdown, Sharpe, beta, tendencia, RSI,
    P/E, upside al consenso y su contribución al riesgo de la cartera;
  * de la cartera: retorno, volatilidad, drawdown, Sharpe, beta, VaR / CVaR
    histórico diario, ratio de diversificación, número efectivo de activos y
    de apuestas, correlación promedio y peso del primer factor común;
  * correlaciones entre activos y concentración por sector.

Convención: la cartera mantiene los pesos constantes (rebalanceo diario). Así
el retorno, la volatilidad y la descomposición del riesgo usan los mismos
pesos que se muestran. No es un optimizador: no sugiere pesos.

El módulo no importa Streamlit.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from agent import (TRADING_DAYS, TickerNotFoundError, add_indicators, benchmark_for, fetch_history, fetch_info,
                   fetch_risk_free, normalize_ticker)

WINDOWS = {"6M": 183, "1Y": 366, "3Y": 1096, "5Y": 1827}
MAX_TICKERS = 15
MIN_OBS = 40


@dataclass
class PortfolioResult:
    window: str
    benchmark: str
    weights: pd.Series                       # ticker -> peso (suma 1)
    normalized: pd.DataFrame                 # base 100 por activo
    value: pd.Series                         # cartera base 100
    bench_value: pd.Series | None            # benchmark base 100
    drawdown: pd.Series                      # drawdown de la cartera
    corr: pd.DataFrame
    table: list[dict[str, Any]]
    summary: dict[str, Any]
    sectors: dict[str, float]
    facts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pesos
# ---------------------------------------------------------------------------

def resolve_weights(rows: list[tuple[str, float | None]]) -> tuple[list[str], list[float], list[str]]:
    """Normaliza la tabla de tickers y pesos que carga el usuario.

    * Tickers repetidos se suman; vacíos se ignoran.
    * Las filas sin peso se reparten en partes iguales lo que falta para 100%.
    * Si los pesos no suman 100, se reescalan y se avisa.
    Devuelve (tickers, pesos en %, avisos).
    """
    notes: list[str] = []
    merged: dict[str, float | None] = {}
    for raw, w in rows:
        raw = "" if raw is None or (isinstance(raw, float) and math.isnan(raw)) else str(raw).strip()
        if not raw:
            continue
        try:
            tk = normalize_ticker(raw)
        except TickerNotFoundError:
            notes.append(f"'{raw}' no es un ticker válido: se ignoró.")
            continue
        w = None if w is None or (isinstance(w, float) and math.isnan(w)) else float(w)
        if w is not None and w < 0:
            notes.append(f"{tk}: peso negativo ignorado (la cartera no admite posiciones cortas).")
            w = None
        if tk in merged:
            notes.append(f"{tk} estaba repetido: se sumaron sus pesos.")
            prev = merged[tk]
            merged[tk] = None if prev is None and w is None else (prev or 0) + (w or 0)
        else:
            merged[tk] = w
    if len(merged) > MAX_TICKERS:
        notes.append(f"Se usan los primeros {MAX_TICKERS} tickers.")
        merged = dict(list(merged.items())[:MAX_TICKERS])
    explicit = sum(w for w in merged.values() if w is not None)
    blanks = [t for t, w in merged.items() if w is None]
    if blanks:
        rest = max(100.0 - explicit, 0.0)
        if rest == 0 and explicit > 0:
            notes.append("Los pesos cargados ya suman 100% o más: las filas sin peso quedan en 0%.")
        share = rest / len(blanks) if explicit > 0 else 100.0 / len(blanks)
        for t in blanks:
            merged[t] = share
    total = sum(merged.values())
    if total <= 0:
        return [], [], notes + ["Todos los pesos son cero."]
    if abs(total - 100) > 0.05:
        notes.append(f"Los pesos sumaban {total:.1f}%: se reescalaron para que sumen 100%.")
    zero = [t for t, w in merged.items() if w <= 0]
    if zero:
        notes.append(f"Sin peso (excluidos de la cartera): {', '.join(zero)}.")
    kept = [(t, w / total * 100) for t, w in merged.items() if w > 0]
    return [t for t, _ in kept], [w for _, w in kept], notes


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

def load_assets(tickers: list[str]) -> tuple[dict[str, pd.DataFrame], dict[str, dict], dict[str, str]]:
    """Historia de 5 años e info de cada ticker, en paralelo. Devuelve (historias, infos, errores)."""
    def one(tk: str):
        try:
            return tk, fetch_history(tk, "5Y"), fetch_info(tk), None
        except Exception as exc:
            return tk, None, None, f"{type(exc).__name__}: {exc}"

    hist, info, errors = {}, {}, {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for tk, h, i, err in pool.map(one, tickers):
            if err or h is None or h.empty:
                errors[tk] = err or "sin datos"
            else:
                hist[tk], info[tk] = h, i or {}
    return hist, info, errors


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def _max_dd(series: pd.Series) -> float:
    return float((series / series.cummax() - 1).min())


def _trend(full: pd.DataFrame) -> tuple[str | None, float | None]:
    """Tendencia de medias (mismo criterio que el agente) y RSI 14 de la última rueda."""
    last = add_indicators(full).iloc[-1]
    price, s50, s200, rsi = (last.get(c) for c in ("Close", "SMA50", "SMA200", "RSI14"))
    trend = None
    if pd.notna(s50) and pd.notna(s200):
        trend = "alcista" if price > s50 > s200 else "bajista" if price < s50 < s200 else "mixta"
    return trend, (None if pd.isna(rsi) else round(float(rsi), 1))


def _pct(x: Any, digits: int = 1) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x) * 100, digits)


def _r(x: Any, digits: int = 2) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), digits)


def _corr_word(c: float) -> str:
    a = abs(c)
    return "muy alta" if a >= 0.8 else "alta" if a >= 0.6 else "moderada" if a >= 0.3 else "baja"


def analyze_portfolio(tickers: list[str], weights_pct: list[float], window: str = "1Y") -> PortfolioResult:
    if window not in WINDOWS:
        raise ValueError(f"Ventana inválida: {window}")
    if len(tickers) < 2:
        raise ValueError("Cargá al menos dos tickers para analizar una cartera.")
    hist, info, errors = load_assets(tickers)
    notes = [f"{tk}: no se pudo descargar ({err}). Quedó fuera de la cartera." for tk, err in errors.items()]
    ok = [t for t in tickers if t in hist]
    if len(ok) < 2:
        raise ValueError("Hacen falta al menos dos tickers con datos. " + " ".join(notes))
    w = pd.Series({t: weights_pct[tickers.index(t)] for t in ok})
    if len(ok) < len(tickers):
        notes.append("Los pesos de los tickers restantes se reescalaron para sumar 100%.")
    w = w / w.sum()

    start = pd.Timestamp.now().normalize() - pd.Timedelta(days=WINDOWS[window])
    closes = pd.concat({t: hist[t]["Close"] for t in ok}, axis=1)
    # Mercados con feriados distintos: se arrastra el último cierre hasta 5 ruedas.
    closes = closes.ffill(limit=5).loc[lambda d: d.index >= start].dropna()
    if len(closes) < MIN_OBS:
        raise ValueError(f"Muy pocas ruedas en común entre los tickers ({len(closes)}). "
                         "Probá una ventana más larga o revisá los tickers.")
    first_dates = {t: hist[t].index[0] for t in ok}
    late = [t for t, d in first_dates.items() if d > start + pd.Timedelta(days=10)]
    if late:
        notes.append(f"{', '.join(late)} cotiza desde hace menos que la ventana: la cartera arranca el "
                     f"{closes.index[0]:%Y-%m-%d}.")
    currencies = {t: (info[t].get("currency") or "N/D") for t in ok}
    if len(set(currencies.values())) > 1:
        notes.append("La cartera mezcla monedas (" + ", ".join(f"{t} {c}" for t, c in currencies.items()) +
                     "). Los retornos se miden en la moneda de cada activo: las correlaciones y el retorno "
                     "conjunto quedan distorsionados por el tipo de cambio.")

    rets = closes.pct_change().dropna()
    port_ret = rets @ w
    value = (1 + port_ret).cumprod() * 100
    value = pd.concat([pd.Series([100.0], index=[closes.index[0]]), value])
    normalized = closes / closes.iloc[0] * 100

    # Benchmark (SPY, o Merval si todos los activos cotizan en pesos).
    all_ars = all(c == "ARS" for c in currencies.values())
    bench_tk = benchmark_for(ok[0] if all_ars else "SPY", "ARS" if all_ars else "USD")
    bench_value, bench_ret = None, None
    try:
        b = fetch_history(bench_tk, "5Y")["Close"].reindex(closes.index).ffill(limit=5)
        if b.notna().sum() > MIN_OBS:
            bench_value = b / b.dropna().iloc[0] * 100
            bench_ret = b.pct_change().reindex(rets.index)
    except Exception:
        notes.append(f"No se pudo descargar el benchmark {bench_tk}.")

    rf = (fetch_risk_free() or 0) / 100
    ann = TRADING_DAYS
    cov = rets.cov() * ann
    vol_i = np.sqrt(np.diag(cov))
    var_p = float(w @ cov @ w)
    vol_p = math.sqrt(var_p)
    rc = (w * (cov @ w)) / var_p                       # contribución al riesgo (suma 1)
    corr = rets.corr()

    def beta(r: pd.Series) -> float | None:
        if bench_ret is None:
            return None
        both = pd.concat([r, bench_ret], axis=1).dropna()
        return float(both.cov().iloc[0, 1] / both.iloc[:, 1].var()) if len(both) > MIN_OBS else None

    def sharpe(r: pd.Series) -> float | None:
        lr = np.log1p(r)
        sd = lr.std() * math.sqrt(ann)
        return (lr.mean() * ann - rf) / sd if sd else None

    table = []
    for i, t in enumerate(ok):
        inf = info[t]
        price = float(closes[t].iloc[-1])
        target = inf.get("targetMeanPrice")
        trend, rsi = _trend(hist[t])
        sector = inf.get("sector") or ("ETF" if inf.get("quoteType") == "ETF" else inf.get("quoteType") or "Otro")
        table.append({
            "ticker": t,
            "name": inf.get("shortName") or inf.get("longName") or t,
            "sector": sector,
            "currency": currencies[t],
            "weight_pct": _pct(w[t]),
            "return_pct": _pct(closes[t].iloc[-1] / closes[t].iloc[0] - 1),
            "volatility_pct": _pct(vol_i[i]),
            "max_drawdown_pct": _pct(_max_dd(closes[t])),
            "sharpe": _r(sharpe(rets[t])),
            "beta": _r(beta(rets[t])),
            "risk_contribution_pct": _pct(rc[t]),
            "corr_with_portfolio": _r(rets[t].corr(port_ret)),
            "trend": trend,
            "rsi14": rsi,
            "pe_forward": _r(inf.get("forwardPE"), 1) if (inf.get("forwardPE") or 0) > 0 else None,
            "upside_pct": _pct(target / price - 1) if target and price else None,
        })

    var95 = float(-np.percentile(port_ret, 5))
    cvar95 = float(-port_ret[port_ret <= -var95].mean())
    off = corr.where(~np.eye(len(ok), dtype=bool)).stack()
    eig = np.linalg.eigvalsh(corr.to_numpy())
    rc_share = rc.to_numpy()
    sectors = (pd.Series({r["ticker"]: r["sector"] for r in table}).to_frame("s").assign(w=w)
               .groupby("s")["w"].sum().sort_values(ascending=False))
    summary = {
        "window": window,
        "start": closes.index[0].strftime("%Y-%m-%d"),
        "end": closes.index[-1].strftime("%Y-%m-%d"),
        "observations": len(rets),
        "n_assets": len(ok),
        "return_pct": _pct(value.iloc[-1] / 100 - 1),
        "volatility_pct": _pct(vol_p),
        "weighted_avg_volatility_pct": _pct(float(w @ vol_i)),
        "max_drawdown_pct": _pct(_max_dd(value)),
        "current_drawdown_pct": _pct(value.iloc[-1] / value.max() - 1),
        "sharpe": _r(sharpe(port_ret)),
        "beta": _r(beta(port_ret)),
        "benchmark": bench_tk,
        "benchmark_return_pct": _pct(bench_value.dropna().iloc[-1] / 100 - 1) if bench_value is not None else None,
        "var95_1d_pct": _pct(var95, 2),
        "cvar95_1d_pct": _pct(cvar95, 2),
        "diversification_ratio": _r(float(w @ vol_i) / vol_p),
        "effective_assets": _r(1 / float((w ** 2).sum()), 1),
        "effective_bets": _r(1 / float((rc_share ** 2).sum()), 1),
        "avg_correlation": _r(float(off.mean())),
        "first_factor_share_pct": _pct(float(eig.max() / eig.sum())),
        "max_sector": " y ".join(sectors[sectors >= sectors.iloc[0] - 1e-9].index),
        "max_sector_weight_pct": _pct(float(sectors.iloc[0])),
        "risk_free_pct": _r(rf * 100),
    }
    if summary["benchmark_return_pct"] is not None:
        summary["vs_benchmark_pp"] = _r(summary["return_pct"] - summary["benchmark_return_pct"], 1)

    pairs = off[[a < b for a, b in off.index]].sort_values()
    result = PortfolioResult(window, bench_tk, w, normalized, value, bench_value, value / value.cummax() - 1,
                             corr, table, summary, {k: _pct(v) for k, v in sectors.items()}, notes=notes)
    result.facts = portfolio_facts(result, pairs)
    return result


def portfolio_facts(r: PortfolioResult, pairs: pd.Series) -> list[str]:
    s = r.summary
    facts = [
        f"Cartera de {s['n_assets']} activos, ventana {s['window']} ({s['start']} a {s['end']}): retorno {s['return_pct']:+}%, "
        f"volatilidad anual {s['volatility_pct']}%, máximo drawdown {s['max_drawdown_pct']}%, Sharpe {s['sharpe']}."
    ]
    if s.get("benchmark_return_pct") is not None:
        verb = "le ganó" if s["vs_benchmark_pp"] > 0 else "quedó detrás de"
        facts.append(f"La cartera {verb} {s['benchmark']} por {abs(s['vs_benchmark_pp'])} pp "
                     f"({s['benchmark']} {s['benchmark_return_pct']:+}%); beta {s['beta']}.")
    dr = s["diversification_ratio"]
    facts.append(
        f"Diversificación: la volatilidad de la cartera ({s['volatility_pct']}%) es menor que el promedio ponderado de "
        f"las volatilidades individuales ({s['weighted_avg_volatility_pct']}%); ratio de diversificación {dr} "
        f"({'bajo' if dr < 1.15 else 'moderado' if dr < 1.4 else 'alto'}: 1.0 sería no diversificar nada).")
    facts.append(
        f"Correlación promedio entre activos {s['avg_correlation']} ({_corr_word(s['avg_correlation'])}); un factor común "
        f"explica el {s['first_factor_share_pct']}% de la variación conjunta.")
    facts.append(
        f"Número efectivo de activos por peso: {s['effective_assets']} de {s['n_assets']}; número efectivo de apuestas por "
        f"riesgo: {s['effective_bets']}. Si el segundo es mucho menor, pocos activos concentran el riesgo.")
    gaps = sorted(r.table, key=lambda x: (x["risk_contribution_pct"] or 0) - (x["weight_pct"] or 0), reverse=True)
    top = gaps[0]
    if top["risk_contribution_pct"] - top["weight_pct"] > 2:
        facts.append(f"{top['ticker']} pesa {top['weight_pct']}% pero aporta el {top['risk_contribution_pct']}% del riesgo "
                     f"(volatilidad {top['volatility_pct']}%, correlación con la cartera {top['corr_with_portfolio']}).")
    low = gaps[-1]
    if low["weight_pct"] - low["risk_contribution_pct"] > 2:
        facts.append(f"{low['ticker']} es el mayor diversificador: pesa {low['weight_pct']}% y aporta solo el "
                     f"{low['risk_contribution_pct']}% del riesgo.")
    if len(pairs):
        (a1, b1), c1 = pairs.index[-1], pairs.iloc[-1]
        (a2, b2), c2 = pairs.index[0], pairs.iloc[0]
        facts.append(f"Par más correlacionado: {a1} y {b1} ({c1:.2f}, {_corr_word(c1)}). "
                     f"Par menos correlacionado: {a2} y {b2} ({c2:.2f}, {_corr_word(c2)}).")
    facts.append(f"Sector con más peso: {s['max_sector']} ({s['max_sector_weight_pct']}%).")
    facts.append(f"VaR histórico 95% a 1 día: {s['var95_1d_pct']}% (1 de cada 20 ruedas se pierde al menos eso); "
                 f"CVaR 95%: {s['cvar95_1d_pct']}%.")
    return facts
