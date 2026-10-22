"""
simulation.py
=============
Simulación Monte Carlo del precio de un activo, para medir qué tan exigente es
el precio objetivo de los analistas frente a la volatilidad real de la acción.

Dos métodos:
  * "bootstrap": remuestrea bloques de retornos diarios históricos (bloques de
    10 ruedas). Conserva colas gordas y parte de la agrupación de volatilidad,
    sin suponer normalidad.
  * "gbm": movimiento browniano geométrico (retornos log normales), el modelo
    de libro con la media y el desvío históricos.

Dos supuestos de deriva (tendencia):
  * "neutral": se quita la tendencia histórica y el precio crece en promedio a
    la tasa libre de riesgo. Es el supuesto prudente: que la acción haya subido
    100% el último año no significa que lo vuelva a hacer.
  * "historica": se usa la tendencia de la muestra tal cual.

El módulo no importa Streamlit: se puede usar desde un notebook o un test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS = 252
HORIZONS = {"1M": 21, "3M": 63, "6M": 126, "12M": 252}
SAMPLES = {"1Y": 252, "3Y": 756, "5Y": 1260}
METHODS = {"bootstrap": "Bootstrap histórico por bloques", "gbm": "Lognormal (GBM)"}
DRIFTS = {"neutral": "Neutral al riesgo", "historica": "Tendencia histórica"}
PERCENTILES = (5, 25, 50, 75, 95)
BLOCK = 10
MIN_OBS = 120


@dataclass
class SimulationResult:
    ticker: str
    horizon: str
    method: str
    drift: str
    n_paths: int
    start_price: float
    start_date: pd.Timestamp
    dates: pd.DatetimeIndex                  # fechas hábiles futuras (incluye el punto de partida)
    bands: dict[int, np.ndarray]             # percentil -> trayectoria
    sample_paths: np.ndarray                 # algunas trayectorias para graficar
    final_prices: np.ndarray
    summary: dict[str, Any] = field(default_factory=dict)
    facts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _pct(x: float | None, digits: int = 1) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x) * 100, digits)


def _paths(log_ret: np.ndarray, horizon: int, n_paths: int, method: str, mu_daily: float,
           rng: np.random.Generator) -> np.ndarray:
    """Retornos log acumulados (n_paths x horizon) con la deriva diaria indicada."""
    sigma = log_ret.std(ddof=1)
    if method == "gbm":
        steps = rng.normal(mu_daily, sigma, size=(n_paths, horizon))
    else:
        centered = log_ret - log_ret.mean() + mu_daily
        n_blocks = math.ceil(horizon / BLOCK)
        starts = rng.integers(0, len(centered) - BLOCK + 1, size=(n_paths, n_blocks))
        idx = (starts[:, :, None] + np.arange(BLOCK)).reshape(n_paths, -1)[:, :horizon]
        steps = centered[idx]
    return np.cumsum(steps, axis=1)


def simulate_price(close: pd.Series, ticker: str = "", horizon: str = "12M", method: str = "bootstrap",
                   drift: str = "neutral", sample: str = "3Y", risk_free_pct: float | None = None,
                   target: float | None = None, support: dict[str, Any] | None = None,
                   currency: str | None = None, n_paths: int = 10_000, seed: int = 7) -> SimulationResult:
    """Simula n_paths trayectorias de precio para el horizonte pedido.

    target:  precio objetivo medio del consenso (opcional).
    support: soporte técnico más cercano {"level", "source"} (opcional).
    La semilla es fija para que los resultados no cambien al recargar la página.
    """
    if horizon not in HORIZONS or method not in METHODS or drift not in DRIFTS or sample not in SAMPLES:
        raise ValueError("Parámetros de simulación inválidos")
    close = close.dropna()
    log_ret = np.log(close).diff().dropna().to_numpy()[-SAMPLES[sample]:]
    if len(log_ret) < MIN_OBS:
        raise ValueError(f"Hacen falta al menos {MIN_OBS} ruedas de historia para simular "
                         f"(hay {len(log_ret)}).")
    notes: list[str] = []
    if len(log_ret) < SAMPLES[sample] * 0.9:
        notes.append(f"El activo tiene menos historia que la muestra pedida: se usaron {len(log_ret)} ruedas.")

    h = HORIZONS[horizon]
    s0 = float(close.iloc[-1])
    sigma_d = float(log_ret.std(ddof=1))
    mu_hist_d = float(log_ret.mean())
    rf = (risk_free_pct or 0.0) / 100
    if drift == "neutral":
        # Precio esperado creciendo a la tasa libre de riesgo: E[S_T] = S0 * (1 + rf)^T.
        mu_d = math.log(1 + rf) / TRADING_DAYS - 0.5 * sigma_d ** 2
        if currency and currency != "USD":
            notes.append(f"La deriva neutral usa la tasa del Tesoro de EE.UU.; para un activo en {currency} "
                         "no refleja la inflación ni las tasas locales. Considerá la tendencia histórica.")
    else:
        mu_d = mu_hist_d

    rng = np.random.default_rng(seed)
    cum = _paths(log_ret, h, n_paths, method, mu_d, rng)
    prices = s0 * np.exp(np.hstack([np.zeros((n_paths, 1)), cum]))
    final = prices[:, -1]
    rets = final / s0 - 1

    bands = {q: np.percentile(prices, q, axis=0) for q in PERCENTILES}
    dates = pd.bdate_range(close.index[-1], periods=h + 1)
    p5 = np.percentile(rets, 5)
    summary: dict[str, Any] = {
        "ticker": ticker,
        "horizon": horizon,
        "horizon_trading_days": h,
        "method": METHODS[method],
        "drift": DRIFTS[drift],
        "sample": sample,
        "sample_trading_days": len(log_ret),
        "paths": n_paths,
        "start_price": round(s0, 2),
        "start_date": close.index[-1].strftime("%Y-%m-%d"),
        "volatility_annual_pct": _pct(sigma_d * math.sqrt(TRADING_DAYS)),
        "historical_return_annual_pct": _pct(math.exp(mu_hist_d * TRADING_DAYS) - 1),
        "risk_free_pct": round(risk_free_pct, 2) if drift == "neutral" and risk_free_pct is not None else None,
        "final_price_percentiles": {f"p{q}": round(float(np.percentile(final, q)), 2) for q in PERCENTILES},
        "median_return_pct": _pct(np.median(rets)),
        "prob_gain_pct": _pct((final > s0).mean()),
        "prob_loss_over_20_pct": _pct((rets <= -0.20).mean()),
        "var95_pct": _pct(-p5),
        "cvar95_pct": _pct(-rets[rets <= p5].mean()),
    }

    if target:
        up = target >= s0
        touched = (prices.max(axis=1) >= target) if up else (prices.min(axis=1) <= target)
        ended = (final >= target) if up else (final <= target)
        sd_move = math.log(target / s0) / (sigma_d * math.sqrt(h))
        summary["target"] = {
            "price": round(float(target), 2),
            "distance_pct": _pct(target / s0 - 1),
            "prob_touch_pct": _pct(touched.mean()),
            "prob_end_beyond_pct": _pct(ended.mean()),
            "required_move_sd": round(sd_move, 2),
        }
    if support and support.get("level") and support["level"] < s0:
        lvl = float(support["level"])
        summary["support"] = {
            "price": round(lvl, 2),
            "source": support.get("source"),
            "distance_pct": _pct(lvl / s0 - 1),
            "prob_touch_pct": _pct((prices.min(axis=1) <= lvl).mean()),
            "prob_end_below_pct": _pct((final < lvl).mean()),
        }

    take = rng.choice(n_paths, size=min(40, n_paths), replace=False)
    result = SimulationResult(ticker, horizon, method, drift, n_paths, s0, close.index[-1], dates, bands,
                              prices[take], final, summary, notes=notes)
    result.facts = simulation_facts(summary)
    return result


def simulation_facts(s: dict[str, Any]) -> list[str]:
    """Lecturas en texto, con las mismas cifras que se muestran en pantalla."""
    pc = s["final_price_percentiles"]
    facts = [
        f"Supuestos: {s['method'].lower()}, {s['drift'].lower()}, volatilidad anual {s['volatility_annual_pct']}% "
        f"medida sobre {s['sample_trading_days']} ruedas; {s['paths']:_} trayectorias a {s['horizon']}.".replace("_", "."),
        f"Rango probable a {s['horizon']}: entre {pc['p5']} y {pc['p95']} en el 90% de los escenarios "
        f"(mediana {pc['p50']}, {s['median_return_pct']:+}% vs {s['start_price']}).",
        f"Probabilidad de terminar con ganancia: {s['prob_gain_pct']}%. Pérdida mayor al 20%: "
        f"{s['prob_loss_over_20_pct']}%.",
        f"VaR 95% a {s['horizon']}: {s['var95_pct']}% (en el 5% de los peores escenarios se pierde al menos eso); "
        f"CVaR 95%: {s['cvar95_pct']}% (pérdida promedio en esos escenarios).",
    ]
    t = s.get("target")
    if t:
        verb = "suba" if t["distance_pct"] >= 0 else "baje"
        facts.append(
            f"Precio objetivo medio {t['price']} ({t['distance_pct']:+}%): el precio lo toca en algún momento en el "
            f"{t['prob_touch_pct']}% de los escenarios y termina más allá en el {t['prob_end_beyond_pct']}%.")
        size = abs(t["required_move_sd"])
        word = "chico" if size < 0.5 else "moderado" if size < 1 else "exigente" if size < 1.5 else "muy exigente"
        facts.append(f"Llegar al objetivo requiere que el precio {verb} {size} desvíos estándar del horizonte: "
                     f"un movimiento {word} para la volatilidad de la acción.")
    sp = s.get("support")
    if sp:
        facts.append(f"Soporte {sp['price']} ({sp['source']}, {sp['distance_pct']}%): se perfora en algún momento en el "
                     f"{sp['prob_touch_pct']}% de los escenarios y se termina por debajo en el {sp['prob_end_below_pct']}%.")
    if s.get("risk_free_pct") is not None:
        facts.append(f"La deriva neutral supone que el precio crece en promedio a la tasa libre de riesgo "
                     f"({s['risk_free_pct']}%), no a la tendencia histórica ({s['historical_return_annual_pct']}% anual).")
    return facts
