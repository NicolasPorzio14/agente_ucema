"""
agent.py
========
Lógica del Agente de Análisis Financiero.

Contiene:
  * Un registro de herramientas (tools) con esquema JSON, para que el agente
    las invoque dinámicamente.
  * Cinco herramientas sobre yfinance y Google News:
      - get_market_data: perfil, cotización, múltiplos, rentabilidad TTM,
        balance, consenso de analistas, tenencia y próximo balance.
      - get_financial_statements: estados contables de 4/5 años y últimos
        trimestres, con márgenes, ROIC, FCF, apalancamiento y CAGR.
      - calculate_technical_indicators: medias y cruces, RSI, MACD, Bollinger,
        ATR, retornos multi horizonte, riesgo y fuerza relativa vs benchmark.
      - compare_peers: múltiplos y rentabilidad de comparables con rankings.
      - get_company_news_sentiment: titulares de Yahoo Finance y Google News
        (español e inglés) con un sentimiento léxico rápido.
    Cada herramienta devuelve además "facts": lecturas ya calculadas (rankings,
    posición vs mediana, posición técnica) para que el LLM no tenga que
    comparar números por su cuenta, que es donde más se equivoca.
  * FinancialAgent, con dos motores intercambiables:
      1. "crewai": un Crew de CrewAI con un agente analista que usa OpenAI
         como LLM. El agente llama las herramientas, elige comparables si el
         usuario no los indica y redacta el informe. Después un paso de
         verificación coteja el borrador contra los datos y lo corrige.
      2. "reglas": un planificador determinístico que llama las herramientas
         en orden y sintetiza el informe con reglas cuantitativas.
         Funciona sin API key y no consume créditos.
    Si el motor "crewai" falla (key inválida, sin créditos, red, rate limit),
    el agente cae automáticamente al motor de reglas, así la app no se rompe.

Convenciones de datos (importantes para no mezclar fuentes):
  * TTM = suma de los últimos 4 trimestres reportados. Márgenes, EBITDA, FCF,
    P/S, EV/EBITDA y FCF yield salen de ahí, con el mismo método para la
    empresa y para sus comparables. Yahoo "info" solo es respaldo.
  * FY = año fiscal (estados anuales). MRQ = último balance trimestral.
  * Si una empresa cotiza en una moneda y reporta en otra (YPF, GGAL, PBR),
    se omiten los múltiplos que mezclarían monedas.

El módulo no importa Streamlit: se puede testear o reutilizar desde un
notebook, un script o una API.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# CrewAI envía telemetría por defecto; la desactivamos para no agregar latencia.
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")


def _configure_extra_ca() -> None:
    """Confía en el certificado raíz de antivirus / proxies corporativos.

    Antivirus como Avast o Kaspersky interceptan HTTPS con su propio
    certificado. curl_cffi (usado por yfinance) y requests solo confían en
    certifi, así que fallan con "unable to get local issuer certificate".
    Si EXTRA_CA_CERTS (o NODE_EXTRA_CA_CERTS, que Avast define solo) apunta
    a un .pem, armamos un bundle certifi + ese certificado.
    """
    extra = os.getenv("EXTRA_CA_CERTS") or os.getenv("NODE_EXTRA_CA_CERTS")
    if not extra or not os.path.isfile(extra) or os.getenv("CURL_CA_BUNDLE"):
        return
    try:
        import certifi
        bundle = os.path.join(tempfile.gettempdir(), "financial_agent_ca_bundle.pem")
        with open(bundle, "wb") as out:
            for src in (certifi.where(), extra):
                with open(src, "rb") as f:
                    out.write(f.read().rstrip() + b"\n")
        for var in ("CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"):
            os.environ.setdefault(var, bundle)
    except Exception:
        pass  # si falla, seguimos con certifi tal cual


_configure_extra_ca()

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------

# Días calendario de cada temporalidad que se muestra en la UI.
PERIOD_DAYS: dict[str, int] = {"1M": 31, "6M": 183, "1Y": 366, "5Y": 1827}

# Historia extra que se descarga para que la SMA 200 exista desde el primer
# día visible (200 ruedas son unos 290 días calendario; dejamos margen).
WARMUP_DAYS = 320

TRADING_DAYS = 252
# Modelo de OpenAI por defecto (se puede cambiar desde la UI o con OPENAI_MODEL).
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
MAX_AGENT_ITERATIONS = 14
MAX_PEERS = 6
NEWS_LOOKBACK_DAYS = 30
MAX_HEADLINES = 15
MIN_OBS_RISK_RATIOS = 60  # con menos ruedas, Sharpe y Sortino son ruido
CACHE_TTL_SECONDS = 300  # evita castigar a Yahoo con requests repetidos
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) financial-agent/2.0"}
REVIEW_MARKER = "<<<CORRECCIONES>>>"


class TickerNotFoundError(Exception):
    """El ticker no existe o Yahoo no devuelve datos para él."""


class DataSourceError(Exception):
    """Yahoo Finance falló (red, rate limit, cambios de API)."""


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _clean_number(value: Any) -> float | None:
    """Convierte a float y descarta NaN, inf y valores no numéricos."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def _r(value: Any, digits: int = 2) -> float | None:
    v = _clean_number(value)
    return None if v is None else round(v, digits)


def _pct(value: Any, digits: int = 1) -> float | None:
    """Fracción (0.123) a porcentaje redondeado (12.3)."""
    v = _clean_number(value)
    return None if v is None else round(v * 100, digits)


def _div(a: Any, b: Any) -> float | None:
    a, b = _clean_number(a), _clean_number(b)
    if a is None or b is None or b == 0:
        return None
    return a / b


def _positive(value: Any) -> float | None:
    """Múltiplos con denominador negativo (pérdidas) no son comparables."""
    v = _clean_number(value)
    return v if v is not None and v > 0 else None


def _cagr(first: Any, last: Any, years: float) -> float | None:
    """CAGR en %. Solo tiene sentido con ambos extremos positivos."""
    first, last = _clean_number(first), _clean_number(last)
    if first is None or last is None or first <= 0 or last <= 0 or years <= 0:
        return None
    return round(((last / first) ** (1 / years) - 1) * 100, 1)


def _g(v: float) -> str:
    """Formato compacto para textos de facts: 22.31 -> '22.31', 5.0 -> '5'.

    Sin separador de miles: en un informe en español "5,670" se lee como 5.67.
    """
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _money(amount: float | None, currency: str = "") -> str:
    """Monto absoluto en texto sin ambigüedad: 329134200000 -> '329.1 mil millones ARS'."""
    if amount is None:
        return "N/D"
    a = abs(amount)
    sign = "-" if amount < 0 else ""
    if a >= 1e9:
        txt = f"{sign}{a / 1e9:.1f} mil millones"
    elif a >= 1e6:
        txt = f"{sign}{a / 1e6:.1f} millones"
    else:
        txt = f"{sign}{a:.0f}"
    return f"{txt} {currency}".strip()


def _retry(func: Callable[[], Any], attempts: int = 3, base_delay: float = 1.0) -> Any:
    """Reintenta con backoff exponencial (1s, 2s, 4s) ante errores de red."""
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return func()
        except TickerNotFoundError:
            raise  # no tiene sentido reintentar un ticker inexistente
        except Exception as exc:  # yfinance lanza tipos muy variados
            last_exc = exc
            if i < attempts - 1:
                time.sleep(base_delay * (2 ** i))
    raise DataSourceError(f"Yahoo Finance no respondió: {last_exc}") from last_exc


_cache: dict[tuple, tuple[float, Any]] = {}


def _cached(key: tuple, loader: Callable[[], Any]) -> Any:
    """Caché en memoria con TTL, compartida entre ejecuciones del agente."""
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < CACHE_TTL_SECONDS:
        return hit[1]
    value = loader()
    _cache[key] = (now, value)
    return value


def normalize_ticker(ticker: str) -> str:
    t = (ticker or "").strip().upper()
    if not t or len(t) > 15 or any(c.isspace() for c in t):
        raise TickerNotFoundError(f"Ticker inválido: '{ticker}'")
    return t


def parse_peers(peers: str | list[str] | None, exclude: str = "") -> list[str]:
    """'KO, PEP;MDLZ' -> ['KO', 'PEP', 'MDLZ'] sin duplicados ni el propio ticker."""
    if not peers:
        return []
    raw = peers if isinstance(peers, list) else re.split(r"[,;\s]+", peers)
    out: list[str] = []
    for p in raw:
        p = (p or "").strip().upper()
        if p and p != exclude and p not in out and len(p) <= 15:
            out.append(p)
    return out[:MAX_PEERS]


def benchmark_for(ticker: str, currency: str | None = None) -> str:
    """Índice de referencia: Merval para BYMA, S&P 500 (SPY) para el resto."""
    return "^MERV" if ticker.endswith(".BA") or currency == "ARS" else "SPY"


# ---------------------------------------------------------------------------
# Descarga de datos
# ---------------------------------------------------------------------------

def fetch_history(ticker: str, period: str) -> pd.DataFrame:
    """Descarga OHLCV diario para la temporalidad pedida más el warm up."""
    if period not in PERIOD_DAYS:
        raise ValueError(f"Temporalidad inválida: {period}. Usá {list(PERIOD_DAYS)}")
    start = datetime.now(timezone.utc) - timedelta(days=PERIOD_DAYS[period] + WARMUP_DAYS)

    def _load() -> pd.DataFrame:
        df = yf.Ticker(ticker).history(start=start.strftime("%Y-%m-%d"), interval="1d",
                                       auto_adjust=True)
        if df is None or df.empty or "Close" not in df:
            raise TickerNotFoundError(
                f"No hay datos de precios para '{ticker}'. Verificá que el ticker exista "
                "en Yahoo Finance (ej. GGAL, VIST, AAPL, SPY; para BYMA usá sufijo .BA)."
            )
        df = df.dropna(subset=["Close"])
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df[~df.index.duplicated(keep="last")]

    return _cached(("history", ticker, period), lambda: _retry(_load))


def fetch_info(ticker: str) -> dict[str, Any]:
    """Perfil y ratios. Reintenta y, si .info falla, usa fast_info como respaldo."""
    def _load() -> dict[str, Any]:
        t = yf.Ticker(ticker)
        try:
            info = _retry(lambda: t.info or {}, attempts=2)
        except DataSourceError:
            info = {}
        if not info.get("marketCap"):
            try:
                fi = t.fast_info
                info.setdefault("marketCap", fi.get("marketCap"))
                info.setdefault("currency", fi.get("currency"))
                info.setdefault("exchange", fi.get("exchange"))
            except Exception:
                pass
        return info

    return _cached(("info", ticker), _load)


def fetch_statements(ticker: str) -> dict[str, pd.DataFrame]:
    """Estados contables anuales y trimestrales (DataFrames vacíos si no hay)."""
    def _load() -> dict[str, pd.DataFrame]:
        t = yf.Ticker(ticker)
        out = {}
        for key, attr in (("income", "income_stmt"), ("balance", "balance_sheet"),
                          ("cashflow", "cashflow"), ("q_income", "quarterly_income_stmt"),
                          ("q_cashflow", "quarterly_cashflow")):
            try:
                df = getattr(t, attr)
                out[key] = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
            except Exception:
                out[key] = pd.DataFrame()
        return out

    return _cached(("statements", ticker), _load)


def fetch_analyst_data(ticker: str) -> dict[str, Any]:
    """Tendencia de recomendaciones y últimos cambios de calificación."""
    def _load() -> dict[str, Any]:
        t = yf.Ticker(ticker)
        out: dict[str, Any] = {}
        try:
            rs = t.recommendations_summary
            if isinstance(rs, pd.DataFrame) and not rs.empty:
                out["trend"] = rs.to_dict("records")
        except Exception:
            pass
        try:
            ud = t.upgrades_downgrades
            if isinstance(ud, pd.DataFrame) and not ud.empty:
                ud = ud.sort_index(ascending=False).head(6).reset_index()
                out["rating_changes"] = [
                    {"date": str(r.get("GradeDate", ""))[:10], "firm": r.get("Firm"),
                     "action": r.get("Action"), "from": r.get("FromGrade"),
                     "to": r.get("ToGrade"), "price_target": _r(r.get("currentPriceTarget"))}
                    for r in ud.to_dict("records")
                ]
        except Exception:
            pass
        return out

    return _cached(("analyst", ticker), _load)


def fetch_risk_free() -> float | None:
    """Rendimiento del Treasury a 10 años en % (^TNX)."""
    def _load() -> float | None:
        try:
            h = yf.Ticker("^TNX").history(period="10d")
            return _r(h["Close"].dropna().iloc[-1], 3)
        except Exception:
            return None

    return _cached(("rf",), _load)


# ---------------------------------------------------------------------------
# Fundamentales: tabla anual, TTM y snapshot comparable entre empresas
# ---------------------------------------------------------------------------

def _row(df: pd.DataFrame, *names: str) -> pd.Series | None:
    for n in names:
        if df is not None and not df.empty and n in df.index:
            return df.loc[n]
    return None


def _valid_cols(series: pd.Series | None, newest_first: bool = False) -> list:
    if series is None:
        return []
    cols = [c for c in series.index if _clean_number(series[c]) is not None]
    return sorted(cols, reverse=newest_first)


def ttm_from_quarters(st: dict[str, pd.DataFrame]) -> dict[str, Any] | None:
    """Suma de los últimos 4 trimestres consecutivos (None si no hay 4)."""
    qi, qc = st.get("q_income"), st.get("q_cashflow")
    cols = _valid_cols(_row(qi, "Total Revenue", "Operating Revenue"), newest_first=True)[:4]
    if len(cols) < 4 or not 240 <= (cols[0] - cols[3]).days <= 300:
        return None

    def total(df, *names):
        r = _row(df, *names)
        if r is None:
            return None
        vals = [_clean_number(r.get(c)) for c in cols]
        return sum(vals) if all(v is not None for v in vals) else None

    out = {
        "end": cols[0].strftime("%Y-%m-%d"),
        "revenue": total(qi, "Total Revenue", "Operating Revenue"),
        "gross_profit": total(qi, "Gross Profit"),
        "operating_income": total(qi, "Operating Income", "EBIT"),
        "ebitda": total(qi, "EBITDA", "Normalized EBITDA"),
        "net_income": total(qi, "Net Income Common Stockholders", "Net Income"),
        "operating_cf": total(qc, "Operating Cash Flow"),
        "capex": total(qc, "Capital Expenditure"),
        "fcf": total(qc, "Free Cash Flow"),
    }
    if out["fcf"] is None and out["operating_cf"] is not None and out["capex"] is not None:
        out["fcf"] = out["operating_cf"] + out["capex"]
    if not out["gross_profit"]:  # bancos y aseguradoras: Yahoo informa 0, no aplica
        out["gross_profit"] = None
    return out


def last_quarter_yoy(st: dict[str, pd.DataFrame]) -> dict[str, Any] | None:
    """Último trimestre vs el mismo trimestre del año anterior (por fecha, no por posición)."""
    qi = st.get("q_income")
    rev = _row(qi, "Total Revenue", "Operating Revenue")
    cols = _valid_cols(rev, newest_first=True)
    if len(cols) < 2:
        return None
    last = cols[0]
    prior = next((c for c in cols[1:] if 330 <= (last - c).days <= 400), None)
    if prior is None:
        return None
    ni = _row(qi, "Net Income Common Stockholders", "Net Income")

    def growth(s):
        if s is None:
            return None
        a, b = _clean_number(s.get(last)), _clean_number(s.get(prior))
        return _r((a / b - 1) * 100, 1) if a is not None and b and b > 0 else None

    return {"quarter": last.strftime("%Y-%m"), "vs_quarter": prior.strftime("%Y-%m"),
            "revenue_pct": growth(rev), "net_income_pct": growth(ni)}


FINANCIAL_INDUSTRIES = ("Banks", "Insurance", "Mortgage Finance", "Capital Markets")


def is_financial_institution(industry: str | None) -> bool:
    return bool(industry) and industry.startswith(FINANCIAL_INDUSTRIES)


def fundamental_snapshot(ticker: str) -> dict[str, Any]:
    """Métricas comparables entre empresas, todas con el mismo método.

    La usan get_market_data (empresa analizada) y compare_peers (comparables),
    así los números del informe y de la tabla de comparables coinciden.
    """
    def _load() -> dict[str, Any]:
        info = fetch_info(ticker)
        quote_type = info.get("quoteType")
        st = fetch_statements(ticker) if quote_type != "ETF" else {}
        ttm = ttm_from_quarters(st) if st else None
        yoy = last_quarter_yoy(st) if st else None
        cur = info.get("currency")
        rep = info.get("financialCurrency") or cur
        mismatch = bool(cur and rep and cur != rep)
        mcap = _clean_number(info.get("marketCap"))
        ev = _clean_number(info.get("enterpriseValue"))
        total_debt = _clean_number(info.get("totalDebt"))
        total_cash = _clean_number(info.get("totalCash"))
        net_debt = total_debt - total_cash if total_debt is not None and total_cash is not None else None

        def margin(key, info_key):
            if ttm and ttm.get(key) is not None and ttm.get("revenue"):
                return _pct(ttm[key] / ttm["revenue"])
            return _pct(info.get(info_key))

        ttm_ebitda = ttm.get("ebitda") if ttm else None
        ttm_rev = ttm.get("revenue") if ttm else None
        snap = {
            "ticker": ticker,
            "name": info.get("shortName") or info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "quote_type": quote_type,
            "currency": cur,
            "reporting_currency": rep,
            "currency_mismatch": mismatch,
            "fundamentals_basis": (f"TTM = 4 trimestres al {ttm['end']}" if ttm
                                   else "Yahoo Finance (info)"),
            "market_cap": mcap,
            "enterprise_value": ev,
            "pe_trailing": _r(_positive(info.get("trailingPE"))),
            "pe_forward": _r(_positive(info.get("forwardPE"))),
            "ev_ebitda": None if mismatch else _r(_positive(
                _div(ev, ttm_ebitda) if ttm_ebitda else info.get("enterpriseToEbitda"))),
            "ev_revenue": None if mismatch else _r(_positive(
                _div(ev, ttm_rev) if ttm_rev else info.get("enterpriseToRevenue"))),
            "price_to_sales": None if mismatch else _r(_positive(
                _div(mcap, ttm_rev) if ttm_rev else info.get("priceToSalesTrailing12Months"))),
            "price_to_book": None if mismatch else _r(_positive(info.get("priceToBook"))),
            "fcf_yield_pct": None if mismatch or not ttm else _pct(_div(ttm.get("fcf"), mcap), 2),
            "gross_margin_pct": margin("gross_profit", "grossMargins"),
            "operating_margin_pct": margin("operating_income", "operatingMargins"),
            "net_margin_pct": margin("net_income", "profitMargins"),
            "ebitda_margin_pct": margin("ebitda", "ebitdaMargins"),
            "roe_pct": _pct(info.get("returnOnEquity")),
            "revenue_growth_yoy_pct": None if mismatch else (
                yoy["revenue_pct"] if yoy and yoy.get("revenue_pct") is not None
                else _pct(info.get("revenueGrowth"))),
            "debt_to_equity_pct": _r(info.get("debtToEquity")),
            "net_debt": net_debt,
            "net_debt_to_ebitda": _r(_div(net_debt, info.get("ebitda") if mismatch or not ttm_ebitda
                                          else ttm_ebitda)),
            "dividend_yield_pct": None if mismatch else _pct(info.get("trailingAnnualDividendYield"), 2),
            "return_52w_pct": _pct(info.get("52WeekChange")) if info.get("52WeekChange") is not None
            else _r(info.get("fiftyTwoWeekChangePercent")),
            "ttm": ttm,
            "last_quarter_yoy": yoy,
        }
        if snap["net_debt_to_ebitda"] is not None and (_clean_number(info.get("ebitda")) or 0) <= 0 \
                and not ttm_ebitda:
            snap["net_debt_to_ebitda"] = None
        # En bancos y aseguradoras la deuda es materia prima del negocio: EBITDA,
        # deuda neta, margen bruto y FCF no tienen sentido económico.
        snap["financial_institution"] = is_financial_institution(info.get("industry"))
        if snap["financial_institution"]:
            for k in ("ev_ebitda", "ev_revenue", "fcf_yield_pct", "gross_margin_pct",
                      "ebitda_margin_pct", "net_debt", "net_debt_to_ebitda"):
                snap[k] = None
        return snap

    return _cached(("snapshot", ticker), _load)


def _statement_table(inc: pd.DataFrame, bal: pd.DataFrame, cf: pd.DataFrame) -> pd.DataFrame:
    """Arma una tabla año por fila con las líneas clave y ratios derivados."""
    cols = _valid_cols(_row(inc, "Total Revenue", "Operating Revenue"))
    if not cols:
        return pd.DataFrame()

    def line(df, *names):
        s = _row(df, *names)
        return s.reindex(cols).astype(float) if s is not None else pd.Series(np.nan, index=cols)

    t = pd.DataFrame(index=pd.Index(cols, name="fecha"))
    t["revenue"] = line(inc, "Total Revenue", "Operating Revenue")
    t["gross_profit"] = line(inc, "Gross Profit").replace(0, np.nan)  # 0 en bancos: no aplica
    t["operating_income"] = line(inc, "Operating Income", "EBIT")
    t["ebitda"] = line(inc, "EBITDA", "Normalized EBITDA")
    t["net_income"] = line(inc, "Net Income Common Stockholders", "Net Income")
    t["eps_diluted"] = line(inc, "Diluted EPS")
    t["shares_diluted"] = line(inc, "Diluted Average Shares")
    t["interest_expense"] = line(inc, "Interest Expense", "Interest Expense Non Operating")
    t["tax_rate"] = line(inc, "Tax Rate For Calcs")
    t["operating_cf"] = line(cf, "Operating Cash Flow")
    t["capex"] = line(cf, "Capital Expenditure")
    t["fcf"] = line(cf, "Free Cash Flow")
    t["fcf"] = t["fcf"].fillna(t["operating_cf"] + t["capex"])
    t["dividends_paid"] = -line(cf, "Cash Dividends Paid")
    t["buybacks"] = -line(cf, "Repurchase Of Capital Stock")
    t["sbc"] = line(cf, "Stock Based Compensation")
    t["total_debt"] = line(bal, "Total Debt")
    t["cash"] = line(bal, "Cash Cash Equivalents And Short Term Investments",
                     "Cash And Cash Equivalents")
    t["net_debt"] = t["total_debt"] - t["cash"]
    t["equity"] = line(bal, "Stockholders Equity", "Common Stock Equity")
    t["invested_capital"] = line(bal, "Invested Capital")

    t["gross_margin"] = t["gross_profit"] / t["revenue"] * 100
    t["operating_margin"] = t["operating_income"] / t["revenue"] * 100
    t["net_margin"] = t["net_income"] / t["revenue"] * 100
    t["fcf_margin"] = t["fcf"] / t["revenue"] * 100
    t["fcf_to_net_income"] = t["fcf"] / t["net_income"].where(t["net_income"] > 0) * 100
    t["capex_to_revenue"] = -t["capex"] / t["revenue"] * 100
    t["roe"] = t["net_income"] / t["equity"].where(t["equity"] > 0) * 100
    tax = t["tax_rate"].clip(0, 0.5).fillna(0.25)
    t["roic"] = t["operating_income"] * (1 - tax) / t["invested_capital"].where(
        t["invested_capital"] > 0) * 100
    t["net_debt_to_ebitda"] = t["net_debt"] / t["ebitda"].where(t["ebitda"] > 0)
    t["interest_coverage"] = t["operating_income"] / t["interest_expense"].abs().where(
        t["interest_expense"].abs() > 0)
    t["revenue_growth"] = t["revenue"].pct_change(fill_method=None) * 100
    # Montos que yfinance deja en -0.0 cuando no hubo pagos.
    for c in ("dividends_paid", "buybacks"):
        t[c] = t[c].where(t[c] != 0, 0.0)
    return t


# ---------------------------------------------------------------------------
# Noticias (Yahoo Finance search + Google News RSS)
# ---------------------------------------------------------------------------

_NAME_SUFFIXES = re.compile(
    r"[,.]?\s+(inc|corp|corporation|co|company|ltd|limited|plc|sa|s\.a|s\.a\.|"
    r"s\.a\.b\. de c\.v\.|sab de cv|de c\.v\.|n\.v\.|nv|se|ag|ab|asa|spa|holdings?|"
    r"group|the|class [a-z]|adr|ads)\.?$",
    re.IGNORECASE,
)


def company_query_name(name: str) -> str:
    """'Vista Energy, S.A.B. de C.V.' -> 'Vista Energy'."""
    n = (name or "").strip()
    for _ in range(4):
        new = _NAME_SUFFIXES.sub("", n).strip(" ,.")
        if new == n:
            break
        n = new
    return n or (name or "").strip()


def _fetch_yahoo_news(ticker: str) -> list[dict[str, Any]]:
    # Ticker.news usa un endpoint que hoy devuelve una lista vacía para todos
    # los tickers; yf.Search sí trae titulares relacionados.
    base = ticker.split(".")[0]
    s = yf.Search(base, news_count=20, max_results=1, enable_fuzzy_query=False)
    items = []
    for n in s.news or []:
        related = {r.upper() for r in (n.get("relatedTickers") or [])}
        if related and not ({ticker, base} & related):
            continue
        ts = n.get("providerPublishTime")
        items.append({
            "title": n.get("title"),
            "publisher": n.get("publisher"),
            "url": n.get("link"),
            "date": datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None,
            "source": "Yahoo Finance",
            "lang": "en",
        })
    return items


def _fetch_google_news(query: str, lang: str) -> list[dict[str, Any]]:
    locale = "hl=es-419&gl=AR&ceid=AR:es-419" if lang == "es" else "hl=en-US&gl=US&ceid=US:en"
    url = (f"https://news.google.com/rss/search?q={quote_plus(query)}"
           f"+when:{NEWS_LOOKBACK_DAYS}d&{locale}")
    resp = requests.get(url, headers=HTTP_HEADERS, timeout=15)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    items = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        src = it.find("source")
        publisher = src.text.strip() if src is not None and src.text else None
        # Google agrega " - Medio" al final del título.
        if publisher and title.endswith(f" - {publisher}"):
            title = title[: -len(publisher) - 3].strip()
        try:
            date = parsedate_to_datetime(it.findtext("pubDate") or "")
        except (TypeError, ValueError):
            date = None
        items.append({"title": title, "publisher": publisher, "url": it.findtext("link"),
                      "date": date, "source": "Google News", "lang": lang})
    return items


def fetch_news(ticker: str, company_name: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Titulares de varias fuentes, deduplicados y ordenados por fecha.

    Devuelve (titulares, errores por fuente). Una fuente caída no anula a las demás.
    """
    def _load():
        name = company_query_name(company_name) or ticker.split(".")[0]
        sources = {
            "Yahoo Finance": lambda: _fetch_yahoo_news(ticker),
            "Google News (ES)": lambda: _fetch_google_news(f'"{name}"', "es"),
            "Google News (EN)": lambda: _fetch_google_news(f'"{name}" stock', "en"),
        }
        items, errors = [], []
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {k: pool.submit(f) for k, f in sources.items()}
            for k, fut in futures.items():
                try:
                    items.extend(fut.result(timeout=25))
                except Exception as exc:
                    errors.append(f"{k}: {type(exc).__name__}")

        cutoff = datetime.now(timezone.utc) - timedelta(days=NEWS_LOOKBACK_DAYS)
        seen, per_source = set(), {}
        for it in sorted(items, key=lambda x: x["date"] or cutoff, reverse=True):
            if not it.get("title"):
                continue
            if it["date"] and it["date"] < cutoff:
                continue
            key = re.sub(r"[^a-z0-9áéíóúñ]", "", it["title"].lower())[:60]
            if key in seen:
                continue
            seen.add(key)
            per_source.setdefault((it["source"], it["lang"]), []).append(it)
        # Round robin entre fuentes para que una sola no tape a las demás.
        picked: list[dict[str, Any]] = []
        queues = list(per_source.values())
        while queues and len(picked) < MAX_HEADLINES:
            for q in list(queues):
                if q:
                    picked.append(q.pop(0))
                if not q:
                    queues.remove(q)
        picked.sort(key=lambda x: x["date"] or cutoff, reverse=True)
        return picked[:MAX_HEADLINES], errors

    return _cached(("news", ticker, company_name), _load)


# ---------------------------------------------------------------------------
# Indicadores técnicos (pandas nativo, sin TA-Lib)
# ---------------------------------------------------------------------------

def rsi_wilder(close: pd.Series, window: int = 14) -> pd.Series:
    """RSI con el suavizado de Wilder (equivale al RSI de TA-Lib)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    # Sin pérdidas en la ventana el RSI es 100 por definición.
    return rsi.where(avg_loss != 0, 100.0)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]
    out["SMA20"] = close.rolling(20, min_periods=20).mean()
    out["SMA50"] = close.rolling(50, min_periods=50).mean()
    out["SMA200"] = close.rolling(200, min_periods=200).mean()
    out["RSI14"] = rsi_wilder(close, 14)
    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACDSignal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACDHist"] = out["MACD"] - out["MACDSignal"]
    # Bandas de Bollinger (20, 2)
    std20 = close.rolling(20, min_periods=20).std()
    out["BBUpper"] = out["SMA20"] + 2 * std20
    out["BBLower"] = out["SMA20"] - 2 * std20
    # ATR (14, Wilder)
    if {"High", "Low"} <= set(out.columns):
        prev = close.shift(1)
        tr = pd.concat([out["High"] - out["Low"], (out["High"] - prev).abs(),
                        (out["Low"] - prev).abs()], axis=1).max(axis=1)
        out["ATR14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    out["LogRet"] = np.log(close / close.shift(1))
    out["Vol20"] = out["LogRet"].rolling(20).std() * math.sqrt(TRADING_DAYS)
    return out


def visible_window(df: pd.DataFrame, period: str) -> pd.DataFrame:
    cutoff = df.index.max() - pd.Timedelta(days=PERIOD_DAYS[period])
    return df[df.index >= cutoff]


def _return_since(close: pd.Series, days: int) -> float | None:
    past = close[close.index <= close.index.max() - pd.Timedelta(days=days)]
    if past.empty:
        return None
    return round((close.iloc[-1] / past.iloc[-1] - 1) * 100, 2)


def _last_cross(a: pd.Series, b: pd.Series) -> dict[str, Any] | None:
    """Último cruce de la serie a sobre la b, con la fecha desde la que rige."""
    both = pd.concat([a, b], axis=1).dropna()
    if len(both) < 2:
        return None
    sign = np.sign(both.iloc[:, 0] - both.iloc[:, 1]).replace(0, np.nan).ffill().dropna()
    flips = sign[sign.diff().fillna(0) != 0]
    if flips.empty:
        return {"date": None, "direction": "arriba" if sign.iloc[-1] > 0 else "abajo",
                "since_at_least": both.index[0].strftime("%Y-%m-%d")}
    d = flips.index[-1]
    return {"date": d.strftime("%Y-%m-%d"), "direction": "arriba" if flips.iloc[-1] > 0 else "abajo",
            "days_ago": int((both.index[-1] - d).days),
            "crosses_last_20_sessions": int((sign.tail(21).diff().fillna(0) != 0).sum())}


# ---------------------------------------------------------------------------
# Registro de herramientas
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    func: Callable[..., dict[str, Any]]


TOOLS: dict[str, Tool] = {}


def tool(name: str, description: str, properties: dict[str, Any]):
    """Decorador que registra una función como herramienta del agente."""
    def decorator(func: Callable[..., dict[str, Any]]):
        TOOLS[name] = Tool(
            name=name,
            description=description,
            input_schema={
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
            func=func,
        )
        return func
    return decorator


@dataclass
class AgentContext:
    """Memoria de trabajo del agente.

    Las herramientas devuelven al LLM un resumen JSON compacto, pero guardan
    acá los objetos pesados (DataFrames) que la UI necesita para graficar.
    """
    period: str = "1Y"
    info: dict[str, Any] = field(default_factory=dict)
    market: dict[str, Any] | None = None
    financials: dict[str, Any] | None = None
    technicals: dict[str, Any] | None = None
    peers: dict[str, Any] | None = None
    news: dict[str, Any] | None = None
    chart_df: pd.DataFrame | None = None
    benchmark_df: pd.DataFrame | None = None
    annual_df: pd.DataFrame | None = None
    quarterly_df: pd.DataFrame | None = None


TICKER_PROP = {"type": "string", "description": "Ticker de Yahoo Finance, ej. AAPL, GGAL, VIST, SPY, YPFD.BA"}


@tool(
    "get_market_data",
    "Perfil de la empresa, cotización, múltiplos de valuación (P/E trailing y forward, PEG, "
    "EV/EBITDA, EV/Ventas, P/S, P/B, earnings y FCF yield vs bono del Tesoro a 10 años), "
    "rentabilidad TTM (márgenes, ROE), balance (deuda neta, liquidez), consenso de analistas "
    "(precio objetivo, upside, recomendación y cambios recientes), tenencia de insiders e "
    "institucionales, short interest, beta y fecha del próximo balance. Incluye 'facts' con "
    "lecturas ya calculadas.",
    {"ticker": TICKER_PROP},
)
def get_market_data(ticker: str, ctx: AgentContext) -> dict[str, Any]:
    ticker = normalize_ticker(ticker)
    hist = fetch_history(ticker, "1Y")  # 1Y + warm up alcanza para todo lo de abajo
    info = fetch_info(ticker)
    ctx.info = info
    snap = fundamental_snapshot(ticker)

    close = hist["Close"]
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) > 1 else None
    change_pct = (last / prev - 1) * 100 if prev else None
    quote_type = info.get("quoteType")
    currency = info.get("currency", "USD")

    # Dividend yield: el mismo dato que la tabla de comparables. Si no está (o
    # la moneda de reporte difiere), los dividendos cobrados en los últimos
    # 365 días sobre el precio, ambos en la moneda de cotización.
    div_yield = snap["dividend_yield_pct"]
    if div_yield is None and "Dividends" in hist:
        last_year = hist[hist.index > hist.index.max() - pd.Timedelta(days=365)]
        paid = float(last_year["Dividends"].sum())
        div_yield = paid / last * 100 if paid > 0 else 0.0

    trailing_pe, forward_pe = snap["pe_trailing"], snap["pe_forward"]
    rf = fetch_risk_free()
    earnings_yield = 100 / trailing_pe if trailing_pe else None

    target_mean = _clean_number(info.get("targetMeanPrice"))
    upside = (target_mean / last - 1) * 100 if target_mean else None

    earnings_ts = info.get("earningsTimestampStart") or info.get("earningsTimestamp")
    next_earnings = None
    if earnings_ts:
        d = datetime.fromtimestamp(earnings_ts, tz=timezone.utc)
        if d >= datetime.now(timezone.utc) - timedelta(days=1):
            next_earnings = d.strftime("%Y-%m-%d")

    analyst = fetch_analyst_data(ticker) if quote_type != "ETF" else {}
    trend = {row.get("period"): row for row in analyst.get("trend") or []}
    rec_trend = None
    if trend.get("0m"):
        def _bullish(row):
            if not row:
                return None
            tot = sum(row.get(k, 0) or 0 for k in ("strongBuy", "buy", "hold", "sell", "strongSell"))
            return round(((row.get("strongBuy") or 0) + (row.get("buy") or 0)) / tot * 100, 1) if tot else None
        rec_trend = {"pct_buy_now": _bullish(trend.get("0m")),
                     "pct_buy_3m_ago": _bullish(trend.get("-3m")),
                     "now": {k: trend["0m"].get(k) for k in ("strongBuy", "buy", "hold", "sell", "strongSell")}}

    high_52w = float(close.tail(TRADING_DAYS).max())
    low_52w = float(close.tail(TRADING_DAYS).min())
    data = {
        "ticker": ticker,
        "name": info.get("longName") or info.get("shortName") or ticker,
        "quote_type": quote_type,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "country": info.get("country"),
        "employees": info.get("fullTimeEmployees"),
        "currency": currency,
        "financial_currency": snap["reporting_currency"],
        "summary": (info.get("longBusinessSummary") or "")[:900],
        # Cotización
        "price": round(last, 4),
        "previous_close": round(prev, 4) if prev else None,
        "change_pct": round(change_pct, 2) if change_pct is not None else None,
        "volume": int(hist["Volume"].iloc[-1]) if "Volume" in hist else None,
        "avg_volume_3m": int(hist["Volume"].tail(63).mean()) if "Volume" in hist else None,
        "high_52w_close": round(high_52w, 4),
        "low_52w_close": round(low_52w, 4),
        "last_date": hist.index[-1].strftime("%Y-%m-%d"),
        # Tamaño
        "market_cap": snap["market_cap"],
        "enterprise_value": snap["enterprise_value"],
        # Valuación
        "pe_trailing": trailing_pe,
        "pe_forward": forward_pe,
        "peg": _r(_positive(info.get("trailingPegRatio") or info.get("pegRatio"))),
        "ev_ebitda": snap["ev_ebitda"],
        "ev_revenue": snap["ev_revenue"],
        "price_to_sales": snap["price_to_sales"],
        "price_to_book": snap["price_to_book"],
        "eps_trailing": _r(info.get("trailingEps")),
        "eps_forward": _r(info.get("forwardEps")),
        "earnings_yield_pct": _r(earnings_yield),
        "fcf_yield_pct": snap["fcf_yield_pct"],
        "us10y_yield_pct": rf,
        "dividend_yield_pct": round(div_yield, 2) if div_yield is not None else None,
        "payout_ratio_pct": _pct(info.get("payoutRatio")),
        # Rentabilidad TTM (misma base que la tabla de comparables)
        "fundamentals_basis": snap["fundamentals_basis"],
        "gross_margin_ttm_pct": snap["gross_margin_pct"],
        "ebitda_margin_ttm_pct": snap["ebitda_margin_pct"],
        "operating_margin_ttm_pct": snap["operating_margin_pct"],
        "net_margin_ttm_pct": snap["net_margin_pct"],
        "roe_ttm_pct": snap["roe_pct"],
        "revenue_ttm": (snap["ttm"] or {}).get("revenue"),
        "ebitda_ttm": (snap["ttm"] or {}).get("ebitda"),
        "free_cash_flow_ttm": (snap["ttm"] or {}).get("fcf"),
        "last_quarter_yoy": snap["last_quarter_yoy"],
        # Balance (MRQ = último balance trimestral)
        "net_debt_mrq": snap["net_debt"],
        "net_debt_to_ebitda_ttm": snap["net_debt_to_ebitda"],
        "debt_to_equity_pct": snap["debt_to_equity_pct"],
        "current_ratio": _r(info.get("currentRatio")),
        "quick_ratio": _r(info.get("quickRatio")),
        # Riesgo y tenencia
        "beta_5y_monthly": _r(info.get("beta")),
        "short_pct_float": _pct(info.get("shortPercentOfFloat")),
        "short_ratio_days": _r(info.get("shortRatio")),
        "insiders_pct": _pct(info.get("heldPercentInsiders")),
        "institutions_pct": _pct(info.get("heldPercentInstitutions")),
        # Consenso de analistas
        "analyst_count": info.get("numberOfAnalystOpinions"),
        "recommendation": info.get("recommendationKey"),
        "recommendation_mean": _r(info.get("recommendationMean")),  # 1 = compra fuerte, 5 = venta
        "target_mean": _r(target_mean),
        "target_high": _r(info.get("targetHighPrice")),
        "target_low": _r(info.get("targetLowPrice")),
        "upside_to_target_mean_pct": _r(upside),
        "recommendation_trend": rec_trend,
        "recent_rating_changes": analyst.get("rating_changes"),
        "next_earnings_date": next_earnings,
    }
    if quote_type == "ETF":
        data.update({
            "etf_expense_ratio_pct": _r(info.get("netExpenseRatio")),
            "etf_total_assets": _clean_number(info.get("totalAssets")),
            "etf_yield_pct": _pct(info.get("yield")),
        })

    notes = []
    if snap["currency_mismatch"]:
        notes.append(f"Cotiza en {currency} pero reporta en {snap['reporting_currency']}: se omitieron "
                     "EV/EBITDA, EV/Ventas, P/S, P/B y FCF yield porque mezclarían monedas.")
    if snap["reporting_currency"] == "ARS":
        notes.append("Estados en ARS: el crecimiento nominal está inflado por la inflación; "
                     "priorizá márgenes, ROE y métricas en USD.")
    if snap["financial_institution"]:
        notes.append("Entidad financiera: margen bruto, EBITDA, deuda neta/EBITDA y free cash flow no "
                     "son métricas relevantes y se omiten. Analizala por ROE, margen neto, P/E y P/B.")
    if not info.get("longBusinessSummary") and not trailing_pe and quote_type != "ETF":
        notes.append("Yahoo devolvió pocos datos de perfil y ratios para este ticker.")
    if notes:
        data["notes"] = notes
    data["facts"] = _market_facts(data)
    ctx.market = data
    return data


def _market_facts(m: dict[str, Any]) -> list[str]:
    """Lecturas de valuación ya calculadas, para que el LLM no las derive."""
    f: list[str] = []
    cur = m.get("currency", "")
    pe, fpe = m.get("pe_trailing"), m.get("pe_forward")
    if pe and fpe:
        implied = (pe / fpe - 1) * 100
        if m.get("financial_currency") != cur or abs(implied) > 100:
            f.append(f"P/E trailing {_g(pe)}x y forward {_g(fpe)}x: la diferencia implicaría una variación "
                     f"de la utilidad por acción de {implied:+.0f}%, un valor extremo que suele indicar que "
                     "el EPS histórico y el estimado están en bases distintas (moneda, inflación o "
                     "ajustes). No lo presentes como crecimiento esperado; mencionalo como dato a verificar.")
        else:
            f.append(f"P/E forward ({_g(fpe)}x) {'menor' if fpe < pe else 'mayor'} que el trailing ({_g(pe)}x): "
                     f"el consenso espera que la utilidad por acción {'crezca' if implied > 0 else 'caiga'} "
                     f"{abs(implied):.1f}% en los próximos 12 meses.")
    elif m.get("quote_type") != "ETF" and not pe:
        f.append("Sin P/E trailing: la empresa tiene pérdidas o Yahoo no publica el dato.")
    margins = [(label, m.get(k)) for k, label in (("gross_margin_ttm_pct", "bruto"),
                                                  ("operating_margin_ttm_pct", "operativo"),
                                                  ("net_margin_ttm_pct", "neto"))
               if m.get(k) is not None]
    if margins:
        f.append(f"Márgenes TTM ({m.get('fundamentals_basis')}): "
                 + ", ".join(f"{label} {_g(v)}%" for label, v in margins)
                 + (f"; ROE TTM {_g(m['roe_ttm_pct'])}%." if m.get("roe_ttm_pct") is not None else ".")
                 + " Son datos de los últimos 12 meses: no los compares con años fiscales como si "
                   "fueran la misma serie.")
    if m.get("net_debt_mrq") is not None:
        f.append(f"Deuda neta (último balance, MRQ): {_money(m['net_debt_mrq'], m.get('financial_currency') or '')}"
                 + (" (caja neta)." if m["net_debt_mrq"] < 0 else "."))
    rf = m.get("us10y_yield_pct")
    for key, label in (("earnings_yield_pct", "Earnings yield"), ("fcf_yield_pct", "FCF yield (TTM)")):
        v = m.get(key)
        if v is not None and rf is not None:
            gap = v - rf
            f.append(f"{label} {_g(v)}% {'por encima' if gap > 0 else 'por debajo'} del Treasury a "
                     f"10 años ({_g(rf)}%) por {abs(gap):.2f} pp.")
    if m.get("target_mean"):
        up = m.get("upside_to_target_mean_pct")
        f.append(f"Precio objetivo MEDIO del consenso {_g(m['target_mean'])} {cur} "
                 f"(rango {_g(m['target_low'] or 0)} a {_g(m['target_high'] or 0)}), "
                 f"{'upside' if (up or 0) >= 0 else 'downside'} {abs(up or 0):.1f}% vs el precio "
                 f"{_g(m['price'])}, con {m.get('analyst_count')} analistas.")
        tl = m.get("target_low")
        if tl and tl > m["price"]:
            f.append(f"Incluso el precio objetivo más bajo ({_g(tl)}) está por encima del precio actual.")
    rt = m.get("recommendation_trend")
    if rt and rt.get("pct_buy_now") is not None:
        f.append(f"{_g(rt['pct_buy_now'])}% de los analistas recomienda comprar"
                 + (f" (hace 3 meses: {_g(rt['pct_buy_3m_ago'])}%)." if rt.get("pct_buy_3m_ago") is not None else "."))
    nd = m.get("net_debt_to_ebitda_ttm")
    if nd is not None:
        tag = "caja neta (deuda neta negativa)" if nd < 0 else "bajo" if nd < 1.5 else \
              "moderado" if nd < 3 else "alto"
        f.append(f"Deuda neta MRQ / EBITDA TTM {_g(nd)}x: apalancamiento {tag}.")
    if m.get("dividend_yield_pct") is not None and m.get("payout_ratio_pct") is not None \
            and m["payout_ratio_pct"] > 100:
        f.append(f"Payout {_g(m['payout_ratio_pct'])}%: paga más dividendos que su utilidad TTM, "
                 "nivel no sostenible si la utilidad no se recupera.")
    if m.get("beta_5y_monthly") is not None:
        f.append(f"Beta de Yahoo (5 años, mensual) {_g(m['beta_5y_monthly'])}; la beta diaria de 1 año "
                 "vs benchmark está en el análisis técnico y puede diferir.")
    yoy = m.get("last_quarter_yoy")
    if yoy and yoy.get("revenue_pct") is not None:
        f.append(f"Último trimestre ({yoy['quarter']}) vs mismo trimestre del año anterior "
                 f"({yoy['vs_quarter']}): ventas {yoy['revenue_pct']:+.1f}%"
                 + (f", utilidad neta {yoy['net_income_pct']:+.1f}%." if yoy.get("net_income_pct") is not None else "."))
    if m.get("next_earnings_date"):
        f.append(f"Próximo balance: {m['next_earnings_date']}.")
    return f


@tool(
    "get_financial_statements",
    "Estados contables por año fiscal (4/5 años) y últimos trimestres: ventas, márgenes bruto, "
    "operativo y neto, EBITDA, utilidad y EPS, flujo operativo, capex, free cash flow, "
    "conversión de caja, ROE, ROIC, deuda neta/EBITDA, cobertura de intereses, dilución o "
    "recompras, dividendos y CAGR. Incluye 'facts' con las tendencias ya calculadas. "
    "No aplica a ETFs.",
    {"ticker": TICKER_PROP},
)
def get_financial_statements(ticker: str, ctx: AgentContext) -> dict[str, Any]:
    ticker = normalize_ticker(ticker)
    info = ctx.info or fetch_info(ticker)
    if info.get("quoteType") == "ETF":
        data = {"ticker": ticker, "applicable": False,
                "note": "Es un ETF: no tiene estados contables propios."}
        ctx.financials = data
        return data

    st = fetch_statements(ticker)
    annual = _statement_table(st["income"], st["balance"], st["cashflow"])
    if annual.empty:
        data = {"ticker": ticker, "applicable": False,
                "note": "Yahoo Finance no devolvió estados contables para este ticker."}
        ctx.financials = data
        return data
    financial_inst = is_financial_institution(info.get("industry"))
    if financial_inst:
        # En bancos el flujo de caja operativo incluye la variación de préstamos y
        # depósitos: FCF, EBITDA, deuda neta y cobertura de intereses no dicen nada.
        for col in ("gross_profit", "gross_margin", "ebitda", "operating_cf", "capex", "fcf",
                    "fcf_margin", "fcf_to_net_income", "capex_to_revenue", "net_debt",
                    "net_debt_to_ebitda", "interest_coverage", "roic"):
            annual[col] = np.nan
    ctx.annual_df = annual

    fy_end = [d.strftime("%Y-%m-%d") for d in annual.index]
    years = [f"FY{d.year}" for d in annual.index]
    n_years = (annual.index[-1] - annual.index[0]).days / 365.25

    def mm(col):  # montos en millones
        return [_r(v / 1e6, 1) for v in annual[col]]

    def pc(col, digits=1):
        return [_r(v, digits) for v in annual[col]]

    first, last = annual.iloc[0], annual.iloc[-1]
    shares_change = _div(last["shares_diluted"], first["shares_diluted"])

    quarterly = None
    q_rev = _row(st["q_income"], "Total Revenue", "Operating Revenue")
    qcols = _valid_cols(q_rev)
    if qcols:
        q_ni = _row(st["q_income"], "Net Income Common Stockholders", "Net Income")
        q_oi = _row(st["q_income"], "Operating Income", "EBIT")
        q = pd.DataFrame(index=qcols)
        q["revenue"] = q_rev.reindex(qcols).astype(float)
        q["net_income"] = q_ni.reindex(qcols).astype(float) if q_ni is not None else np.nan
        q["operating_income"] = q_oi.reindex(qcols).astype(float) if q_oi is not None else np.nan
        q["operating_margin"] = q["operating_income"] / q["revenue"] * 100
        ctx.quarterly_df = q
        quarterly = {
            "quarter_end": [d.strftime("%Y-%m-%d") for d in q.index],
            "revenue_m": [_r(v / 1e6, 1) for v in q["revenue"]],
            "net_income_m": [_r(v / 1e6, 1) for v in q["net_income"]],
            "operating_margin_pct": [_r(v, 1) for v in q["operating_margin"]],
            "last_quarter_yoy": last_quarter_yoy(st),
        }

    data = {
        "ticker": ticker,
        "applicable": True,
        "currency": info.get("financialCurrency") or info.get("currency"),
        "units": ("campos terminados en _m: millones de la moneda de reporte (un valor 329134.2 "
                  "son 329134.2 millones = 329.1 mil millones); ratios en %; FY = año fiscal"),
        "definitions": {
            "fcf_to_net_income_pct": "free cash flow / utilidad neta (conversión de utilidad en caja)",
            "capex_to_revenue_pct": "capex / ventas",
            "roic_pct": "resultado operativo después de impuestos / capital invertido",
        },
        "financial_institution": financial_inst,
        "fiscal_years": years,
        "fiscal_year_end": fy_end,
        "revenue_m": mm("revenue"),
        "revenue_growth_pct": pc("revenue_growth"),
        "gross_margin_pct": pc("gross_margin"),
        "operating_margin_pct": pc("operating_margin"),
        "net_margin_pct": pc("net_margin"),
        "ebitda_m": mm("ebitda"),
        "net_income_m": mm("net_income"),
        "eps_diluted": pc("eps_diluted", 2),
        "operating_cf_m": mm("operating_cf"),
        "capex_m": mm("capex"),
        "fcf_m": mm("fcf"),
        "fcf_margin_pct": pc("fcf_margin"),
        "fcf_to_net_income_pct": pc("fcf_to_net_income"),
        "capex_to_revenue_pct": pc("capex_to_revenue"),
        "roe_pct": pc("roe"),
        "roic_pct": pc("roic"),
        "net_debt_m": mm("net_debt"),
        "net_debt_to_ebitda": pc("net_debt_to_ebitda", 2),
        "interest_coverage_x": pc("interest_coverage", 1),
        "dividends_paid_m": mm("dividends_paid"),
        "buybacks_m": mm("buybacks"),
        "stock_based_comp_m": mm("sbc"),
        "summary": {
            "period": f"{years[0]} a {years[-1]}",
            "revenue_cagr_pct": _cagr(first["revenue"], last["revenue"], n_years),
            "eps_cagr_pct": _cagr(first["eps_diluted"], last["eps_diluted"], n_years),
            "fcf_cagr_pct": _cagr(first["fcf"], last["fcf"], n_years),
            "operating_margin_change_pp": _r(last["operating_margin"] - first["operating_margin"], 1),
            "avg_roic_pct": _r(annual["roic"].mean(), 1),
            "avg_fcf_to_net_income_pct": _r(annual["fcf_to_net_income"].mean(), 1),
            "years_with_negative_fcf": None if financial_inst else int((annual["fcf"] < 0).sum()),
            "diluted_shares_change_pct": _r((shares_change - 1) * 100, 1) if shares_change else None,
        },
        "quarterly": quarterly,
    }
    data["facts"] = _statement_facts(annual, data)
    ctx.financials = data
    return data


def _trend_word(first: float | None, last: float | None, tol: float = 0.5) -> str | None:
    if first is None or last is None:
        return None
    return "subió" if last - first > tol else "bajó" if first - last > tol else "se mantuvo estable"


def _statement_facts(annual: pd.DataFrame, d: dict[str, Any]) -> list[str]:
    f: list[str] = []
    y0, y1 = d["fiscal_years"][0], d["fiscal_years"][-1]
    s = d["summary"]
    for col, label in (("gross_margin", "Margen bruto"), ("operating_margin", "Margen operativo"),
                       ("net_margin", "Margen neto"), ("roic", "ROIC"), ("roe", "ROE")):
        a, b = _r(annual[col].iloc[0], 1), _r(annual[col].iloc[-1], 1)
        w = _trend_word(a, b)
        if w:  # diferencia sobre los valores redondeados que se muestran
            f.append(f"{label} {w}: {a:.1f}% en {y0} vs {b:.1f}% en {y1} ({b - a:+.1f} pp).")
    if s["revenue_cagr_pct"] is not None:
        f.append(f"CAGR de ventas {y0} a {y1}: {s['revenue_cagr_pct']:.1f}% anual (nominal, "
                 f"en {d['currency']}).")
    if d.get("financial_institution"):
        f.append("Entidad financiera: se omiten margen bruto, EBITDA, free cash flow, deuda neta, "
                 "cobertura de intereses y ROIC, que no aplican a bancos y aseguradoras.")
    neg = [y for y, v in zip(d["fiscal_years"], annual["fcf"]) if _clean_number(v) is not None and v < 0]
    if neg:
        f.append(f"Free cash flow NEGATIVO en {', '.join(neg)}; por eso no se calcula CAGR de FCF "
                 "y el promedio de FCF / utilidad neta queda distorsionado.")
    elif s["fcf_cagr_pct"] is not None:
        f.append(f"Free cash flow positivo todos los años; CAGR {s['fcf_cagr_pct']:.1f}%.")
    conv = _clean_number(annual["fcf_to_net_income"].iloc[-1])
    if conv is not None:
        f.append(f"FCF / utilidad neta en {y1}: {conv:.1f}% "
                 f"({'convierte más que su utilidad en caja' if conv >= 100 else 'convierte en caja menos que su utilidad'}).")
    cx0, cx1 = _r(annual["capex_to_revenue"].iloc[0], 1), _r(annual["capex_to_revenue"].iloc[-1], 1)
    if cx0 is not None and cx1 is not None:
        rel = ("más rápido que las ventas" if cx1 - cx0 > 0.3 else
               "más lento que las ventas" if cx0 - cx1 > 0.3 else "en línea con las ventas")
        f.append(f"Capex / ventas: {cx0:.1f}% en {y0} vs {cx1:.1f}% en {y1}: el capex creció {rel}.")
    cov = [_clean_number(v) for v in annual["interest_coverage"]]
    if cov[-1] is not None and cov[0] is not None:
        f.append(f"Cobertura de intereses {_trend_word(cov[0], cov[-1], 1)}: {cov[0]:.1f}x en {y0} vs "
                 f"{cov[-1]:.1f}x en {y1}.")
    nd = [_clean_number(v) for v in annual["net_debt_to_ebitda"]]
    if nd[-1] is not None and nd[0] is not None:
        f.append(f"Deuda neta / EBITDA (FY) {_trend_word(nd[0], nd[-1], 0.2)}: {nd[0]:.2f}x en {y0} "
                 f"vs {nd[-1]:.2f}x en {y1}.")
    dil = s["diluted_shares_change_pct"]
    if dil is not None:
        f.append(f"Acciones diluidas {dil:+.1f}% entre {y0} y {y1} "
                 f"({'recompras netas' if dil < -1 else 'dilución' if dil > 1 else 'sin cambios relevantes'}).")
    divs = _clean_number(annual["dividends_paid"].iloc[-1]) or 0
    bb = _clean_number(annual["buybacks"].iloc[-1]) or 0
    cur = d["currency"]
    f.append(f"En {y1} pagó dividendos por {_money(divs, cur)} y recompró acciones por "
             f"{_money(bb, cur)}.")
    for col, label in (("revenue", "Ventas"), ("net_income", "Utilidad neta"), ("fcf", "Free cash flow"),
                       ("net_debt", "Deuda neta")):
        v = _clean_number(annual[col].iloc[-1])
        if v is not None:
            f.append(f"{label} {y1}: {_money(v, cur)}.")
    return f


def _benchmark_stats(ticker: str, bench_ticker: str, close: pd.Series, start: pd.Timestamp,
                     fetch_period: str, ctx: AgentContext) -> dict[str, Any]:
    """Beta y correlación diarias a 1 año y desempeño relativo en la ventana visible."""
    if bench_ticker == ticker:
        return {"benchmark": bench_ticker, "note": "el activo es el propio benchmark"}
    try:
        bclose = fetch_history(bench_ticker, fetch_period)["Close"]
    except Exception:
        return {"benchmark": bench_ticker, "error": "sin datos del benchmark"}
    joined = pd.concat([close.rename("a"), bclose.rename("b")], axis=1).dropna()
    lr = np.log(joined / joined.shift(1)).dropna().tail(TRADING_DAYS)
    beta = lr["a"].cov(lr["b"]) / lr["b"].var() if len(lr) > 30 else None
    corr = lr["a"].corr(lr["b"]) if len(lr) > 30 else None
    out: dict[str, Any] = {"benchmark": bench_ticker, "beta_1y_daily": _r(beta),
                           "correlation_1y_daily": _r(corr)}
    bwin = joined[joined.index >= start]
    if len(bwin) > 1:
        ra = bwin["a"].iloc[-1] / bwin["a"].iloc[0] - 1
        rb = bwin["b"].iloc[-1] / bwin["b"].iloc[0] - 1
        ctx.benchmark_df = (bwin / bwin.iloc[0] * 100).rename(columns={"a": ticker, "b": bench_ticker})
        out["asset_return_pct"] = _r(ra * 100)
        out["benchmark_return_pct"] = _r(rb * 100)
        out["relative_performance_pp"] = _r((ra - rb) * 100)
    return out


def _levels(full: pd.DataFrame, price: float, sma50: float | None, sma200: float | None) -> tuple[list, list]:
    """Soportes y resistencias con su origen, sin niveles casi duplicados."""
    hi = full["High"] if "High" in full else full["Close"]
    lo = full["Low"] if "Low" in full else full["Close"]
    cands = [
        (float(lo.tail(21).min()), "mínimo 1 mes (intradiario)"),
        (float(hi.tail(21).max()), "máximo 1 mes (intradiario)"),
        (float(lo.tail(63).min()), "mínimo 3 meses (intradiario)"),
        (float(hi.tail(63).max()), "máximo 3 meses (intradiario)"),
        (float(lo.tail(TRADING_DAYS).min()), "mínimo 52 semanas (intradiario)"),
        (float(hi.tail(TRADING_DAYS).max()), "máximo 52 semanas (intradiario)"),
    ]
    if sma50:
        cands.append((sma50, "SMA 50"))
    if sma200:
        cands.append((sma200, "SMA 200"))

    def pick(items, reverse):
        out: list[dict[str, Any]] = []
        for lvl, src in sorted(items, key=lambda x: x[0], reverse=reverse):
            if out and abs(lvl / out[-1]["level"] - 1) < 0.005:
                out[-1]["source"] += f" / {src}"
                continue
            out.append({"level": round(lvl, 2), "distance_pct": round((lvl / price - 1) * 100, 2),
                        "source": src})
        return out[:4]

    supports = pick([c for c in cands if c[0] < price * 0.999], reverse=True)
    resistances = pick([c for c in cands if c[0] > price * 1.001], reverse=False)
    return supports, resistances


@tool(
    "calculate_technical_indicators",
    "Análisis técnico y de riesgo para la temporalidad indicada: SMA 20/50/200, último cruce "
    "SMA50/SMA200 (golden o death cross) y cruces del precio con cada media, RSI(14), "
    "MACD(12,26,9), Bandas de Bollinger, ATR, retornos 1M/3M/6M/YTD/1Y, volatilidad, máximo "
    "drawdown, Sharpe y Sortino, beta y correlación diarias de 1 año vs benchmark (SPY o "
    "Merval), fuerza relativa, soportes y resistencias con su origen, y volumen relativo. "
    "Incluye 'facts' con las lecturas técnicas ya calculadas.",
    {"ticker": TICKER_PROP,
     "period": {"type": "string", "enum": list(PERIOD_DAYS), "description": "Temporalidad"}},
)
def calculate_technical_indicators(ticker: str, period: str, ctx: AgentContext) -> dict[str, Any]:
    ticker = normalize_ticker(ticker)
    # Siempre al menos 1 año para retornos 1Y, beta y niveles de 52 semanas.
    fetch_period = "5Y" if period == "5Y" else "1Y"
    full = add_indicators(fetch_history(ticker, fetch_period))
    win = visible_window(full, period)
    if len(win) < 5:
        raise DataSourceError("Muy pocas ruedas para calcular indicadores.")
    ctx.chart_df = win

    last = win.iloc[-1]
    close_full = full["Close"]
    price = float(last["Close"])
    sma20, sma50, sma200, rsi = (_clean_number(last[c]) for c in ("SMA20", "SMA50", "SMA200", "RSI14"))

    # Cruces sobre toda la historia descargada, no solo la ventana visible: si
    # no, con 1M o 6M el dato quedaba vacío y el LLM lo inventaba.
    ma_cross = _last_cross(full["SMA50"], full["SMA200"])
    if ma_cross and ma_cross.get("date"):
        ma_cross["type"] = "golden_cross" if ma_cross["direction"] == "arriba" else "death_cross"
    price_x50 = _last_cross(full["Close"], full["SMA50"])
    price_x200 = _last_cross(full["Close"], full["SMA200"])

    rets = win["LogRet"].dropna()
    vol_period = float(rets.std() * math.sqrt(TRADING_DAYS) * 100) if len(rets) > 1 else None
    drawdown_series = win["Close"] / win["Close"].cummax() - 1
    drawdown = float(drawdown_series.min() * 100)
    current_dd = float(drawdown_series.iloc[-1] * 100)

    # Sharpe / Sortino anualizados sobre la ventana visible.
    currency = (ctx.info or fetch_info(ticker)).get("currency")
    rf = fetch_risk_free() if currency in (None, "USD") else None
    ann_ret = float(rets.mean() * TRADING_DAYS) if len(rets) > 1 else None
    ann_vol = float(rets.std() * math.sqrt(TRADING_DAYS)) if len(rets) > 1 else None
    rf_dec = (rf or 0) / 100
    sharpe = (ann_ret - rf_dec) / ann_vol if ann_ret is not None and ann_vol else None
    downside = rets[rets < 0].std() * math.sqrt(TRADING_DAYS) if (rets < 0).sum() > 1 else None
    sortino = (ann_ret - rf_dec) / downside if ann_ret is not None and downside else None

    bench = _benchmark_stats(ticker, benchmark_for(ticker, currency), close_full,
                             win.index[0], fetch_period, ctx)

    # MACD: estado, signo de la línea y si el histograma se expande o contrae.
    macd, macd_sig, macd_hist = (_clean_number(last[c]) for c in ("MACD", "MACDSignal", "MACDHist"))
    hist_tail = win["MACDHist"].dropna().tail(4)
    hist_dir = None
    if len(hist_tail) == 4:
        hist_dir = "expandiéndose" if abs(hist_tail.iloc[-1]) > abs(hist_tail.iloc[0]) else "contrayéndose"
    macd_cross = _last_cross(win["MACD"], win["MACDSignal"])

    bb_up, bb_lo = _clean_number(last["BBUpper"]), _clean_number(last["BBLower"])
    pct_b = (price - bb_lo) / (bb_up - bb_lo) if bb_up and bb_lo and bb_up != bb_lo else None
    if pct_b is None:
        bb_pos = None
    elif pct_b > 1:
        bb_pos = "por encima de la banda superior"
    elif pct_b >= 0.8:
        bb_pos = "cerca de la banda superior"
    elif pct_b > 0.5:
        bb_pos = "en la mitad superior de las bandas"
    elif pct_b > 0.2:
        bb_pos = "en la mitad inferior de las bandas"
    elif pct_b >= 0:
        bb_pos = "cerca de la banda inferior"
    else:
        bb_pos = "por debajo de la banda inferior"
    atr = _clean_number(last.get("ATR14"))

    supports, resistances = _levels(full, price, sma50, sma200)
    vol20 = full["Volume"].tail(20).mean() if "Volume" in full else None
    vol90 = full["Volume"].tail(90).mean() if "Volume" in full else None

    if rsi is None:
        rsi_zone = None
    elif rsi >= 70:
        rsi_zone = "sobrecompra"
    elif rsi <= 30:
        rsi_zone = "sobreventa"
    else:
        rsi_zone = "neutral"
    rsi_5d = _clean_number(win["RSI14"].iloc[-6]) if len(win) > 6 else None

    # YTD contra el último cierre del año anterior (no contra el primer día del año).
    prev_year = close_full[close_full.index.year < close_full.index[-1].year]
    ytd = _r((close_full.iloc[-1] / prev_year.iloc[-1] - 1) * 100) if not prev_year.empty else None
    high_52w = float(close_full.tail(TRADING_DAYS).max())
    low_52w = float(close_full.tail(TRADING_DAYS).min())

    data = {
        "ticker": ticker,
        "period": period,
        "window_start": win.index[0].strftime("%Y-%m-%d"),
        "observations": int(len(win)),
        "price": round(price, 4),
        "sma20": _r(sma20, 4), "sma50": _r(sma50, 4), "sma200": _r(sma200, 4),
        "price_vs_sma20_pct": round((price / sma20 - 1) * 100, 2) if sma20 else None,
        "price_vs_sma50_pct": round((price / sma50 - 1) * 100, 2) if sma50 else None,
        "price_vs_sma200_pct": round((price / sma200 - 1) * 100, 2) if sma200 else None,
        "trend": (None if not (sma50 and sma200) else
                  "alcista" if price > sma50 > sma200 else
                  "bajista" if price < sma50 < sma200 else "mixta"),
        "sma50_vs_sma200": None if not (sma50 and sma200) else
        ("SMA50 por encima de SMA200" if sma50 > sma200 else "SMA50 por debajo de SMA200"),
        "last_sma50_sma200_cross": ma_cross,
        "last_price_cross_sma50": price_x50,
        "last_price_cross_sma200": price_x200,
        "rsi14": round(rsi, 2) if rsi is not None else None,
        "rsi14_5_sessions_ago": _r(rsi_5d),
        "rsi_zone": rsi_zone,
        "macd": _r(macd, 4), "macd_signal": _r(macd_sig, 4), "macd_histogram": _r(macd_hist, 4),
        "macd_state": None if macd is None or macd_sig is None else
        ("alcista (MACD sobre señal)" if macd > macd_sig else "bajista (MACD bajo señal)"),
        "macd_line_vs_zero": None if macd is None else ("positiva" if macd > 0 else "negativa"),
        "macd_histogram_trend_3_sessions": hist_dir,
        "last_macd_signal_cross": macd_cross,
        "bollinger_upper": _r(bb_up, 4), "bollinger_lower": _r(bb_lo, 4),
        "bollinger_pct_b": _r(pct_b, 2), "bollinger_position": bb_pos,
        "atr14": _r(atr, 4), "atr14_pct_price": _r(atr / price * 100 if atr else None),
        "returns_pct": {
            "1M": _return_since(close_full, 30), "3M": _return_since(close_full, 91),
            "6M": _return_since(close_full, 182), "1Y": _return_since(close_full, 365),
            "YTD": ytd,
        },
        "return_window_pct": round((price / float(win["Close"].iloc[0]) - 1) * 100, 2),
        "volatility_window_annual_pct": round(vol_period, 2) if vol_period else None,
        "volatility_20d_annual_pct": round(float(last["Vol20"]) * 100, 2)
        if _clean_number(last["Vol20"]) else None,
        "max_drawdown_window_pct": round(drawdown, 2),
        "drawdown_from_window_peak_pct": round(current_dd, 2),
        "sharpe_window": _r(sharpe), "sortino_window": _r(sortino),
        "risk_free_used_pct": rf,
        "vs_benchmark": bench,
        "high_52w_close": round(high_52w, 4), "low_52w_close": round(low_52w, 4),
        "distance_to_52w_high_pct": _r((price / high_52w - 1) * 100),
        "distance_to_52w_low_pct": _r((price / low_52w - 1) * 100),
        "supports": supports,
        "resistances": resistances,
        "volume_20d_vs_90d": _r(_div(vol20, vol90)),
    }
    if len(rets) < MIN_OBS_RISK_RATIOS:
        data["risk_ratios_note"] = (f"Sharpe y Sortino calculados con {len(rets)} ruedas: "
                                    "poco confiables, no los uses para concluir.")
    data["facts"] = _technical_facts(data)
    ctx.technicals = data
    return data


def _cross_text(c: dict[str, Any] | None, what: str) -> str | None:
    if not c:
        return None
    if not c.get("date"):
        return (f"{what}: sin cruces en la historia analizada; está "
                f"{'por encima' if c['direction'] == 'arriba' else 'por debajo'} al menos desde "
                f"{c['since_at_least']}.")
    extra = ""
    if c.get("crosses_last_20_sessions", 0) >= 3:
        extra = f" Hubo {c['crosses_last_20_sessions']} cruces en las últimas 20 ruedas (señal sin dirección clara)."
    return (f"{what}: el último cruce fue {'al alza' if c['direction'] == 'arriba' else 'a la baja'} "
            f"el {c['date']} (hace {c['days_ago']} días) y desde entonces no volvió a cruzar.{extra}")


def _technical_facts(t: dict[str, Any]) -> list[str]:
    f: list[str] = []
    p = t["price"]
    pos = []
    for k, label in (("sma20", "SMA20"), ("sma50", "SMA50"), ("sma200", "SMA200")):
        if t.get(k):
            pos.append(f"{'por encima' if p > t[k] else 'por debajo'} de la {label} ({_g(t[k])}, "
                       f"{t[f'price_vs_{k}_pct']:+.2f}%)")
    if pos:
        f.append(f"Precio {_g(p)}: " + "; ".join(pos) + ".")
    if t.get("trend"):
        f.append(f"Tendencia de medias: {t['trend']} (alcista solo si precio > SMA50 > SMA200; "
                 f"bajista si precio < SMA50 < SMA200). {t['sma50_vs_sma200']}.")
    c = t.get("last_sma50_sma200_cross")
    if c and c.get("date"):
        name = "golden cross" if c["direction"] == "arriba" else "death cross"
        f.append(f"Último cruce SMA50/SMA200: {name} el {c['date']} (hace {c['days_ago']} días); "
                 "desde entonces no hubo otro cruce entre esas medias.")
    elif c:
        f.append(_cross_text(c, "Cruce SMA50/SMA200"))
    for key, label in (("last_price_cross_sma50", "Precio vs SMA50"),
                       ("last_price_cross_sma200", "Precio vs SMA200")):
        txt = _cross_text(t.get(key), label)
        if txt:
            f.append(txt)
    if t.get("rsi14") is not None:
        d = t.get("rsi14_5_sessions_ago")
        f.append(f"RSI(14) {_g(t['rsi14'])}, zona {t['rsi_zone']}"
                 + (f" (hace 5 ruedas: {_g(d)})." if d is not None else "."))
    if t.get("macd_state"):
        f.append(f"MACD {t['macd_state']}, línea MACD {t['macd_line_vs_zero']} respecto de cero, "
                 f"histograma {_g(t['macd_histogram'])} "
                 f"{t['macd_histogram_trend_3_sessions'] or ''}".rstrip() + ".")
    if t.get("bollinger_position"):
        f.append(f"Bollinger: %B {_g(t['bollinger_pct_b'])}, precio {t['bollinger_position']} "
                 f"(banda superior {_g(t['bollinger_upper'])}, inferior {_g(t['bollinger_lower'])}). "
                 "%B > 1 significa por encima de la banda superior; entre 0.5 y 1, mitad superior.")
    if t.get("supports"):
        f.append("Soportes: " + "; ".join(f"{_g(s['level'])} ({s['source']}, {s['distance_pct']:+.1f}%)"
                                          for s in t["supports"]) + ".")
    if t.get("resistances"):
        f.append("Resistencias: " + "; ".join(f"{_g(s['level'])} ({s['source']}, {s['distance_pct']:+.1f}%)"
                                              for s in t["resistances"]) + ".")
    f.append(f"En la ventana {t['period']} (desde {t['window_start']}): retorno {t['return_window_pct']:+.2f}%, "
             f"máximo drawdown {t['max_drawdown_window_pct']:.2f}%, precio {abs(t['drawdown_from_window_peak_pct']):.2f}% "
             f"debajo del máximo de la ventana; máximo de 52 semanas (cierre) {_g(t['high_52w_close'])}, "
             f"a {t['distance_to_52w_high_pct']:+.2f}%.")
    b = t.get("vs_benchmark") or {}
    if b.get("relative_performance_pp") is not None:
        f.append(f"Vs {b['benchmark']} en la ventana: activo {b['asset_return_pct']:+.2f}% vs benchmark "
                 f"{b['benchmark_return_pct']:+.2f}% ({b['relative_performance_pp']:+.2f} pp, "
                 f"{'mejor' if b['relative_performance_pp'] > 0 else 'peor'} que el mercado).")
    if b.get("beta_1y_daily") is not None:
        corr = abs(b["correlation_1y_daily"] or 0)
        level = "baja" if corr < 0.3 else "moderada" if corr < 0.6 else "alta"
        f.append(f"Beta diaria 1 año vs {b['benchmark']} {_g(b['beta_1y_daily'])} con correlación "
                 f"{level} ({_g(b['correlation_1y_daily'])}): "
                 + ("la beta casi no explica el riesgo del activo." if level == "baja" else
                    "el mercado explica parte del movimiento, pero no la mayoría." if level == "moderada" else
                    "el activo acompaña de cerca al mercado."))
    if t.get("risk_ratios_note"):
        f.append(t["risk_ratios_note"])
    return f


PEER_METRICS = [
    # clave, etiqueta, unidad, es múltiplo de valuación
    ("pe_trailing", "P/E trailing", "x", True),
    ("pe_forward", "P/E forward", "x", True),
    ("ev_ebitda", "EV/EBITDA (TTM)", "x", True),
    ("price_to_sales", "P/S (TTM)", "x", True),
    ("price_to_book", "P/B", "x", True),
    ("fcf_yield_pct", "FCF yield (TTM)", "%", False),
    ("gross_margin_pct", "Margen bruto (TTM)", "%", False),
    ("operating_margin_pct", "Margen operativo (TTM)", "%", False),
    ("net_margin_pct", "Margen neto (TTM)", "%", False),
    ("roe_pct", "ROE (TTM)", "%", False),
    ("revenue_growth_yoy_pct", "Crecimiento de ventas último trimestre interanual", "%", False),
    ("net_debt_to_ebitda", "Deuda neta / EBITDA", "x", False),
    ("dividend_yield_pct", "Dividend yield", "%", False),
    ("return_52w_pct", "Retorno 52 semanas", "%", False),
]
PEER_TABLE_KEYS = ["ticker", "name", "sector", "industry", "reporting_currency", "market_cap"] + \
    [k for k, *_ in PEER_METRICS]


@tool(
    "compare_peers",
    "Compara el activo contra 3 a 6 empresas comparables con el mismo método de cálculo: "
    "P/E trailing y forward, EV/EBITDA, P/S, P/B, FCF yield, márgenes, ROE, crecimiento, "
    "deuda neta/EBITDA, dividend yield y retorno 52 semanas. Devuelve la tabla, la mediana, "
    "la prima o descuento y 'facts' con el ranking de cada métrica ya ordenado. Usá tickers "
    "de Yahoo Finance del mismo negocio (misma industria, no solo el mismo sector).",
    {"ticker": TICKER_PROP,
     "peers": {"type": "string",
               "description": "Tickers comparables separados por coma, ej. 'YPF, PAM, TGS'"}},
)
def compare_peers(ticker: str, peers: str, ctx: AgentContext) -> dict[str, Any]:
    ticker = normalize_ticker(ticker)
    peer_list = parse_peers(peers, exclude=ticker)
    if not peer_list:
        raise ValueError("Indicá al menos un comparable.")

    rows: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {t: pool.submit(fundamental_snapshot, t) for t in [ticker] + peer_list}
        for t, fut in futures.items():
            try:
                snap = fut.result(timeout=60)
                if snap.get("market_cap") is None and snap.get("pe_trailing") is None:
                    raise TickerNotFoundError(t)
                rows[t] = {k: snap.get(k) for k in PEER_TABLE_KEYS}
                rows[t]["currency_mismatch"] = snap.get("currency_mismatch")
            except Exception:
                failed.append(t)
    if ticker not in rows:
        raise DataSourceError(f"No se pudieron obtener datos comparables de {ticker}.")
    target = rows[ticker]
    valid_rows = [rows[t] for t in peer_list if t in rows]
    if not valid_rows:
        raise ValueError(f"Ningún comparable válido en Yahoo Finance: {', '.join(failed)}. "
                         "Probá con otros tickers.")
    # Empresas de otro sector (un banco contra energéticas, un supermercado
    # contra home improvement) quedan en la tabla pero fuera de mediana y ranking.
    tsec = target.get("sector")
    same_sector = [r for r in valid_rows if not tsec or not r.get("sector") or r["sector"] == tsec]
    other_sector = [r for r in valid_rows if r not in same_sector]
    peer_rows = same_sector or valid_rows
    for r in valid_rows:
        r["in_median"] = r in peer_rows
    target["in_median"] = True

    medians, premium, facts = {}, {}, []
    if other_sector and same_sector:
        facts.append(f"Excluidos de la mediana y los rankings por ser de otro sector que {ticker} "
                     f"({tsec}): " + ", ".join(f"{r['ticker']} ({r.get('sector')})" for r in other_sector)
                     + ". Figuran en la tabla solo como referencia.")
    elif other_sector:
        facts.append(f"Ningún comparable es del mismo sector que {ticker} ({tsec}): la comparación es "
                     "solo orientativa.")
    for key, label, unit, is_multiple in PEER_METRICS:
        vals = [r[key] for r in peer_rows if r.get(key) is not None]
        if not vals:
            continue
        med = float(np.median(vals))
        medians[key] = round(med, 2)
        tv = target.get(key)
        if tv is None:
            continue
        group = sorted([(r["ticker"], r[key]) for r in [target] + peer_rows if r.get(key) is not None],
                       key=lambda x: x[1], reverse=True)
        pos = [g[0] for g in group].index(ticker) + 1
        rel = "por encima" if tv > med else "por debajo" if tv < med else "igual"
        text = (f"{label}: de mayor a menor " + ", ".join(f"{t} {_g(v)}{unit}" for t, v in group)
                + f". {ticker} queda {pos}° de {len(group)} y {rel} de la mediana de comparables "
                  f"({_g(med)}{unit})")
        if is_multiple and med > 0:
            premium[key] = round((tv / med - 1) * 100, 1)
            text += f": {'prima' if premium[key] > 0 else 'descuento'} de {abs(premium[key]):.1f}%"
        facts.append(text + ".")

    tind = target.get("industry")
    other = [f"{r['ticker']} ({r.get('industry') or 'industria desconocida'})"
             for r in peer_rows if tind and r.get("industry") != tind]
    if other:
        facts.append(f"Atención: {', '.join(other)} son del mismo sector pero no de la misma industria "
                     f"que {ticker} ({tind}); tomá sus múltiplos con cautela.")
    mismatched = [r["ticker"] for r in [target] + valid_rows if r.get("currency_mismatch")]
    if mismatched:
        facts.append(f"{', '.join(mismatched)} reportan en una moneda distinta a la de cotización: "
                     "se excluyeron sus EV/EBITDA, P/S, P/B, FCF yield, crecimiento y dividend yield.")
    if failed:
        facts.append(f"Sin datos en Yahoo Finance para {', '.join(failed)} (ticker inexistente o "
                     "sin cobertura); no forman parte de la comparación.")

    table = [{k: v for k, v in r.items() if k != "currency_mismatch"} for r in [target] + valid_rows]
    data = {
        "ticker": ticker,
        "peers": [r["ticker"] for r in peer_rows],
        "excluded_other_sector": [r["ticker"] for r in other_sector if r not in peer_rows] or None,
        "failed_tickers": failed or None,
        "table": table,
        "peer_median": medians,
        "premium_vs_median_pct": premium,
        "facts": facts,
    }
    if len(peer_rows) < 3 or not same_sector:
        data["action_required"] = (
            f"Solo {len(same_sector)} comparables válidos del sector {tsec} (industria {tind}). "
            "Volvé a llamar compare_peers con 3 a 6 tickers de ese sector, idealmente de la misma "
            "industria, manteniendo los válidos.")
    ctx.peers = data
    return data


# Léxico mínimo para un pantallazo de sentimiento (titulares en inglés y español).
POSITIVE = {"beat", "beats", "surge", "surges", "soar", "soars", "jump", "jumps", "gain",
            "gains", "rally", "record", "upgrade", "upgraded", "upgrades", "outperform", "strong",
            "growth", "profit", "bullish", "buy", "raises", "raised", "tops", "rise",
            "rises", "boost", "expands", "approval", "wins", "higher", "positive", "climbs",
            "sube", "suben", "subió", "récord", "crece", "crecen", "creció", "ganancia",
            "ganancias", "mejora", "mejoró", "alza", "dispara", "supera", "superó", "fuerte",
            "aprobación", "expansión", "acuerdo", "recomienda", "compra", "alcista"}
NEGATIVE = {"miss", "misses", "fall", "falls", "drop", "drops", "plunge", "plunges",
            "slump", "downgrade", "downgraded", "downgrades", "underperform", "weak", "loss", "losses",
            "bearish", "sell", "cuts", "cut", "lawsuit", "probe", "decline", "declines",
            "lower", "warning", "risk", "crisis", "default", "tumble", "tumbles", "negative",
            "cae", "caen", "cayó", "baja", "bajan", "pérdida", "pérdidas", "desploma",
            "derrumbe", "riesgo", "demanda", "caída", "recorte", "rebaja", "investigación",
            "multa", "advertencia", "débil", "venta", "bajista"}


def _headline_score(text: str) -> int:
    words = {w.strip(".,:;!?\"'()¿¡").lower() for w in text.split()}
    return len(words & POSITIVE) - len(words & NEGATIVE)


@tool(
    "get_company_news_sentiment",
    "Titulares de los últimos 30 días de Yahoo Finance y Google News (español e inglés), "
    "deduplicados y ordenados por fecha, con un puntaje de sentimiento léxico orientativo. "
    "Leé los titulares para identificar catalizadores, riesgos y eventos corporativos.",
    {"ticker": TICKER_PROP},
)
def get_company_news_sentiment(ticker: str, ctx: AgentContext) -> dict[str, Any]:
    ticker = normalize_ticker(ticker)
    info = ctx.info or fetch_info(ticker)
    name = info.get("longName") or info.get("shortName") or ticker
    raw, errors = fetch_news(ticker, name)

    headlines = []
    for item in raw[:MAX_HEADLINES]:
        headlines.append({
            "title": item["title"],
            "publisher": item.get("publisher"),
            "url": item.get("url"),
            "date": item["date"].strftime("%Y-%m-%d") if item.get("date") else "",
            "source": item.get("source"),
            "lang": item.get("lang"),
            "score": _headline_score(item["title"]),
        })

    scores = [h["score"] for h in headlines]
    avg = float(np.mean(scores)) if scores else None
    label = ("sin datos" if avg is None else
             "positivo" if avg > 0.25 else "negativo" if avg < -0.25 else "neutral")
    data = {
        "ticker": ticker,
        "query_name": company_query_name(name),
        "count": len(headlines),
        "positive": sum(s > 0 for s in scores),
        "negative": sum(s < 0 for s in scores),
        "neutral": sum(s == 0 for s in scores),
        "sentiment_score": round(avg, 2) if avg is not None else None,
        "sentiment_label": label,
        "sentiment_note": "Puntaje léxico por palabras clave: es orientativo. Algunos titulares "
                          "pueden ser de otras empresas que mencionan a esta.",
        "headlines": headlines,
    }
    if errors:
        data["source_errors"] = errors
    if not headlines and errors:
        data["error"] = "No se pudieron obtener noticias: " + "; ".join(errors)
    ctx.news = data
    return data


# ---------------------------------------------------------------------------
# Verificación de cifras del informe contra los datos
# ---------------------------------------------------------------------------

INDICATOR_PARAMS = {12, 14, 20, 26, 30, 50, 52, 70, 100, 200}
_DATE_RE = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\bFY\d{4}\b")
_NUM_RE = re.compile(r"(?<![A-Za-z0-9.,])[-+−]?\d[\d.,]*(?:x)?(?![A-Za-z0-9])")


def _parse_report_number(token: str) -> tuple[list[float], int]:
    """'1,188.5' -> ([1188.5], 1). Devuelve candidatos (por ambigüedad) y decimales."""
    s = token.replace("−", "-").rstrip("x").rstrip(".,")
    sign = -1 if s.startswith("-") else 1
    s = s.lstrip("+-")
    cands: list[tuple[float, int]] = []
    if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", s):
        v = s.replace(",", "")
        cands.append((float(v), len(v.split(".")[1]) if "." in v else 0))
    elif re.fullmatch(r"\d{1,3}(\.\d{3})+(,\d+)?", s):  # 1.188,5 (estilo español)
        v = s.replace(".", "").replace(",", ".")
        cands.append((float(v), len(v.split(".")[1]) if "." in v else 0))
        cands.append((float(s.replace(",", "")), len(s.split(".")[-1])))
    elif re.fullmatch(r"\d+([.,]\d+)?", s):
        v = s.replace(",", ".")
        cands.append((float(v), len(v.split(".")[1]) if "." in v else 0))
    if not cands:
        return [], 0
    return [sign * c[0] for c in cands], max(c[1] for c in cands)


def _numbers_in(obj: Any, plain: list[float], amounts: list[float], key: str = "") -> None:
    """Junta los números de los datos.

    plain: valores tal como figuran. amounts: montos absolutos en unidades
    (los campos "_m" vienen en millones y se multiplican por 1e6).
    """
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        v = _clean_number(obj)
        if v is not None:
            plain.append(v)
            if key.endswith("_m"):
                amounts.append(v * 1e6)
            elif abs(v) >= 1e5:
                amounts.append(v)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _numbers_in(v, plain, amounts, str(k))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numbers_in(v, plain, amounts, key)
    elif isinstance(obj, str):
        clean = _DATE_RE.sub(" ", obj)
        for m in _NUM_RE.finditer(clean):
            vals, _ = _parse_report_number(m.group())
            unit = _unit_after(clean, m.end())
            plain.extend(vals)
            if unit:
                amounts.extend(v * u for v in vals for u in unit)


# Unidad escrita después de una cifra ("7.5 mil millones", "329,134 millones").
_UNIT_RE = re.compile(r"\s*(mil millones|miles de millones|billones|millones|mill\.|MM|bn|B|M)\b")
_UNIT_FACTORS = {"mil millones": (1e9,), "miles de millones": (1e9,), "bn": (1e9,), "B": (1e9,),
                 "billones": (1e12, 1e9), "millones": (1e6,), "mill.": (1e6,), "MM": (1e6,), "M": (1e6,)}


def _unit_after(text: str, pos: int) -> tuple[float, ...] | None:
    m = _UNIT_RE.match(text, pos)
    return _UNIT_FACTORS.get(m.group(1)) if m else None


def _collect_data_numbers(ctx: AgentContext, extra: Any = None) -> tuple[np.ndarray, np.ndarray]:
    plain: list[float] = []
    amounts: list[float] = []
    for part in (ctx.market, ctx.financials, ctx.technicals, ctx.peers, extra):
        _numbers_in(part or {}, plain, amounts)
    if ctx.news:
        _numbers_in({k: v for k, v in ctx.news.items() if k != "headlines"}, plain, amounts)
    return np.abs(np.array(plain or [0.0])), np.abs(np.array(amounts or [0.0]))


def find_unsupported_numbers(text: str, ctx: AgentContext, extra: Any = None) -> list[str]:
    """Cifras del informe que no coinciden con ningún dato de las herramientas.

    Tolera redondeos según los decimales escritos. Si la cifra lleva unidad
    ("millones", "mil millones"), se compara el monto absoluto: así se detecta
    citar 329.134 millones como "329.1 millones". Un número marcado no es
    necesariamente un error (puede ser un cálculo válido), pero obliga a revisarlo.
    extra: datos adicionales válidos como respaldo (por ejemplo, una simulación).
    """
    plain, amounts = _collect_data_numbers(ctx, extra)
    flagged: list[str] = []
    clean = _DATE_RE.sub(" ", text)
    for m in _NUM_RE.finditer(clean):
        values, decimals = _parse_report_number(m.group())
        if not values:
            continue
        v0 = abs(values[0])
        unit = _unit_after(clean, m.end())
        if decimals == 0 and not unit and (v0 < 10 or 1900 <= v0 <= 2100 or v0 in INDICATOR_PARAMS):
            continue  # enteros chicos (conteos, plazos), años y parámetros (SMA 20, RSI 14)
        # Tolerancia de redondeo según los decimales escritos (22.3 cubre 22.25 a 22.35).
        tol = 0.5 * 10 ** (-decimals) + 1e-9
        if unit:
            ok = any(np.any(np.abs(amounts - abs(v) * u) <= tol * u) for v in values for u in unit)
        else:
            ok = any(np.any(np.abs(plain - abs(v)) <= tol) for v in values)
        if not ok:
            s, e = max(0, m.start() - 45), min(len(clean), m.end() + 25)
            flagged.append(f"{m.group()} en: …{' '.join(clean[s:e].split())}…")
    return flagged


def _diff_changes(before: str, after: str, limit: int = 15) -> list[str]:
    """Cambios reales entre borrador e informe final, línea por línea.

    No se confía en la lista que declara el LLM revisor: a veces dice haber
    cambiado algo que no cambió.
    """
    import difflib

    a, b = before.splitlines(), after.splitlines()
    out: list[str] = []

    def short(lines: list[str]) -> str:
        txt = " ".join(x.strip() for x in lines if x.strip())
        return txt if len(txt) <= 220 else txt[:217] + "..."

    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        old, new = short(a[i1:i2]), short(b[j1:j2])
        if not old and not new:
            continue
        out.append(f"Antes: {old}  |  Después: {new}" if op == "replace" else
                   f"Eliminado: {old}" if op == "delete" else f"Agregado: {new}")
    return out[:limit]


_NEGATION = re.compile(r"(no|sin|lejos de|fuera de|ni)\s+(\w+\s+){0,3}$", re.IGNORECASE)


def find_contradictions(text: str, ctx: AgentContext) -> list[str]:
    """Afirmaciones técnicas del informe que contradicen los datos.

    Solo cubre señales con una única lectura correcta (tendencia de medias,
    tipo del último cruce, zona de RSI, estado del MACD, posición en Bollinger).
    """
    t = ctx.technicals or {}
    out: list[str] = []
    low = text.lower()

    def negated(pos: int) -> bool:
        return bool(_NEGATION.search(low[max(0, pos - 40):pos]))

    trend = t.get("trend")
    if trend:
        for m in re.finditer(r"tendencia de (?:las )?medias(?: móviles)?[^.\n|]{0,30}?\b(alcista|bajista|mixta)", low):
            if m.group(1) != trend and not negated(m.start()):
                out.append(f"Dice 'tendencia de medias {m.group(1)}' pero los datos dicen '{trend}' "
                           f"(precio {t.get('price')}, SMA50 {t.get('sma50')}, SMA200 {t.get('sma200')}).")
    cross = t.get("last_sma50_sma200_cross") or {}
    if cross.get("date"):
        actual = "golden" if cross["direction"] == "arriba" else "death"
        for m in re.finditer(r"(golden|death) cross", low):
            ctx_win = low[max(0, m.start() - 80):m.end() + 40]
            if m.group(1) != actual and ("últim" in ctx_win or "reciente" in ctx_win):
                out.append(f"Dice que el último cruce es '{m.group(1)} cross' pero fue {actual} cross "
                           f"el {cross['date']}.")
    rsi = t.get("rsi14")
    if rsi is not None:
        for word, wrong in (("sobrecompra", rsi < 65), ("sobreventa", rsi > 35)):
            for m in re.finditer(word, low):
                if wrong and not negated(m.start()) and "rsi" in low[max(0, m.start() - 60):m.end() + 30]:
                    out.append(f"Dice RSI en '{word}' pero el RSI es {rsi} (zona {t.get('rsi_zone')}).")
                    break
    state = (t.get("macd_state") or "").split(" ")[0]
    if state:
        for m in re.finditer(r"macd\s*(?:\(12, ?26, ?9\)\s*)?(?:es |está |en |con sesgo |muestra (?:una )?(?:señal )?)?(alcista|bajista)", low):
            if m.group(1) != state and not negated(m.start()):
                out.append(f"Dice MACD '{m.group(1)}' pero el estado es '{t['macd_state']}'.")
    pct_b = t.get("bollinger_pct_b")
    if pct_b is not None:
        checks = ((r"por encima de la banda superior", pct_b <= 1),
                  (r"(en|sobre|toca|tocando) (la )?banda superior", pct_b < 0.8),
                  (r"por debajo de la banda inferior", pct_b >= 0),
                  (r"(en|sobre|toca|tocando) (la )?banda inferior", pct_b > 0.2))
        for pattern, wrong in checks:
            m = re.search(pattern, low)
            if m and wrong and not negated(m.start()):
                out.append(f"Dice '{m.group(0)}' pero %B es {pct_b} ({t.get('bollinger_position')}).")
    return list(dict.fromkeys(out))


def compact_data_for_review(ctx: AgentContext) -> str:
    """JSON con todos los datos de las herramientas, sin campos pesados."""
    market = {k: v for k, v in (ctx.market or {}).items() if k != "summary"}
    news = None
    if ctx.news:
        news = {k: v for k, v in ctx.news.items() if k != "headlines"}
        news["titles"] = [f"{h['date']} | {h['title']}" for h in ctx.news.get("headlines", [])]
    payload = {"get_market_data": market, "get_financial_statements": ctx.financials,
               "calculate_technical_indicators": ctx.technicals, "compare_peers": ctx.peers,
               "get_company_news_sentiment": news}
    return json.dumps(payload, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Agente
# ---------------------------------------------------------------------------

@dataclass
class AgentStep:
    kind: str          # "thought" | "tool_call" | "tool_result" | "error" | "final"
    title: str
    detail: Any = None


@dataclass
class AnalysisResult:
    ticker: str
    period: str
    engine: str
    market: dict[str, Any] | None
    financials: dict[str, Any] | None
    technicals: dict[str, Any] | None
    peers: dict[str, Any] | None
    news: dict[str, Any] | None
    chart_df: pd.DataFrame | None
    benchmark_df: pd.DataFrame | None
    annual_df: pd.DataFrame | None
    quarterly_df: pd.DataFrame | None
    diagnosis: str
    steps: list[AgentStep]
    review: dict[str, Any] | None = None


AGENT_ROLE = "Analista Senior de Equity Research"
AGENT_GOAL = ("Producir un informe de research profesional, fundamental, de valuación y técnico, "
              "sobre {ticker}, útil para que un analista tome decisiones, basado exclusivamente "
              "en datos obtenidos con las herramientas.")
AGENT_BACKSTORY = ("Tenés 15 años de experiencia en equity research buy side en mercados "
                   "desarrollados y emergentes, con foco en Argentina y Latinoamérica. Pensás como "
                   "un portfolio manager: separás calidad del negocio, solidez financiera, valuación "
                   "y timing técnico, y siempre buscás qué está descontando el precio. Sos riguroso, "
                   "contrastás los datos entre sí, señalás inconsistencias y nunca inventás números.")

TASK_DESCRIPTION = """Analizá el activo {ticker} con temporalidad {period}. Fecha de hoy: {today}.

Pasos obligatorios:
1. `get_market_data` con ticker="{ticker}".
2. `get_financial_statements` con ticker="{ticker}" (si es ETF, saltealo).
3. `calculate_technical_indicators` con ticker="{ticker}" y period="{period}".
4. `compare_peers` con ticker="{ticker}" y {peers_instruction}
5. `get_company_news_sentiment` con ticker="{ticker}".
Si una herramienta devuelve un campo "error", indicá qué dato falta y seguí con el resto.

Cómo usar los datos (obligatorio):
* Cada herramienta trae "facts": lecturas ya calculadas y verificadas. Para toda comparación \
(mayor o menor, por encima o por debajo, ranking, prima o descuento), cruce de medias, posición \
en Bollinger, tendencias de márgenes y lecturas de valuación, usá lo que dicen los facts. No \
hagas comparaciones propias entre números: es donde se cometen errores.
* Aclará siempre la base temporal: TTM (últimos 12 meses), FY (año fiscal) o último trimestre. \
No mezcles un margen TTM con uno anual como si fueran el mismo dato.
* Hay dos betas: "beta_5y_monthly" (Yahoo) y "beta_1y_daily" (vs benchmark). Si citás una, \
decí cuál es.
* El precio objetivo del consenso es el MEDIO ("target_mean"). No lo llames mediano.
* Compará contra comparables solo métricas que existan para ellos en la tabla.
* Distinguí "cruce de medias" (SMA50 vs SMA200) de "cruce del precio con una media".

Luego redactá el informe en español, en Markdown, con estas secciones exactas:

### Resumen ejecutivo
Tesis en 3 a 5 viñetas, sesgo (alcista, neutral o bajista), horizonte sugerido y nivel de \
convicción (bajo, medio o alto) justificado.
### Negocio y posicionamiento
Qué hace, de dónde vienen los ingresos, sector y contexto competitivo (usá el perfil y los comparables).
### Calidad, crecimiento y rentabilidad
Tendencias multianuales (FY): CAGR de ventas, EPS y FCF, evolución de márgenes, ROIC y ROE, \
conversión de caja, capex; márgenes TTM y último trimestre interanual. Destacá si la calidad \
mejora o empeora.
### Balance y solidez financiera
Deuda neta, deuda neta/EBITDA, cobertura de intereses, liquidez, dilución o recompras, dividendos.
### Valuación
Múltiplos absolutos, prima o descuento vs mediana de comparables, earnings yield y FCF yield \
contra el bono del Tesoro a 10 años, y consenso de analistas. Explicá qué crecimiento parece \
estar descontando el precio.
### Análisis técnico
Tendencia (medias y cruces), momentum (RSI, MACD), posición en Bollinger, volatilidad y ATR, \
drawdown, Sharpe y Sortino, fuerza relativa vs benchmark, soportes y resistencias con su origen.
### Noticias y catalizadores
Síntesis de los titulares relevantes de la empresa (ignorá los que hablan de otras empresas), \
tu propia lectura del tono (el puntaje léxico es solo orientativo) y próximos eventos, como la \
fecha del próximo balance.
### Riesgos principales
Riesgos específicos de la empresa, sector, balance, valuación y país, apoyados en los datos.
### Escenarios
Tabla Markdown con escenario alcista, base y bajista: qué tiene que pasar y una referencia de \
precio basada en datos disponibles (precios objetivo alto, medio y bajo del consenso o niveles \
técnicos). Indicá de dónde sale cada referencia.
### Conclusión y qué monitorear
Veredicto en 2 o 3 líneas y una lista de métricas y niveles concretos a seguir.

Reglas:
* Citá cifras concretas de las herramientas con su unidad y moneda. No inventes datos.
* Contextualizá cada ratio según el sector y contra los comparables.
* Si un dato no está disponible o no aplica (por ejemplo P/E en un ETF o con pérdidas), decilo.
* Si la empresa reporta en ARS, aclará que el crecimiento nominal está distorsionado por la \
inflación y priorizá márgenes y métricas en USD.
* Señalá contradicciones entre señales (por ejemplo fundamentos sólidos con técnico bajista).
* No uses guiones largos.
* Cerrá con una línea aclarando que no es una recomendación de inversión."""

TASK_EXPECTED_OUTPUT = ("Informe de research en Markdown con las secciones Resumen ejecutivo, "
                        "Negocio y posicionamiento, Calidad crecimiento y rentabilidad, Balance y "
                        "solidez financiera, Valuación, Análisis técnico, Noticias y catalizadores, "
                        "Riesgos principales, Escenarios y Conclusión y qué monitorear.")

REVIEW_SYSTEM = f"""Sos el revisor de calidad de un equipo de equity research. Recibís los DATOS \
(JSON con la salida de todas las herramientas, incluidos sus "facts", que son lecturas ya \
verificadas) y el BORRADOR de un informe. Tu trabajo es que el informe no tenga ningún error.

Verificá una por una:
1. Cada cifra: que exista en los datos o se derive correctamente de ellos, con la unidad, \
moneda y base temporal correctas (TTM, FY, último trimestre, ventana del período).
2. Cada comparación (mayor, menor, por encima, por debajo, prima, descuento, ranking): que \
coincida con los datos y con los facts. Si el borrador dice que A es mayor que B y los números \
dicen lo contrario, corregilo.
3. Fechas y señales técnicas: cruces de medias, posición en Bollinger, estado de MACD y RSI, \
soportes y resistencias, según los facts técnicos.
4. Etiquetas: precio objetivo MEDIO (no mediano), qué beta se cita, si una métrica es TTM o FY.
5. Coherencia interna: que el resumen ejecutivo, las secciones y la conclusión no se \
contradigan entre sí ni con el sesgo declarado.
6. Afirmaciones sin respaldo en los datos (por ejemplo "histograma creciente" si no figura, o \
"supera a comparables" en una métrica que no existe para ellos): eliminarlas o reformularlas.

7. Montos: los campos terminados en "_m" están en millones. 329134.2 en un campo "_m" son \
329134.2 millones = 329.1 mil millones, NO 329.1 millones. Revisá cada monto con su unidad.
8. Métricas que no aplican (por ejemplo margen bruto, EBITDA o FCF en bancos, o datos en null): \
no las cites como si existieran.

Además recibís:
* CONTRADICCIONES DETECTADAS: afirmaciones que un control automático ya comprobó que son falsas. \
Corregirlas es OBLIGATORIO.
* NÚMEROS DEL BORRADOR QUE NO APARECEN EN LOS DATOS: pueden ser cálculos válidos (diferencias, \
porcentajes derivados) o errores, incluidos errores de unidad. Revisá cada uno; si no podés \
justificarlo con los datos, corregilo o eliminalo.

Devolvé SOLO el informe completo corregido, en Markdown, con las mismas secciones y estilo, \
sin comentarios antes ni después. Si algo es correcto, dejalo exactamente igual. Si necesitás \
agregar una nota final, ponela después de una línea con {REVIEW_MARKER} (se descarta)."""


class FinancialAgent:
    """Agente que orquesta las herramientas y sintetiza un informe.

    engine:  "crewai" (CrewAI + OpenAI), "reglas" o "auto" (crewai si hay API key).
    on_step: callback opcional que recibe cada AgentStep en tiempo real
             (la UI lo usa para pintar el panel de razonamiento mientras corre).
    review:  si True (por defecto), el borrador del LLM pasa por un paso de
             verificación contra los datos antes de entregarse.
    """

    def __init__(self, engine: str = "auto", on_step: Callable[[AgentStep], None] | None = None,
                 api_key: str | None = None, model: str | None = None, review: bool = True):
        self.api_key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        self.model = model or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        if engine == "auto":
            engine = "crewai" if self.api_key else "reglas"
        self.engine = engine
        self.review = review
        self.on_step = on_step
        self.steps: list[AgentStep] = []

    # -- logging del razonamiento ----------------------------------------
    def _log(self, kind: str, title: str, detail: Any = None) -> None:
        step = AgentStep(kind, title, detail)
        self.steps.append(step)
        if self.on_step:
            try:
                self.on_step(step)
            except Exception:
                pass  # un error de UI no debe tumbar el análisis

    # -- ejecución de herramientas -----------------------------------------
    def execute_tool(self, name: str, args: dict[str, Any], ctx: AgentContext) -> tuple[dict, bool]:
        """Ejecuta una herramienta. Devuelve (resultado, es_error)."""
        self._log("tool_call", f"Llamando `{name}`", args)
        t = TOOLS.get(name)
        if t is None:
            result, is_error = {"error": f"Herramienta desconocida: {name}"}, True
        else:
            try:
                result, is_error = t.func(**args, ctx=ctx), False
            except Exception as exc:
                result, is_error = {"error": f"{type(exc).__name__}: {exc}"}, True
        heavy = ("headlines", "summary", "table", "recent_rating_changes", "quarterly", "facts")
        preview = {k: v for k, v in result.items() if k not in heavy}
        if result.get("facts"):
            preview["facts"] = result["facts"]
        self._log("error" if is_error else "tool_result",
                  f"{'Error en' if is_error else 'Resultado de'} `{name}`", preview)
        return result, is_error

    # -- punto de entrada ----------------------------------------------------
    def run(self, ticker: str, period: str = "1Y", peers: str | list[str] | None = None) -> AnalysisResult:
        ticker = normalize_ticker(ticker)
        if period not in PERIOD_DAYS:
            raise ValueError(f"Temporalidad inválida: {period}")
        peer_list = parse_peers(peers, exclude=ticker)
        ctx = AgentContext(period=period)
        motor = f"CrewAI + OpenAI ({self.model})" if self.engine == "crewai" else "reglas"
        self._log("thought", f"Objetivo: analizar {ticker} en temporalidad {period} "
                             f"(motor: {motor}).")

        # Validación temprana: si el ticker no existe, cortamos con un error claro
        # antes de gastar créditos de OpenAI.
        self._log("thought", "Valido que el ticker tenga cotizaciones en Yahoo Finance.")
        fetch_history(ticker, period)

        diagnosis = None
        review = None
        engine_used = motor
        if self.engine == "crewai":
            try:
                diagnosis = self._run_crewai(ticker, period, peer_list, ctx)
            except Exception as exc:
                self._log("error", "CrewAI/OpenAI falló, paso al motor de reglas.",
                          f"{type(exc).__name__}: {exc}")
                engine_used = "reglas (fallback)"

        # Garantiza que la UI tenga todos los datos aunque el LLM haya
        # salteado alguna herramienta.
        self._ensure_data(ticker, period, peer_list, ctx)

        if diagnosis and self.review:
            diagnosis, review = self._review_report(diagnosis, ctx)

        if not diagnosis:
            self._log("thought", "Sintetizo el informe con reglas cuantitativas.")
            diagnosis = build_rule_based_diagnosis(ctx)

        self._log("final", "Análisis completo.")
        return AnalysisResult(ticker, period, engine_used, ctx.market, ctx.financials,
                              ctx.technicals, ctx.peers, ctx.news, ctx.chart_df,
                              ctx.benchmark_df, ctx.annual_df, ctx.quarterly_df,
                              diagnosis, self.steps, review)

    def _ensure_data(self, ticker: str, period: str, peer_list: list[str], ctx: AgentContext) -> None:
        plan = [
            ("get_market_data", {"ticker": ticker}, ctx.market),
            ("get_financial_statements", {"ticker": ticker}, ctx.financials),
            ("calculate_technical_indicators", {"ticker": ticker, "period": period}, ctx.technicals),
            ("get_company_news_sentiment", {"ticker": ticker}, ctx.news),
        ]
        if peer_list:
            plan.insert(3, ("compare_peers", {"ticker": ticker, "peers": ", ".join(peer_list)},
                            ctx.peers))
        pending = [(n, a) for n, a, done in plan if done is None]
        if pending and self.engine == "reglas":
            self._log("thought", "Plan: datos de mercado, estados contables, técnico"
                                 f"{', comparables' if peer_list else ''} y noticias.")
        for name, args in pending:
            self.execute_tool(name, args, ctx)

    # -- motor CrewAI + OpenAI ---------------------------------------------
    def _llm(self, temperature: float = 0.2):
        from crewai import LLM
        return LLM(model=self.model, api_key=self.api_key, timeout=180, max_retries=2,
                   temperature=temperature)

    def _build_crewai_tools(self, ctx: AgentContext) -> list:
        """Envuelve las herramientas del registro como tools de CrewAI.

        Se crean por ejecución (closures) para que cada tool escriba en el
        AgentContext de este análisis y quede registrada en el panel de
        razonamiento de la UI.
        """
        from crewai.tools import tool as crew_tool

        def call(name: str, **kwargs: Any) -> str:
            result, _ = self.execute_tool(name, kwargs, ctx)
            if result.get("headlines"):
                # Las URLs de Google News son largas y no le aportan nada al LLM.
                result = {**result, "headlines": [{k: v for k, v in h.items() if k != "url"}
                                                  for h in result["headlines"]]}
            return json.dumps(result, ensure_ascii=False, default=str)

        @crew_tool("get_market_data")
        def market_data_tool(ticker: str) -> str:
            """Perfil, cotización, múltiplos de valuación (P/E, PEG, EV/EBITDA, P/S, P/B,
            earnings y FCF yield vs Treasury 10 años), márgenes TTM, ROE, balance, consenso
            de analistas, tenencia, short interest, betas y próximo balance, con 'facts'
            ya calculados. Parámetro: ticker de Yahoo Finance."""
            return call("get_market_data", ticker=ticker)

        @crew_tool("get_financial_statements")
        def statements_tool(ticker: str) -> str:
            """Estados contables por año fiscal y últimos trimestres: ventas, márgenes,
            EBITDA, utilidad, EPS, flujo operativo, capex, FCF, ROIC, ROE, deuda neta/EBITDA,
            cobertura de intereses, recompras, dividendos y CAGR, con 'facts' de tendencias.
            Parámetro: ticker."""
            return call("get_financial_statements", ticker=ticker)

        @crew_tool("calculate_technical_indicators")
        def technicals_tool(ticker: str, period: str) -> str:
            """SMA 20/50/200 y cruces, RSI, MACD, Bollinger, ATR, retornos, volatilidad,
            drawdown, Sharpe, Sortino, beta diaria y fuerza relativa vs benchmark, soportes
            y resistencias con su origen, con 'facts' técnicos ya calculados. Parámetros:
            ticker y period (1M, 6M, 1Y o 5Y)."""
            return call("calculate_technical_indicators", ticker=ticker, period=period)

        @crew_tool("compare_peers")
        def peers_tool(ticker: str, peers: str) -> str:
            """Compara múltiplos, márgenes, ROE, crecimiento y retorno contra comparables
            con el mismo método, y devuelve mediana, prima o descuento y 'facts' con el
            ranking de cada métrica. Parámetros: ticker y peers (3 a 6 tickers de Yahoo
            Finance de la misma industria, separados por coma, ej. 'YPF, PAM, TGS')."""
            return call("compare_peers", ticker=ticker, peers=peers)

        @crew_tool("get_company_news_sentiment")
        def news_tool(ticker: str) -> str:
            """Titulares de los últimos 30 días de Yahoo Finance y Google News (español e
            inglés) con un sentimiento léxico orientativo. Parámetro: ticker."""
            return call("get_company_news_sentiment", ticker=ticker)

        return [market_data_tool, statements_tool, technicals_tool, peers_tool, news_tool]

    def _on_crew_step(self, step: Any) -> None:
        """step_callback de CrewAI: expone el 'thought' del agente en la UI."""
        thought = getattr(step, "thought", None)
        text = str(thought or "").strip()
        if text.lower().startswith("thought:"):
            text = text[len("thought:"):].strip()
        if text:
            self._log("thought", text[:800])

    def _run_crewai(self, ticker: str, period: str, peer_list: list[str], ctx: AgentContext) -> str:
        if not self.api_key:
            raise RuntimeError("Falta OPENAI_API_KEY.")
        # Import diferido: el motor de reglas no necesita CrewAI.
        from crewai import Agent, Crew, Process, Task

        if peer_list:
            peers_instruction = f'peers="{", ".join(peer_list)}" (elegidos por el usuario).'
        else:
            peers_instruction = (
                "3 a 6 comparables que elijas vos según el sector e industria que devuelve "
                "get_market_data: misma industria (no solo el mismo sector), tamaño y región "
                "similares, en formato de Yahoo Finance separados por coma. Los comparables de "
                "otro sector se excluyen de la mediana. Referencias de ADRs argentinos en NYSE por "
                "industria: bancos BMA, SUPV, BBAR (y regionales como ITUB, BSBR, CIB, BAP); "
                "petróleo y gas YPF, VIST, PBR, EC; utilities CEPU, EDN, PAM; transporte de gas "
                "TGS; telecom TEO; real estate IRS, CRESY; materiales LOMA. Si la respuesta trae "
                "action_required o failed_tickers, volvé a llamar compare_peers con reemplazos.")

        self._log("thought", "Armo el Crew: 1 agente analista con 5 herramientas y 1 tarea.")
        analyst = Agent(
            role=AGENT_ROLE,
            goal=AGENT_GOAL.format(ticker=ticker),
            backstory=AGENT_BACKSTORY,
            tools=self._build_crewai_tools(ctx),
            llm=self._llm(),
            allow_delegation=False,
            max_iter=MAX_AGENT_ITERATIONS,
            max_execution_time=420,
            step_callback=self._on_crew_step,
            verbose=False,
        )
        task = Task(
            description=TASK_DESCRIPTION.format(ticker=ticker, period=period,
                                                today=f"{datetime.now():%Y-%m-%d}",
                                                peers_instruction=peers_instruction),
            expected_output=TASK_EXPECTED_OUTPUT,
            agent=analyst,
        )
        crew = Crew(agents=[analyst], tasks=[task], process=Process.sequential, verbose=False)

        self._log("thought", f"Ejecuto el Crew con OpenAI ({self.model}).")
        output = crew.kickoff()
        text = (getattr(output, "raw", None) or str(output or "")).strip()
        if not text:
            raise RuntimeError("El Crew no devolvió un diagnóstico.")
        usage = getattr(output, "token_usage", None)
        if usage is not None:
            self._log("thought", "Consumo de tokens de OpenAI (redacción)", {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            })
        return text

    def _review_report(self, draft: str, ctx: AgentContext) -> tuple[str, dict[str, Any]]:
        """Coteja el borrador contra los datos y devuelve la versión corregida.

        Primero un chequeo determinístico marca cifras sin respaldo; después el
        LLM, con todos los datos a la vista, corrige números, comparaciones,
        etiquetas y contradicciones. Si la revisión falla, se entrega el borrador.
        """
        flagged = find_unsupported_numbers(draft, ctx)
        contradictions = find_contradictions(draft, ctx)
        self._log("thought", f"Verifico el borrador contra los datos: {len(contradictions)} "
                             f"contradicciones y {len(flagged)} cifras sin respaldo directo.",
                  {"contradicciones": contradictions, "cifras": flagged} if flagged or contradictions else None)
        data_json = compact_data_for_review(ctx)
        report = draft
        # Hasta dos pasadas: la segunda solo si quedaron contradicciones comprobadas.
        for attempt in range(2):
            user = (f"DATOS:\n{data_json}\n\n"
                    "CONTRADICCIONES DETECTADAS (corregir obligatoriamente):\n"
                    + ("\n".join(f"- {c}" for c in contradictions) if contradictions else "- Ninguna")
                    + "\n\nNÚMEROS DEL BORRADOR QUE NO APARECEN EN LOS DATOS:\n"
                    + ("\n".join(f"- {s}" for s in flagged) if flagged else "- Ninguno")
                    + f"\n\nBORRADOR:\n{report}")
            try:
                out = self._llm(temperature=0).call([{"role": "system", "content": REVIEW_SYSTEM},
                                                     {"role": "user", "content": user}])
                out = str(out or "").strip()
            except Exception as exc:
                self._log("error", "La verificación falló; se entrega el texto sin revisar.",
                          f"{type(exc).__name__}: {exc}")
                return report, {"status": "error" if attempt == 0 else "ok", "corrections": _diff_changes(draft, report),
                                "flagged_before": flagged, "contradictions_after": contradictions}

            new_report = out.partition(REVIEW_MARKER)[0]
            new_report = new_report.strip()
            if len(new_report) < 0.5 * len(report) or "###" not in new_report:
                self._log("error", "La verificación devolvió un informe incompleto; se descarta esa pasada.")
                break
            report = new_report
            contradictions = find_contradictions(report, ctx)
            flagged = find_unsupported_numbers(report, ctx)
            if not contradictions:
                break

        status = "ok" if report is not draft else "descartada"
        corrections = _diff_changes(draft, report)
        self._log("thought", f"Verificación completa: {len(corrections)} cambios en el texto, "
                             f"{len(contradictions)} contradicciones pendientes, {len(flagged)} cifras "
                             "derivadas sin coincidencia exacta.",
                  {"correcciones": corrections, "contradicciones": contradictions, "cifras": flagged}
                  if corrections or contradictions or flagged else None)
        return report, {"status": status, "corrections": corrections,
                        "contradictions_after": contradictions, "flagged_after": flagged}

# ---------------------------------------------------------------------------
# Diagnóstico por reglas (motor sin LLM)
# ---------------------------------------------------------------------------

def fmt_big(n: float | None, currency: str = "") -> str:
    if n is None:
        return "N/D"
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= div:
            return f"{currency}{n / div:,.2f} {suf}".strip()
    return f"{currency}{n:,.0f}".strip()


def _fmt(v: Any, fmt: str = "{:.1f}", suffix: str = "") -> str:
    return "N/D" if v is None else fmt.format(v) + suffix


def _last(values: list | None) -> Any:
    vals = [v for v in (values or []) if v is not None]
    return vals[-1] if vals else None


def build_rule_based_diagnosis(ctx: AgentContext) -> str:
    m, f, t = ctx.market or {}, ctx.financials or {}, ctx.technicals or {}
    p, n = ctx.peers or {}, ctx.news or {}
    lines: list[str] = []
    fund = 0   # puntaje fundamental y de valuación
    tech = 0   # puntaje técnico
    risks: list[str] = []
    cur = m.get("currency", "")

    # Negocio
    lines.append("### Negocio y posicionamiento")
    if m:
        desc = ", ".join(x for x in (m.get("sector"), m.get("industry"), m.get("country")) if x)
        lines.append(f"**{m.get('name')}** ({m.get('ticker')}){': ' + desc if desc else ''}. "
                     f"Último cierre {m['price']:,.2f} {cur} al {m.get('last_date')}. "
                     f"Capitalización {fmt_big(m.get('market_cap'))} {cur}, EV "
                     f"{fmt_big(m.get('enterprise_value'))}.")
        for note in m.get("notes") or []:
            lines.append(f"> {note}")
    else:
        lines.append("No se pudo obtener el perfil del activo.")

    # Calidad y crecimiento
    lines.append("\n### Calidad, crecimiento y rentabilidad")
    if f.get("applicable"):
        s = f["summary"]
        lines.append(f"* Período {s['period']}: CAGR de ventas **{_fmt(s['revenue_cagr_pct'], suffix='%')}**, "
                     f"EPS **{_fmt(s['eps_cagr_pct'], suffix='%')}**, FCF **{_fmt(s['fcf_cagr_pct'], suffix='%')}**.")
        for fact in f.get("facts", [])[:5]:
            lines.append(f"* {fact}")
        roic = s["avg_roic_pct"]
        if roic is not None:
            fund += 1 if roic > 12 else -1 if roic < 5 else 0
        conv = s["avg_fcf_to_net_income_pct"]
        if conv is not None:
            fund += 1 if conv > 80 else -1 if conv < 50 else 0
            if conv < 50:
                risks.append(f"Baja conversión de utilidades en caja (promedio {conv:.0f}%).")
        if (s["revenue_cagr_pct"] or 0) > 10:
            fund += 1
        if (s["operating_margin_change_pp"] or 0) < -3:
            fund -= 1
            risks.append(f"Compresión del margen operativo ({s['operating_margin_change_pp']:+.1f} pp en {s['period']}).")
    elif f:
        lines.append(f.get("note", "Sin estados contables."))
    else:
        lines.append("Sin estados contables.")
    if m.get("operating_margin_ttm_pct") is not None:
        lines.append(f"* TTM ({m.get('fundamentals_basis')}): margen bruto {_fmt(m.get('gross_margin_ttm_pct'), suffix='%')}, "
                     f"operativo {_fmt(m.get('operating_margin_ttm_pct'), suffix='%')}, neto "
                     f"{_fmt(m.get('net_margin_ttm_pct'), suffix='%')}, ROE {_fmt(m.get('roe_ttm_pct'), suffix='%')}.")
    yoy = m.get("last_quarter_yoy")
    if yoy and yoy.get("revenue_pct") is not None:
        lines.append(f"* Último trimestre ({yoy['quarter']}) vs {yoy['vs_quarter']}: ventas "
                     f"{yoy['revenue_pct']:+.1f}%, utilidad {_fmt(yoy.get('net_income_pct'), '{:+.1f}', '%')}.")

    # Balance
    lines.append("\n### Balance y solidez financiera")
    if m:
        nde = m.get("net_debt_to_ebitda_ttm")
        lines.append(f"* Deuda neta (MRQ) {fmt_big(m.get('net_debt_mrq'))} {m.get('financial_currency') or ''}, "
                     f"deuda neta/EBITDA TTM **{_fmt(nde, '{:.2f}', 'x')}**, liquidez corriente "
                     f"{_fmt(m.get('current_ratio'), '{:.2f}')}.")
        if nde is not None:
            fund += 1 if nde < 1 else -1 if nde > 3 else 0
            if nde > 3:
                risks.append(f"Apalancamiento elevado (deuda neta/EBITDA {nde:.1f}x).")
        if f.get("applicable"):
            cov = _last(f["interest_coverage_x"])
            if cov is not None and cov < 3:
                risks.append(f"Cobertura de intereses ajustada ({cov:.1f}x en el último año fiscal).")

    # Valuación
    lines.append("\n### Valuación")
    if m:
        lines.append(f"* P/E trailing **{_fmt(m.get('pe_trailing'), suffix='x')}**, forward "
                     f"**{_fmt(m.get('pe_forward'), suffix='x')}**, EV/EBITDA **{_fmt(m.get('ev_ebitda'), suffix='x')}**, "
                     f"P/S {_fmt(m.get('price_to_sales'), '{:.2f}', 'x')}, P/B {_fmt(m.get('price_to_book'), '{:.2f}', 'x')}.")
        for fact in m.get("facts", []):
            if any(k in fact for k in ("P/E forward", "yield", "objetivo", "recomienda")):
                lines.append(f"* {fact}")
        fy, rf = m.get("fcf_yield_pct"), m.get("us10y_yield_pct")
        if fy is not None and rf is not None:
            fund += 1 if fy > rf + 2 else -1 if fy < rf - 2 else 0
        up = m.get("upside_to_target_mean_pct")
        if up is not None:
            fund += 1 if up > 15 else -1 if up < 0 else 0
        if p.get("premium_vs_median_pct"):
            core = [v for k, v in p["premium_vs_median_pct"].items() if k in ("pe_forward", "ev_ebitda")]
            if core:
                avg = sum(core) / len(core)
                fund += 1 if avg < -15 else -1 if avg > 25 else 0
            lines.append(f"* Comparables: {', '.join(p['peers'])}.")
            for fact in p.get("facts", []):
                if fact.startswith(("P/E", "EV/EBITDA", "Atención", "Sin datos")) or "reportan" in fact:
                    lines.append(f"  * {fact}")
        if m.get("quote_type") == "ETF":
            lines.append(f"* ETF: gasto anual {_fmt(m.get('etf_expense_ratio_pct'), '{:.2f}', '%')}, "
                         f"yield {_fmt(m.get('etf_yield_pct'), suffix='%')}.")

    # Técnico
    lines.append("\n### Análisis técnico")
    if t:
        trend = t.get("trend")
        tech += 2 if trend == "alcista" else -2 if trend == "bajista" else 0
        rsi = t.get("rsi14")
        if rsi is not None:
            tech += -1 if rsi >= 70 else 1 if rsi <= 30 else 0
        if t.get("macd_state"):
            tech += 1 if t["macd_state"].startswith("alcista") else -1
        b = t.get("vs_benchmark") or {}
        rel = b.get("relative_performance_pp")
        if rel is not None:
            tech += 1 if rel > 5 else -1 if rel < -5 else 0
        for fact in t.get("facts", []):
            lines.append(f"* {fact}")
        r = t.get("returns_pct") or {}
        lines.append("* Retornos: " + ", ".join(f"{k} {_fmt(v, '{:+.1f}', '%')}" for k, v in r.items()) + ".")
        if (t.get("volatility_window_annual_pct") or 0) > 50:
            risks.append(f"Volatilidad muy alta ({t['volatility_window_annual_pct']:.0f}% anualizada).")
    else:
        lines.append("Sin datos técnicos.")

    # Noticias
    lines.append("\n### Noticias y catalizadores")
    if n.get("headlines"):
        lines.append(f"Sentimiento léxico **{n['sentiment_label']}** (orientativo) sobre {n['count']} "
                     f"titulares de los últimos {NEWS_LOOKBACK_DAYS} días ({n['positive']} positivos, "
                     f"{n['negative']} negativos, {n['neutral']} neutros). Más recientes:")
        for h in n["headlines"][:5]:
            lines.append(f"* {h['date']} | {h['title']} ({h.get('publisher') or h.get('source')})")
    else:
        lines.append(n.get("error") or "No se encontraron titulares recientes.")
    if m.get("next_earnings_date"):
        lines.append(f"\nPróximo balance: **{m['next_earnings_date']}**.")

    # Riesgos
    if (m.get("short_pct_float") or 0) > 10:
        risks.append(f"Short interest alto ({m['short_pct_float']:.1f}% del float).")
    if m.get("financial_currency") == "ARS" or m.get("country") == "Argentina":
        risks.append("Riesgo país, cambiario e inflacionario de Argentina.")
    lines.append("\n### Riesgos principales")
    if risks:
        lines.extend(f"* {r}" for r in risks)
    else:
        lines.append("No se detectaron alertas cuantitativas; revisá riesgos cualitativos del negocio.")

    # Resumen y conclusión
    total = fund + tech
    bias = "alcista" if total >= 3 else "bajista" if total <= -3 else "neutral"
    summary = ["### Resumen ejecutivo",
               f"* Sesgo agregado **{bias}** (puntaje fundamental {fund:+d}, técnico {tech:+d}).",
               f"* Fundamentos y valuación: {'favorables' if fund > 1 else 'desfavorables' if fund < -1 else 'mixtos'}. "
               f"Técnico: {'favorable' if tech > 1 else 'desfavorable' if tech < -1 else 'mixto'}."]
    if fund > 1 and tech < -1:
        summary.append("* Divergencia: fundamentos sólidos con técnico débil; conviene esperar "
                       "confirmación de tendencia.")
    elif fund < -1 and tech > 1:
        summary.append("* Divergencia: el precio sube pese a fundamentos flojos; momentum sin respaldo.")
    lines = summary + [""] + lines

    lines.append("\n### Conclusión y qué monitorear")
    watch = []
    if t.get("sma200"):
        watch.append(f"SMA50 en {t['sma50']:,.2f} y SMA200 en {t['sma200']:,.2f}")
    if m.get("next_earnings_date"):
        watch.append(f"balance del {m['next_earnings_date']}")
    if m.get("net_debt_to_ebitda_ttm") is not None:
        watch.append("evolución de deuda neta/EBITDA")
    if f.get("applicable"):
        watch.append("margen operativo y conversión de caja")
    lines.append(f"Sesgo **{bias}**. A monitorear: {'; '.join(watch) or 'precio y volumen'}.")
    lines.append("\n_Análisis automático con reglas cuantitativas, con fines educativos. "
                 "No es una recomendación de inversión._")
    return "\n".join(lines)
