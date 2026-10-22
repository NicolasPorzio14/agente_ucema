"""
assistant.py
============
Asistente de chat sobre un análisis ya generado.

No vuelve a correr el crew ni a descargar datos: cada pregunta es una sola
llamada al LLM con el informe final y todos los datos que devolvieron las
herramientas (más la simulación Monte Carlo, si se corrió). Cada respuesta
pasa por los mismos controles determinísticos que el informe:
  * cifras que no figuran en los datos (find_unsupported_numbers), y
  * contradicciones técnicas (find_contradictions).
Además, el modelo declara de qué bloque de datos sacó cada respuesta.

El módulo no importa Streamlit.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from agent import (DEFAULT_OPENAI_MODEL, AgentContext, AnalysisResult, compact_data_for_review,
                   find_contradictions, find_unsupported_numbers)

SOURCES = {
    "informe": "Informe",
    "mercado": "Mercado y valuación",
    "estados": "Estados contables",
    "tecnico": "Técnico",
    "comparables": "Comparables",
    "noticias": "Noticias",
    "simulacion": "Monte Carlo",
}
SOURCE_MARKER = "<<<FUENTES>>>"
MAX_HISTORY_MESSAGES = 8

CHAT_SYSTEM = """Sos el asistente de un analista de equity research. Respondés preguntas sobre el \
análisis de {ticker} (temporalidad {period}) que ya generó el equipo. Fecha de hoy: {today}.

Tenés tres fuentes, y SOLO esas:
1. INFORME: el informe final, ya verificado.
2. DATOS: el JSON con lo que devolvió cada herramienta (get_market_data = mercado y valuación, \
get_financial_statements = estados contables, calculate_technical_indicators = técnico, \
compare_peers = comparables, get_company_news_sentiment = noticias). Los campos "facts" son \
lecturas ya calculadas: usalas tal cual para comparaciones, rankings y posiciones.
3. SIMULACION: resultados de una simulación Monte Carlo del precio (puede no estar).

Reglas:
- Respondé en español, directo y breve (3 a 8 oraciones o una lista corta). Usá Markdown simple.
- Cada cifra que cites tiene que estar en los DATOS, el INFORME o la SIMULACION, con el mismo \
redondeo. No inventes números, fechas, comparables ni noticias. Si calculás algo (una diferencia, \
un cociente), decí que es un cálculo y mostrá la cuenta.
- Si la respuesta no está en las fuentes, decilo ("eso no está en los datos del análisis") y \
sugerí qué dato haría falta. No completes con conocimiento general sobre la empresa.
- Distinguí TTM, FY y MRQ como figuran en los datos. No contradigas los "facts".
- Podés explicar conceptos (qué es un golden cross, qué mide el VaR), pero aplicados a estos datos.
- No des recomendaciones personalizadas de inversión; podés resumir lo que dicen los datos y el informe.
- Al final, en una línea aparte, escribí {marker} seguido de las fuentes que usaste, separadas \
por coma, elegidas de: {sources}. Ejemplo: {marker} tecnico, mercado
"""


@dataclass
class ChatAnswer:
    text: str
    sources: list[str] = field(default_factory=list)
    flagged: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    error: str | None = None


def context_from_result(result: AnalysisResult) -> AgentContext:
    """Reconstruye el contexto del agente para reutilizar los verificadores."""
    return AgentContext(period=result.period, market=result.market, financials=result.financials,
                        technicals=result.technicals, peers=result.peers, news=result.news)


def suggested_questions(result: AnalysisResult, has_simulation: bool = False) -> list[str]:
    tk = result.ticker
    m, f, p = result.market or {}, result.financials or {}, result.peers or {}
    qs = [f"¿Cuáles son los 3 principales riesgos de {tk} según los datos?",
          "¿Qué dicen los indicadores técnicos hoy y qué niveles conviene vigilar?"]
    if p.get("table"):
        qs.append(f"¿{tk} está cara o barata contra sus comparables? ¿En qué múltiplos?")
    if f.get("applicable"):
        qs.append("¿Qué tan sólida es la generación de caja y el balance?")
    if has_simulation and m.get("target_mean"):
        qs.append("¿Qué tan probable es que llegue al precio objetivo de los analistas?")
    if (result.news or {}).get("headlines"):
        qs.append("¿Qué noticias recientes podrían mover la acción?")
    qs.append("¿Qué tendría que pasar para que la tesis cambie?")
    return qs[:6]


def _parse_sources(raw: str) -> tuple[str, list[str]]:
    body, _, tail = raw.partition(SOURCE_MARKER)
    found: list[str] = []
    for token in re.split(r"[,\s]+", tail.lower()):
        key = (token.strip(" .;:*`").replace("é", "e").replace("ó", "o").replace("í", "i"))
        if key in SOURCES and key not in found:
            found.append(key)
    return body.strip(), found


def build_messages(question: str, history: list[dict[str, str]], result: AnalysisResult,
                   simulation: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """Mensajes para el LLM. El bloque de datos va primero y fijo, así OpenAI
    puede reutilizarlo con su caché de prompts entre preguntas."""
    system = CHAT_SYSTEM.format(ticker=result.ticker, period=result.period, today=date.today().isoformat(),
                                marker=SOURCE_MARKER, sources=", ".join(SOURCES))
    data = compact_data_for_review(context_from_result(result))
    sim = json.dumps(simulation, ensure_ascii=False, default=str) if simulation else "No se corrió."
    system += f"\n\nDATOS:\n{data}\n\nINFORME:\n{result.diagnosis}\n\nSIMULACION:\n{sim}"
    msgs = [{"role": "system", "content": system}]
    for h in history[-MAX_HISTORY_MESSAGES:]:
        msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": question})
    return msgs


def answer_question(question: str, history: list[dict[str, str]], result: AnalysisResult,
                    api_key: str, model: str | None = None,
                    simulation: dict[str, Any] | None = None) -> ChatAnswer:
    """Responde una pregunta sobre el análisis y verifica la respuesta contra los datos."""
    from crewai import LLM

    model = model or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    try:
        llm = LLM(model=model, api_key=api_key, timeout=90, max_retries=2, temperature=0.2)
        raw = str(llm.call(build_messages(question, history, result, simulation)) or "").strip()
    except Exception as exc:
        return ChatAnswer("", error=f"{type(exc).__name__}: {exc}")
    if not raw:
        return ChatAnswer("", error="El modelo devolvió una respuesta vacía.")
    text, sources = _parse_sources(raw)
    ctx = context_from_result(result)
    # El informe ya verificado también respalda cifras (por ejemplo, cálculos que el revisor aceptó).
    extra = {"simulacion": simulation, "informe": result.diagnosis}
    return ChatAnswer(text, sources, flagged=find_unsupported_numbers(text, ctx, extra),
                      contradictions=find_contradictions(text, ctx))
