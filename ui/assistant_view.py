"""
ui/assistant_view.py
====================
Pestaña "Asistente": chat sobre el análisis ya generado, sin volver a correr
el crew. La conversación vive en st.session_state y se reinicia cuando cambia
el análisis.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from agent import AnalysisResult
from assistant import SOURCES, answer_question, suggested_questions
from ui import components as ui


AVATARS = {"assistant": ":material/query_stats:", "user": ":material/person:"}


def _ask(question: str) -> None:
    st.session_state["chat_pending"] = question


def _clear() -> None:
    st.session_state["chat"]["messages"] = []


def _show(msg: dict[str, Any]) -> None:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        # "$" sin escapar activa LaTeX en st.markdown (ej. "US$ 5 ... US$ 7").
        st.markdown(msg["content"].replace("$", "\\$"))
        if msg["role"] == "assistant" and msg.get("verified"):
            ui.render(ui.chat_meta(msg.get("sources", []), SOURCES, msg.get("flagged", []),
                                   msg.get("contradictions", [])))


def render(result: AnalysisResult, panel, api_key: str, model: str, simulation: dict[str, Any] | None) -> None:
    state = st.session_state.setdefault("chat", {"key": None, "messages": []})
    key = (result.ticker, result.period, result.engine, (result.market or {}).get("last_date"), len(result.diagnosis))
    if state["key"] != key:
        state.update(key=key, messages=[])

    left, right = st.columns([2.3, 1], gap="medium")
    with right:
        with panel("Preguntas sugeridas", "un clic para preguntar", key="chat-suggest"):
            with st.container(key="suggest"):
                for i, q in enumerate(suggested_questions(result, simulation is not None)):
                    st.button(q, key=f"sq{i}", on_click=_ask, args=(q,), disabled=not api_key, width="stretch")
        with panel("Cómo funciona", key="chat-help"):
            ui.render(
                "<ul class='facts'>"
                "<li>Responde con el informe final y los datos que ya trajeron las herramientas, más la "
                "simulación Monte Carlo. No vuelve a correr el agente ni a descargar datos.</li>"
                "<li>Cada respuesta pasa por los mismos controles que el informe: cifras sin respaldo en los "
                "datos y contradicciones técnicas.</li>"
                "<li>Si algo no está en los datos, lo dice en lugar de inventarlo.</li></ul>")
            if state["messages"]:
                st.button("Borrar conversación", on_click=_clear, width="stretch")

    with left:
        with panel(f"Asistente de {result.ticker}", f"modelo {model}" if api_key else "requiere OPENAI_API_KEY",
                   key="chat"):
            # Los mensajes van en un contenedor creado antes del input, así las
            # respuestas nuevas quedan arriba del cuadro de texto.
            box = st.container()
            with box:
                _show({"role": "assistant", "content":
                       f"Hola. Ya leí el informe de **{result.ticker}** ({result.period}) y todos los datos del análisis. "
                       "Preguntame lo que quieras: valuación, técnico, balance, comparables, noticias o la simulación."})
                for msg in state["messages"]:
                    _show(msg)
            if not api_key:
                st.info("El asistente usa OpenAI. Cargá OPENAI_API_KEY en el archivo .env o en la barra lateral "
                        "(motor CrewAI + OpenAI).")
            question = st.chat_input(f"Preguntá sobre {result.ticker}…", disabled=not api_key, key="chat_input")
            question = question or st.session_state.pop("chat_pending", None)
            if question and api_key:
                history = [{"role": m["role"], "content": m["content"]} for m in state["messages"]]
                user_msg = {"role": "user", "content": question}
                state["messages"].append(user_msg)
                with box:
                    _show(user_msg)
                    with st.spinner("Pensando y verificando contra los datos…"):
                        ans = answer_question(question, history, result, api_key, model, simulation)
                    if ans.error:
                        state["messages"].pop()
                        st.error(f"No se pudo responder: {ans.error}")
                    else:
                        msg = {"role": "assistant", "content": ans.text, "sources": ans.sources, "verified": True,
                               "flagged": ans.flagged, "contradictions": ans.contradictions}
                        state["messages"].append(msg)
                        _show(msg)
