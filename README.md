# Agente de Análisis Financiero (Streamlit + CrewAI + OpenAI + yfinance)

Un agente analista con cinco herramientas que arma un informe de research sobre una acción,
ADR o ETF: negocio, calidad y crecimiento, balance, valuación contra comparables y consenso,
análisis técnico, noticias, riesgos y escenarios.

## Estructura

```
financial_agent/
├── app.py            # Interfaz Streamlit: secciones Análisis y Cartera
├── agent.py          # Herramientas, Crew de CrewAI, verificadores y motor de reglas de respaldo
├── assistant.py      # Asistente de chat sobre el informe (sin volver a correr el crew)
├── simulation.py     # Monte Carlo del precio (bootstrap por bloques o lognormal)
├── portfolio.py      # Cartera: pesos, correlaciones, diversificación, contribución al riesgo
├── report_pdf.py     # Exportación del informe a PDF (fpdf2)
├── ui/               # Tema, componentes HTML, gráficos y vistas de cada pestaña
├── assets/fonts/     # IBM Plex Sans (licencia OFL) para el PDF
├── requirements.txt
├── .env.example      # Plantilla para tu OPENAI_API_KEY
└── .gitignore        # Evita subir .env a un repositorio
```

## Ejecución local

```bash
cd financial_agent
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

# Configurá tu clave de OpenAI: copiá la plantilla y pegá tu clave
cp .env.example .env               # Windows: copy .env.example .env
#   luego editá .env:  OPENAI_API_KEY=sk-...

python -m streamlit run app.py
```

Usá `python -m streamlit` en lugar de `streamlit` para asegurarte de correr el Streamlit del
entorno virtual y no uno instalado globalmente.

Se abre en http://localhost:8501. Requiere Python 3.10 a 3.13.

## Herramientas del agente

| Herramienta | Qué aporta |
|---|---|
| `get_market_data` | Perfil, cotización, P/E trailing y forward, PEG, EV/EBITDA, EV/Ventas, P/S, P/B, earnings y FCF yield vs Treasury 10 años, márgenes, ROE, ROA, deuda neta/EBITDA, liquidez, consenso de analistas (objetivo, upside, tendencia y cambios de calificación), short interest, tenencia y próximo balance. |
| `get_financial_statements` | 4/5 años de estados contables y últimos trimestres: ventas, márgenes, EBITDA, EPS, flujo operativo, capex, FCF, conversión de caja, ROE, ROIC, cobertura de intereses, dilución, recompras, dividendos y CAGR. |
| `calculate_technical_indicators` | SMA 20/50/200 y cruces, RSI, MACD, Bollinger, ATR, retornos 1M/3M/6M/YTD/1Y, volatilidad, drawdown, Sharpe, Sortino, beta y fuerza relativa vs SPY o Merval, soportes y resistencias. |
| `compare_peers` | Múltiplos y rentabilidad de 3 a 6 comparables, mediana y prima o descuento. Con CrewAI el agente elige los comparables si no los cargás. |
| `get_company_news_sentiment` | Titulares de los últimos 30 días de Yahoo Finance y Google News en español e inglés, deduplicados, con un sentimiento léxico orientativo. |

## Motores del agente

| Motor | Requiere | Cómo decide |
|---|---|---|
| CrewAI + OpenAI | `OPENAI_API_KEY` | Un agente "Analista Senior de Equity Research" que llama las herramientas, elige comparables y redacta el informe completo. Modelo por defecto: `gpt-4.1-mini` (editable en la barra lateral o con `OPENAI_MODEL`); con `gpt-4.1` el informe es más profundo. |
| Reglas cuantitativas | Nada | Plan fijo de herramientas e informe por puntajes fundamental y técnico separados. No consume créditos. |

Si OpenAI falla (clave inválida, sin créditos, red, rate limit), el agente pasa solo al motor de
reglas y lo registra en el panel de razonamiento.

## Después del informe

| Pestaña / sección | Qué hace |
|---|---|
| Asistente | Chat sobre el análisis ya generado. Cada pregunta es una sola llamada al LLM con el informe, los datos de las herramientas y la simulación; no vuelve a correr el crew. Cada respuesta pasa por los mismos verificadores que el informe (cifras sin respaldo y contradicciones técnicas) y declara de qué bloque de datos salió. Requiere `OPENAI_API_KEY`. |
| Simulación | Monte Carlo de 10.000 trayectorias a 1, 3, 6 o 12 meses: probabilidad de tocar y de terminar más allá del precio objetivo del consenso, de perforar un soporte, rango 90%, VaR y CVaR. Bootstrap por bloques de retornos reales o lognormal; deriva neutral al riesgo o histórica. Semilla fija. |
| Descargar PDF | Informe completo con portada, métricas clave, gráfico de precio, simulación, control de calidad y fuentes. Se genera al hacer clic. |
| Cartera | Lista de tickers con pesos editables (vacío = 1/n). Retorno, volatilidad, drawdown, Sharpe, beta, VaR/CVaR, ratio de diversificación, activos y apuestas efectivas, contribución de cada activo al riesgo, matriz de correlaciones, concentración por sector y métricas por activo. No es un optimizador: no sugiere pesos. |

## Cómo se evita que el informe tenga errores

1. **Una sola fuente por métrica.** Márgenes, EBITDA, FCF, EV/EBITDA, P/S y FCF yield se calculan
   sumando los últimos 4 trimestres (TTM), con el mismo método para la empresa y sus comparables.
   Los datos de resumen de Yahoo solo se usan como respaldo; en algunos casos son incorrectos
   (por ejemplo, el margen bruto de VIST o el FCF de WMT).
2. **Lecturas precalculadas ("facts").** Cada herramienta entrega las comparaciones ya hechas:
   rankings, posición contra la mediana, cruces de medias con fecha, posición en Bollinger y
   tendencias de márgenes. El LLM las cita en lugar de comparar números por su cuenta.
3. **Verificación del borrador.** Antes de mostrar el informe:
   * un chequeo automático marca cifras que no existen en los datos (incluidos errores de unidad,
     como "329 millones" cuando son 329 mil millones),
   * un detector marca contradicciones técnicas (tendencia, cruces, RSI, MACD, Bollinger),
   * un revisor LLM corrige el borrador con todos los datos a la vista; si quedan contradicciones,
     hace una segunda pasada.

   En la solapa Informe, "Control de calidad" muestra los cambios reales entre el borrador y la
   versión final.
4. **Comparables filtrados.** Los de otro sector quedan fuera de la mediana. Si quedan menos de 3,
   la herramienta le pide al agente que los reemplace.
5. **Entidades financieras.** En bancos y aseguradoras se omiten margen bruto, EBITDA, deuda neta
   y FCF, que no tienen sentido económico para esas empresas.

## Tickers

Se usan símbolos de Yahoo Finance: `AAPL`, `SPY`, `VIST`, `GGAL` (ADRs en NYSE/NASDAQ).
Para el panel local de BYMA agregá el sufijo `.BA` (ej. `GGAL.BA`, `YPFD.BA`).

Cuando una empresa cotiza en USD pero reporta en otra moneda (YPF y GGAL en ARS, PBR en BRL),
Yahoo calcula EV/EBITDA, EV/Ventas, P/S y P/B mezclando monedas. La app los omite y lo avisa.

## Problemas frecuentes

**`unable to get local issuer certificate` / `CertificateVerifyError`**: un antivirus (Avast,
Kaspersky, ESET) o un proxy corporativo está interceptando HTTPS. La app detecta solo el
certificado de Avast. Para otros, agregá en `.env` la ruta al certificado raíz:
`EXTRA_CA_CERTS=C:\ruta\al\certificado.pem`.
