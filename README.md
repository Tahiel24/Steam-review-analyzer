# Steam Review Intelligence

**Cuando 500 reseñas negativas caen en 48 horas, un desarrollador necesita una respuesta rápida: ¿es un bug real o una campaña coordinada?** Este sistema separa la señal del ruido — distingue crítica legítima de review bombing, y convierte miles de reseñas en un informe ejecutivo accionable.

<!--
📸 TODO: reemplazar por captura real del dashboard de Streamlit
![Dashboard principal](docs/images/dashboard.png)
-->

🔗 **[Demo en vivo](#)** *(pendiente de deploy)* · 📄 **[Swagger / API Docs](#)** *(pendiente de deploy — correr localmente con `/docs`)*

---

## El problema

Cuando un juego recibe un aluvión de reseñas negativas, el desarrollador tiene dos preguntas urgentes y distintas:

1. **¿Es real?** ¿Hay un problema técnico o de contenido que la comunidad está señalando legítimamente?
2. **¿Es orgánico?** ¿O es una campaña coordinada (cambio de precio, decisión polémica, guerra de reseñas) que infla artificialmente el ratio negativo?

Confundir ambas cosas lleva a gastar semanas de desarrollo arreglando algo que no era el problema real, o a ignorar una queja genuina pensando que es ruido. Este proyecto ataca directamente esa ambigüedad.

## Qué hace

| Capacidad | Cómo lo resuelve |
|---|---|
| **Clasificación de sentimiento** | DistilBERT fine-tuneado sobre el dataset de Steam Reviews, con manejo de desbalance de clases (class weighting) |
| **Detección de campañas coordinadas** | Serie temporal por juego con rolling Z-score (14 días), calibrada por volumen relativo y ratio de negatividad — no es un umbral fijo, se ajusta a la línea base de cada juego |
| **Evidencia, no solo un número** | Búsqueda semántica (RAG) sobre las reseñas reales vía ChromaDB + Sentence-Transformers, para respaldar cualquier alerta con texto concreto |
| **Informe ejecutivo en lenguaje natural** | Un LLM (Groq) redacta un diagnóstico y acciones recomendadas en streaming, citando la evidencia recuperada |
| **Todo expuesto y usable** | API REST (FastAPI) + dashboard interactivo (Streamlit) con gráfico de anomalías y asistente conversacional |

## Cómo se ve

<!-- 📸 TODO: agregar 2-3 capturas o un GIF corto del flujo completo (selección de juego → gráfico de anomalías → informe generado) -->

- Grafico con el volumen de reseñas sobre un juego elegido y el porcentaje de reseñas negativas sobre el mismo
- Asistente de triage: consulta en lenguaje natural → evidencia recuperada de reseñas reales → informe generado en streaming
- Analisis de sentimiento de una reseña

## Arquitectura

```
Dataset (Kaggle Steam Reviews, ~21M reseñas, filtrado a inglés)
        │
        ▼
  Limpieza y EDA  ──────────────────────────►  notebooks/01_eda.ipynb
        │
        ▼
  Baseline (TF-IDF + Regresión Logística) ──►  notebooks/02_baseline_ml.ipynb
        │
        ▼
  Fine-tuning DistilBERT (sentimiento) ─────►  notebooks/03_multilabel_dl.ipynb
        │                                       src/models/sentiment_classifier.py
        ▼
  Detección de anomalías (rolling Z-score) ─►  notebooks/04_review_bombing.ipynb
        │                                       src/anomaly/detector.py
        ▼
  Indexación vectorial (ChromaDB) ──────────►  notebooks/05_vector_indexing_rag.ipynb
        │                                       src/rag/vector_store.py
        ▼
  Resumen ejecutivo (LLM vía Groq) ─────────►  notebooks/06_executive_summarizer_llm.ipynb
        │                                       src/genai/summarizer.py
        ▼
  ┌─────────────────┐       ┌──────────────────────┐
  │  API (FastAPI)   │◄──────│ Dashboard (Streamlit)│
  │  app/api/        │       │ app/frontend/         │
  └─────────────────┘       └──────────────────────┘
```

## Stack técnico

- **Datos y procesamiento**: Pandas, NumPy
- **ML clásico**: scikit-learn (TF-IDF + Regresión Logística)
- **Deep Learning**: PyTorch, Hugging Face Transformers (DistilBERT fine-tuneado)
- **Detección de anomalías**: series temporales, Z-score con calibración dinámica por juego
- **RAG**: Sentence-Transformers (`all-MiniLM-L6-v2`) + ChromaDB
- **Generación**: Groq API (streaming)
- **Backend**: FastAPI
- **Frontend**: Streamlit + Plotly
- **Deploy**: *(pendiente — Render / Railway / Hugging Face Spaces)*

## API

| Endpoint | Método | Descripción |
|---|---|---|
| `/health` | GET | Estado del sistema, total de reseñas cargadas, vectores indexados |
| `/games` | GET | Lista de juegos disponibles en el dataset |
| `/analytics/anomalies` | GET | Anomalías detectadas para un juego (fecha, ratio negativo, Z-score) |
| `/predict/sentiment` | POST | Clasifica el sentimiento de un texto de reseña |
| `/reports/generate-stream` | GET | Genera el informe ejecutivo en streaming (RAG + LLM) |

Documentación interactiva completa disponible en `/docs` (Swagger UI) al correr el proyecto localmente.

## Cómo correrlo localmente

```bash
git clone https://github.com/Tahiel24/steam_review_analyzer.git
cd steam_review_analyzer

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configurar variables de entorno (API key de Groq, etc.)
cp .env.example .env

# Descargar el dataset (Kaggle "Steam Reviews Dataset") en data/raw/
# Ejecutar el pipeline de limpieza:
python -m src.data.clean

# Indexar reseñas en ChromaDB para RAG:
python -m src.rag.vector_store

# Levantar la API:
uvicorn app.api.main:app --reload

# En otra terminal, levantar el dashboard:
streamlit run app/frontend/streamlit_app.py
```

## Notebooks (proceso de desarrollo)

Cada fase queda documentada como notebook independiente, con el razonamiento detrás de cada decisión:

1. `01_eda.ipynb` — Exploración del dataset, filtrado por idioma, distribución de clases
2. `02_baseline_ml.ipynb` — Modelo baseline (TF-IDF + Regresión Logística) como referencia
3. `03_multilabel_dl.ipynb` — Fine-tuning de DistilBERT, manejo de desbalance de clases
4. `04_review_bombing.ipynb` — Metodología y validación de la detección de anomalías
5. `05_vector_indexing_rag.ipynb` — Indexación y búsqueda semántica
6. `06_executive_summarizer_llm.ipynb` — Diseño del prompt y generación de informes

## Decisiones de alcance
- El dataset se filtró a reseñas en **inglés** para evitar degradar la calidad del fine-tuning con un modelo monolingüe.

## Roadmap

- [ ] Clasificación multi-etiqueta (separar eje técnico/rendimiento del eje contenido/gameplay)
- [ ] Deploy público (Render / HF Spaces)
- [ ] Bot de Discord como cliente alternativo de la API
- [ ] Evaluación cuantitativa de la detección de anomalías contra más casos históricos documentados

## Limitación identificada: oraciones con estructura adversativa

Durante pruebas manuales del clasificador se observó un patrón consistente: en reseñas con estructura adversativa (elogio inicial + crítica introducida por "but"/"however"), el modelo tiende a clasificar según la polaridad de la primera cláusula, subestimando la cláusula crítica que sigue. Por ejemplo:

> *"It's a good game but it has many problems related with FPS when you enter in determinated zones"* → clasificado como **Positivo (97.86% confianza)**

**Mitigaciones planificadas:**
1. **Aspect-Based Sentiment Analysis (ABSA)**: desacoplar la evaluación global hacia un análisis por entidades (rendimiento, gráficos, jugabilidad, monetización), en vez de una polaridad única por reseña.
2. **Data augmentation con estructuras contrastivas**: reentrenamiento incorporando ejemplos sintéticos con conectores adversativos (`but`, `however`, `although`) para forzar al modelo a ponderar la cláusula resolutiva.
3. **Inferencia híbrida en cascada**: derivar hacia el LLM (Groq) los casos con baja confianza o presencia de conectores disyuntivos, para desambiguación contextual que el clasificador binario no puede resolver por sí solo.

## Licencia

MIT — ver [LICENSE](LICENSE)
