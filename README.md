# Movie Match — AI Movie Recommender & Data Pipeline (Apache Airflow)

**Movie Match** es un recomendador de películas por IA y pipeline de ingeniería de datos orquestado con **Apache Airflow**.

Resuelve el clásico dilema: *"¿Qué vemos hoy?"* analizando el significado semántico profundo de las sinopsis (*overviews*) de las películas favoritas de cada persona con **NLP & Embeddings**, calcula el **vector de gusto combinado** (el punto medio en el espacio vectorial) y recomienda películas del catálogo de [TMDB](https://www.themoviedb.org/).

---

## Características Principales

- **Orquestación con Apache Airflow**: DAG modular (`movie_match_pipeline`) que automatiza la ingesta de datos, el cálculo de embeddings y el ranking de recomendaciones.
- **Búsqueda en vivo**: Integración con la API v4 de TMDB.
- **Embeddings Semánticos**: Uso de `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face) para vectorizar las sinopsis.
- **Similitud Coseno**: Medición de distancia semántica entre el gusto conjunto y el catálogo de candidatas.
- **Visualización 2D (PCA)**: Gráfico del espacio vectorial que muestra las distancias entre las películas de cada persona, la intersección y las recomendaciones.
- **Seguridad**: Configuración con variables de entorno (`python-dotenv`) para resguardar tokens de API.

---

## Arquitectura de Orquestación en Airflow

```
[ Task 1: fetch_tmdb_catalog ] ──► Descarga catálogo TMDB (Staging JSON)
              │
              ▼
[ Task 2: generate_embeddings ] ──► Vectoriza sinopsis con SentenceTransformer (.npy)
              │
              ▼
[ Task 3: calculate_recommendations ] ──► Vector de gusto combinado + Cosine Similarity
              │
              ▼
[ Task 4: generate_pca_plot ] ──► Reporte gráfico 2D PCA de recomendaciones
```

---

## Estructura del Repositorio

```
movie/
├── dags/
│   └── movie_match_dag.py # DAG principal de Apache Airflow
├── data/                  # Almacenamiento local staging (catalogo y embeddings)
├── output/                # Artefactos generados (CSV de recomendaciones y gráficos)
├── .env.example           # Plantilla para variables de entorno (Token TMDB)
├── .gitignore              # Archivos excluidos del repositorio (.env, cache, etc.)
├── requirements.txt        # Dependencias de Python (requests, airflow, transformers, etc.)
├── movie_match.ipynb       # Notebook interactivo para pruebas rápidas
├── movie_match.py          # Script ejecutable por consola (CLI interactivo)
└── README.md               # Documentación del proyecto
```

---

## Instalación y Uso

### 1. Clonar e instalar dependencias

```bash
git clone https://github.com/tu-usuario/movie-match.git
cd movie-match

pip install -r requirements.txt
```

### 2. Configurar la API Key de TMDB

Copiá `.env.example` como `.env` y agregá tu token Bearer v4:

```bash
cp .env.example .env
```

```env
TMDB_BEARER_TOKEN=tu_token_bearer_aqui
```

---

## Formas de Ejecución

### Opción A: Orquestado con Apache Airflow

Podés iniciar Airflow de forma standalone para probar el DAG:

```bash
export AIRFLOW_HOME=$(pwd)
airflow standalone
```

Luego, ingresá a la interfaz web de Airflow en `http://localhost:8080`, activá el DAG `movie_match_pipeline` y ejecutalo manualmente.

### Opción B: CLI Interactivo (Script en consola)

```bash
python3 movie_match.py
```

### Opción C: Notebook de Jupyter

```bash
jupyter notebook movie_match.ipynb
```

---

## Tecnologías Utilizadas

- **Orquestación & Data Engineering**: Apache Airflow (`DAG`, `@task` TaskFlow API)
- **Lenguaje**: Python 3.9+
- **NLP & Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Machine Learning**: `scikit-learn` (PCA, Cosine Similarity), `numpy`
- **Data & Visualización**: `pandas`, `matplotlib`
- **API Client**: `requests`, `python-dotenv`

---

## Licencia

Este proyecto está bajo la Licencia MIT.
