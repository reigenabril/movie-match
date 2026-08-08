# Movie Match — AI Movie Recommender & Data Pipeline (Apache Airflow & Streamlit)

**Movie Match** es un recomendador de películas por IA, interfaz web interactiva y pipeline de ingeniería de datos orquestado con **Apache Airflow**.

Resuelve el clásico dilema: *"¿Qué vemos hoy?"* analizando el significado semántico profundo de las sinopsis (*overviews*) de las películas favoritas de cada persona con **NLP & Embeddings**, calcula el **vector de gusto combinado** (el punto medio en el espacio vectorial) y recomienda películas del catálogo de [TMDB](https://www.themoviedb.org/).

---

## Características Principales

- **Modo Grupo (N Personas)**: Permite combinar los gustos de 2 o más participantes con pesos e importancia personalizada por persona.
- **Preferencias Negativas ("Veto")**: Permite añadir películas desaprobadas (restando/penalizando su embedding semántico) y vetar géneros no deseados.
- **Interfaz Web Interactiva (Streamlit)**: Aplicación web moderna (`app.py`) con visualización de carátulas/posters de TMDB, porcentajes de coincidencia y gráficos vectoriales.
- **Orquestación con Apache Airflow & Docker**: DAG modular (`movie_match_pipeline`) desplegado en Docker Compose.
- **Búsqueda en vivo**: Integración con la API v4 de TMDB.
- **Embeddings Semánticos**: Uso de `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face) para vectorizar las sinopsis.
- **Similitud Coseno**: Medición de distancia semántica entre el gusto conjunto y el catálogo de candidatas.
- **Visualización 2D (PCA)**: Gráfico del espacio vectorial que muestra las distancias entre las películas de cada integrante del grupo, películas vetadas y recomendaciones.
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
├── app.py                 # Aplicación Web Interactiva con Streamlit
├── dags/
│   └── movie_match_dag.py # DAG principal de Apache Airflow
├── Dockerfile             # Imagen de Docker optimizada
├── docker-compose.yaml    # Orquestador multi-contenedor para Airflow
├── data/                  # Almacenamiento local staging (catalogo y embeddings)
├── output/                # Artefactos generados (CSV de recomendaciones y gráficos)
├── .env.example           # Plantilla para variables de entorno (Token TMDB)
├── .gitignore              # Archivos excluidos del repositorio (.env, cache, etc.)
├── requirements.txt        # Dependencias de Python (requests, airflow, streamlit, etc.)
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

### Opción A: Interfaz Web Interactiva (Streamlit)

```bash
streamlit run app.py
```

### Opción B: Despliegue con Docker Compose (Airflow)

```bash
docker compose up -d
```

Accedé a la interfaz de Airflow en `http://localhost:8080` (Usuario: `admin` / Contraseña: `admin`).

### Opción C: CLI Interactivo (Script en consola)

```bash
python3 movie_match.py
```

### Opción D: Notebook de Jupyter

```bash
jupyter notebook movie_match.ipynb
```

---

## Tecnologías Utilizadas

- **Frontend & Web UI**: Streamlit
- **Orquestación & Data Engineering**: Apache Airflow (`DAG`, `@task` TaskFlow API), Docker Compose
- **Lenguaje**: Python 3.9+
- **NLP & Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Machine Learning**: `scikit-learn` (PCA, Cosine Similarity), `numpy`
- **Data & Visualización**: `pandas`, `matplotlib`
- **API Client**: `requests`, `python-dotenv`

---

## Licencia

Este proyecto está bajo la Licencia MIT.
