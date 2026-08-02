# 🎬 Movie Match — AI Movie Recommender for Couples & Friends

**Movie Match** es un recomendador de películas por IA que resuelve el clásico dilema: *"¿Qué vemos hoy?"*.

En lugar de recomendar por simples coincidencias de género, **Movie Match** analiza el significado profundo de las sinopsis (*overviews*) de las películas favoritas de cada persona usando **embeddings semánticos**, calcula el **vector de gusto combinado** (el punto medio en el espacio vectorial) y encuentra las mejores recomendaciones en el catálogo de [TMDB](https://www.themoviedb.org/).

---

## 🌟 Características Principal

- 🔍 **Búsqueda en vivo**: Integración con la API de TMDB v4 para obtener sinopsis, poster y detalles actualizados.
- 🧠 **Embeddings NLP**: Uso del modelo `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face) para vectorizar las sinopsis.
- 📐 **Similitud Coseno**: Medición precisa de distancia semántica entre el gusto combinado y las películas candidatas.
- 📊 **Visualización en 2D (PCA)**: Gráfico interactivo/estático que proyecta en 2 dimensiones el espacio vectorial de los gustos de cada persona, el punto de intersección y las películas recomendadas.
- 🔐 **Seguridad**: Configuración limpia mediante variables de entorno (`python-dotenv`) para proteger tus tokens de API.

---

## 🏗️ Arquitectura del Pipeline

```
[ Persona 1: Títulos ] ──► [ TMDB API ] ──► [ Overviews ] ──► [ SentenceTransformer ] ──► Vector Gusto 1 ┐
                                                                                                          ├─► Vector Combinado ─► Cosine Similarity vs Catálogo ─► Top 10 Recomendaciones
[ Persona 2: Títulos ] ──► [ TMDB API ] ──► [ Overviews ] ──► [ SentenceTransformer ] ──► Vector Gusto 2 ┘
```

---

## 📁 Estructura del Proyecto

```
movie/
├── .env.example          # Plantilla para variables de entorno (Token TMDB)
├── .gitignore             # Archivos excluidos del control de versiones (.env, cache, etc.)
├── requirements.txt       # Dependencias de Python necesarias
├── movie_match.ipynb      # Notebook interactivo de Jupyter
├── movie_match.py         # Script / CLI ejecutable de Python
└── README.md              # Documentación del repositorio
```

---

## 🚀 Instalación y Uso

### 1. Clonar el repositorio e instalar dependencias

```bash
git clone https://github.com/tu-usuario/movie-match.git
cd movie-match

pip install -r requirements.txt
```

### 2. Configurar la API Key de TMDB

1. Obtené tu token Bearer gratis en [TMDB API Settings](https://www.themoviedb.org/settings/api).
2. Copiá el archivo `.env.example` como `.env`:

```bash
cp .env.example .env
```

3. Edita `.env` agregando tu token:

```env
TMDB_BEARER_TOKEN=tu_token_bearer_aqui
```

---

## 💻 Ejecución

### Opción A: Desde Jupyter Notebook

Abrí el notebook interactivo en VSCode, JupyterLab o Google Colab:

```bash
jupyter notebook movie_match.ipynb
```

### Opción B: Desde la línea de comandos (Python Script)

```bash
python movie_match.py --save-plot mapa_gustos.png
```

---

## 🛠️ Tecnologías Utilizadas

- **Lenguaje**: Python 3.10+
- **NLP & Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Machine Learning & Math**: `scikit-learn` (PCA, Cosine Similarity), `numpy`
- **Data & Visualization**: `pandas`, `matplotlib`
- **API Client & Secs**: `requests`, `python-dotenv`

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. ¡Sentite libre de usarlo, mejorarlo o clonarlo!
