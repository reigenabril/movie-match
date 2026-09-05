"""
Configuraciones, constantes y variables de entorno para Movie Match.
"""
from __future__ import annotations
import os
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

TMDB_BEARER_TOKEN = os.getenv("TMDB_BEARER_TOKEN")
TMDB_BASE = "https://api.themoviedb.org/3"

HEADERS = {
    "accept": "application/json",
    "Authorization": f"Bearer {TMDB_BEARER_TOKEN}",
} if TMDB_BEARER_TOKEN else {}

POPULAR_PROVIDERS = [
    "Todas las plataformas",
    "Netflix",
    "Amazon Prime Video",
    "Disney Plus",
    "Max (HBO Max)",
    "Paramount Plus",
    "Apple TV+",
    "MUBI",
    "MovistarTV",
    "Claro video",
    "Mercado Play",
]

DEFAULT_GENRES: dict[str, int | None] = {
    "Todos los géneros": None,
    "Acción": 28,
    "Aventura": 12,
    "Animación": 16,
    "Comedia": 35,
    "Crimen": 80,
    "Documental": 99,
    "Drama": 18,
    "Familia": 10751,
    "Fantasía": 14,
    "Historia": 36,
    "Terror": 27,
    "Música": 10402,
    "Misterio": 9648,
    "Romance": 10749,
    "Ciencia ficción": 878,
    "Película de TV": 10770,
    "Suspense": 53,
    "Bélica": 10752,
    "Oeste": 37,
}
