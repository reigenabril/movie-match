"""
Cliente e integración con The Movie Database (TMDB) API v3.
"""
from __future__ import annotations
import json
import os
import requests
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any

from movie_match.config import (
    TMDB_BASE,
    HEADERS,
    TMDB_BEARER_TOKEN,
    DEFAULT_GENRES,
    DATA_DIR,
)


def tmdb_get(endpoint: str, params: dict = None) -> dict:
    """Realiza una petición GET a la API de TMDB."""
    if not TMDB_BEARER_TOKEN:
        raise ValueError(
            "Falta la variable TMDB_BEARER_TOKEN. Crea un archivo .env basado en .env.example con tu token de TMDB."
        )
    params = params or {}
    params.setdefault("language", "es-AR")
    response = requests.get(f"{TMDB_BASE}{endpoint}", params=params, headers=HEADERS, timeout=10)
    response.raise_for_status()
    return response.json()


@lru_cache(maxsize=16)
def get_movie_genres() -> dict[str, int | None]:
    """Obtiene el mapeo de nombres de géneros a IDs desde TMDB (con fallback local)."""
    try:
        data = tmdb_get("/genre/movie/list")
        genres_list = data.get("genres", [])
        mapping: dict[str, int | None] = {"Todos los géneros": None}
        for g in genres_list:
            mapping[g["name"].capitalize()] = g["id"]
        return mapping
    except Exception:
        return DEFAULT_GENRES


def get_genre_id_to_name_map() -> dict[int, str]:
    """Retorna un mapeo inverso de ID a Nombre de género."""
    genres_map = get_movie_genres()
    return {g_id: name for name, g_id in genres_map.items() if g_id is not None}


@lru_cache(maxsize=2048)
def get_movie_watch_providers(movie_id: int, region: str = "AR") -> list[str]:
    """Obtiene la lista de plataformas de streaming por suscripción (flatrate) para una película."""
    try:
        data = tmdb_get(f"/movie/{movie_id}/watch/providers")
        results = data.get("results", {}).get(region, {})
        flatrate = results.get("flatrate", [])
        return [p["provider_name"] for p in flatrate if "provider_name" in p]
    except Exception:
        return []


def matches_provider(selected_provider: str, movie_providers: list[str]) -> bool:
    """Verifica si los proveedores de la película coinciden con el filtro seleccionado."""
    if not selected_provider or selected_provider == "Todas las plataformas":
        return True

    sel = selected_provider.lower()
    provider_keywords = {
        "netflix": ["netflix"],
        "prime": ["prime", "amazon"],
        "amazon": ["prime", "amazon"],
        "disney": ["disney"],
        "hbo": ["hbo", "max"],
        "max": ["hbo", "max"],
        "paramount": ["paramount"],
        "apple": ["apple"],
        "mubi": ["mubi"],
        "movistar": ["movistar"],
        "claro": ["claro"],
        "mercado": ["mercado"],
    }

    keywords = [sel]
    for key, kw_list in provider_keywords.items():
        if key in sel:
            keywords = kw_list
            break

    for p in movie_providers:
        p_lower = p.lower()
        if any(kw in p_lower for kw in keywords):
            return True
    return False


@lru_cache(maxsize=512)
def search_movie(title: str, region: str = "AR") -> dict | None:
    """Busca una película por título y devuelve datos básicos + overview + plataformas."""
    data = tmdb_get("/search/movie", {"query": title})
    results = data.get("results", [])
    if not results:
        print(f"[WARNING] No se encontraron resultados para: {title!r}")
        return None
    top = results[0]
    movie_id = top["id"]
    providers = get_movie_watch_providers(movie_id, region=region)
    return {
        "id": movie_id,
        "title": top["title"],
        "overview": top.get("overview", ""),
        "release_date": top.get("release_date", ""),
        "vote_average": top.get("vote_average"),
        "poster_path": top.get("poster_path"),
        "genre_ids": top.get("genre_ids", []),
        "providers": providers,
    }


@lru_cache(maxsize=32)
def get_candidate_pool(n_pages: int = 5, region: str = "AR", fetch_providers: bool = True) -> list[dict]:
    """Obtiene un catálogo de películas populares y top rated desde TMDB con plataformas y géneros."""
    candidates = {}
    for endpoint in ["/movie/popular", "/movie/top_rated"]:
        for page in range(1, n_pages + 1):
            data = tmdb_get(endpoint, {"page": page})
            for m in data.get("results", []):
                if m.get("overview"):
                    candidates[m["id"]] = {
                        "id": m["id"],
                        "title": m["title"],
                        "overview": m["overview"],
                        "vote_average": m.get("vote_average"),
                        "release_date": m.get("release_date", ""),
                        "poster_path": m.get("poster_path"),
                        "genre_ids": m.get("genre_ids", []),
                        "providers": [],
                    }

    candidate_list = list(candidates.values())

    id_to_name = get_genre_id_to_name_map()
    for m in candidate_list:
        m["genres"] = [id_to_name[gid] for gid in m.get("genre_ids", []) if gid in id_to_name]

    if fetch_providers:
        def _fetch_prov(m):
            m["providers"] = get_movie_watch_providers(m["id"], region=region)

        with ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(_fetch_prov, candidate_list))

    return candidate_list


def fetch_and_save_catalog(output_path: str = None, n_pages: int = 5, region: str = "AR") -> str:
    """Descarga el catálogo de TMDB con plataformas y lo guarda en formato JSON (usado por pipelines)."""
    output_path = output_path or os.path.join(DATA_DIR, "catalog.json")
    catalog_list = get_candidate_pool(n_pages=n_pages, region=region, fetch_providers=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(catalog_list, f, ensure_ascii=False, indent=2)
    return output_path
