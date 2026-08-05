from __future__ import annotations
import os
import argparse
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

from concurrent.futures import ThreadPoolExecutor

from functools import lru_cache

# Cargar variables de entorno desde .env
load_dotenv()

TMDB_BEARER_TOKEN = os.getenv("TMDB_BEARER_TOKEN")
TMDB_BASE = "https://api.themoviedb.org/3"

if TMDB_BEARER_TOKEN:
    HEADERS = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_BEARER_TOKEN}",
    }
else:
    HEADERS = {}

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

DEFAULT_GENRES = {
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

@lru_cache(maxsize=16)
def get_movie_genres() -> dict[str, int | None]:
    """Obtiene el mapeo de nombres de géneros a IDs desde TMDB (con fallback local)."""
    try:
        data = tmdb_get("/genre/movie/list")
        genres_list = data.get("genres", [])
        mapping = {"Todos los géneros": None}
        for g in genres_list:
            mapping[g["name"].capitalize()] = g["id"]
        return mapping
    except Exception:
        return DEFAULT_GENRES

def get_genre_id_to_name_map() -> dict[int, str]:
    """Retorna un mapeo inverso de ID a Nombre de género."""
    genres_map = get_movie_genres()
    id_map = {}
    for name, g_id in genres_map.items():
        if g_id is not None:
            id_map[g_id] = name
    return id_map


# Modelo de embeddings de sinopsis
_model = None

@lru_cache(maxsize=1)
def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

@lru_cache(maxsize=2048)
def embed_text(text: str) -> np.ndarray:
    model = get_embedding_model()
    return model.encode(text, normalize_embeddings=True)

def tmdb_get(endpoint: str, params: dict = None) -> dict:
    if not TMDB_BEARER_TOKEN:
        raise ValueError(
            "Falta la variable TMDB_BEARER_TOKEN. Crea un archivo .env basado en .env.example con tu token de TMDB."
        )
    params = params or {}
    params.setdefault("language", "es-AR")
    response = requests.get(f"{TMDB_BASE}{endpoint}", params=params, headers=HEADERS, timeout=10)
    response.raise_for_status()
    return response.json()

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
    if "netflix" in sel:
        keywords = ["netflix"]
    elif "prime" in sel or "amazon" in sel:
        keywords = ["prime", "amazon"]
    elif "disney" in sel:
        keywords = ["disney"]
    elif "hbo" in sel or "max" in sel:
        keywords = ["hbo", "max"]
    elif "paramount" in sel:
        keywords = ["paramount"]
    elif "apple" in sel:
        keywords = ["apple"]
    elif "mubi" in sel:
        keywords = ["mubi"]
    elif "movistar" in sel:
        keywords = ["movistar"]
    elif "claro" in sel:
        keywords = ["claro"]
    elif "mercado" in sel:
        keywords = ["mercado"]
    else:
        keywords = [sel]

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
        "providers": providers,
    }

def build_taste_vector(movie_titles: list[str], region: str = "AR") -> tuple[np.ndarray, list[dict]]:
    """Genera el vector de gusto normalizado para una lista de títulos."""
    found = []
    vectors = []
    for title in movie_titles:
        movie = search_movie(title, region=region)
        if movie and movie["overview"]:
            found.append(movie)
            vectors.append(embed_text(movie["overview"]))
    
    if not vectors:
        raise ValueError(f"No se pudo generar embedding para ninguna película de la lista: {movie_titles}")
    
    taste_vector = np.mean(vectors, axis=0)
    taste_vector /= np.linalg.norm(taste_vector)
    return taste_vector, found

@lru_cache(maxsize=32)
def get_candidate_pool(n_pages: int = 5, region: str = "AR", fetch_providers: bool = True) -> list[dict]:
    """Obtiene un catálogo de películas populares y top rated desde TMDB con información de plataformas y géneros."""
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
                        "poster_path": m.get("poster_path"),
                        "genre_ids": m.get("genre_ids", []),
                        "providers": [],
                    }
    
    candidate_list = list(candidates.values())

    id_to_name = get_genre_id_to_name_map()
    for m in candidate_list:
        g_names = [id_to_name[gid] for gid in m.get("genre_ids", []) if gid in id_to_name]
        m["genres"] = g_names

    if fetch_providers:
        def _fetch_prov(m):
            m["providers"] = get_movie_watch_providers(m["id"], region=region)

        with ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(_fetch_prov, candidate_list))

    return candidate_list

def recommend(
    people_movies: list[list[str]],
    n_recommendations: int = 10,
    n_pages_pool: int = 5,
    selected_provider: Optional[str] = None,
    selected_genre: Optional[str] = None,
    region: str = "AR",
) -> tuple[pd.DataFrame, dict]:
    """
    Calcula recomendaciones cruzando los gustos de N personas con filtros opcionales de plataforma y género.
    """
    taste_vectors = []
    all_found_movies = []
    seen_ids = set()

    for idx, titles in enumerate(people_movies, 1):
        v, found = build_taste_vector(titles, region=region)
        taste_vectors.append(v)
        all_found_movies.append(found)
        for m in found:
            seen_ids.add(m["id"])
        print(f"Persona {idx} encontró: {[m['title'] for m in found]}")

    # Punto medio entre todos los vectores de gusto
    combined_vector = np.mean(taste_vectors, axis=0)
    combined_vector /= np.linalg.norm(combined_vector)

    # Catálogo candidatas
    pool = get_candidate_pool(n_pages=n_pages_pool, region=region)
    pool_filtered = [m for m in pool if m["id"] not in seen_ids]

    # Filtrar por plataforma si fue seleccionada
    if selected_provider and selected_provider != "Todas las plataformas":
        pool_filtered = [m for m in pool_filtered if matches_provider(selected_provider, m.get("providers", []))]
        if not pool_filtered:
            print(f"[WARNING] No se encontraron películas disponibles en {selected_provider!r}.")

    # Filtrar por género si fue seleccionado
    if selected_genre and selected_genre != "Todos los géneros":
        genre_map = get_movie_genres()
        target_id = genre_map.get(selected_genre)
        if target_id is not None:
            pool_filtered = [m for m in pool_filtered if target_id in m.get("genre_ids", [])]
        else:
            pool_filtered = [m for m in pool_filtered if selected_genre in m.get("genres", [])]
        if not pool_filtered:
            print(f"[WARNING] No se encontraron películas del género {selected_genre!r}.")

    if not pool_filtered:
        filters_desc = []
        if selected_provider and selected_provider != "Todas las plataformas":
            filters_desc.append(f"plataforma '{selected_provider}'")
        if selected_genre and selected_genre != "Todos los géneros":
            filters_desc.append(f"género '{selected_genre}'")
        f_str = " y ".join(filters_desc) if filters_desc else "seleccionados"
        raise ValueError(
            f"No hay películas disponibles en el catálogo para los filtros de {f_str}. "
            "Probá seleccionando 'Todos los géneros' / 'Todas las plataformas' o ampliando las páginas de búsqueda."
        )

    # Calcular embeddings y similitud coseno
    pool_embeddings = np.array([embed_text(m["overview"]) for m in pool_filtered])
    sims = cosine_similarity(combined_vector.reshape(1, -1), pool_embeddings)[0]

    for m, s in zip(pool_filtered, sims):
        m["score"] = float(s)

    ranking = sorted(pool_filtered, key=lambda x: x["score"], reverse=True)
    top_recommendations = ranking[:n_recommendations]

    df_results = pd.DataFrame(top_recommendations)
    if "providers" in df_results.columns:
        df_results["providers_str"] = df_results["providers"].apply(
            lambda p: ", ".join(p) if isinstance(p, list) and p else "No disponible en streaming"
        )
    else:
        df_results["providers_str"] = "No disponible en streaming"

    if "genres" in df_results.columns:
        df_results["genres_str"] = df_results["genres"].apply(
            lambda g: ", ".join(g) if isinstance(g, list) and g else "Sin género"
        )
    else:
        df_results["genres_str"] = "Sin género"

    df_results = df_results[["title", "score", "vote_average", "genres_str", "providers_str", "overview"]]
    df_results.rename(columns={"providers_str": "providers", "genres_str": "genres"}, inplace=True)
    df_results["score"] = df_results["score"].round(3)

    extra_data = {
        "taste_vectors": taste_vectors,
        "combined_vector": combined_vector,
        "all_found_movies": all_found_movies,
        "top_recommendations": top_recommendations,
        "selected_provider": selected_provider,
        "selected_genre": selected_genre,
        "region": region,
    }

    return df_results, extra_data

def plot_taste_map(extra_data: dict, save_path: str = None, ax: plt.Axes = None):
    """Genera el gráfico 2D PCA con el mapa de gustos y las recomendaciones."""
    taste_vectors = extra_data["taste_vectors"]
    combined_vector = extra_data["combined_vector"]
    all_found_movies = extra_data["all_found_movies"]
    top_recommendations = extra_data["top_recommendations"]

    all_vecs = []
    labels_info = []

    for idx, found in enumerate(all_found_movies):
        for m in found:
            all_vecs.append(embed_text(m["overview"]))
            labels_info.append((f"P{idx+1}", m["title"]))

    all_vecs.append(combined_vector)
    labels_info.append(("COMBINED", "Gusto combinado"))

    for m in top_recommendations:
        all_vecs.append(embed_text(m["overview"]))
        labels_info.append(("REC", m["title"]))

    all_vecs = np.array(all_vecs)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_vecs)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = ax.get_figure()

    curr_idx = 0
    colors = ["#4C72B0", "#DD8452", "#9370DB", "#E6550D"]

    for p_idx, found in enumerate(all_found_movies):
        color = colors[p_idx % len(colors)]
        n_m = len(found)
        p_coords = coords[curr_idx : curr_idx + n_m]
        ax.scatter(p_coords[:, 0], p_coords[:, 1], color=color, s=100, label=f"Persona {p_idx+1}")
        for i, m in enumerate(found):
            ax.annotate(m["title"], p_coords[i], fontsize=8, alpha=0.9)
        curr_idx += n_m

    # Combined vector
    comb_coord = coords[curr_idx : curr_idx + 1]
    ax.scatter(comb_coord[:, 0], comb_coord[:, 1], color="black", marker="X", s=220, label="Gusto combinado")
    curr_idx += 1

    # Recommendations
    n_recs = len(top_recommendations)
    rec_coords = coords[curr_idx : curr_idx + n_recs]
    ax.scatter(rec_coords[:, 0], rec_coords[:, 1], color="#55A868", s=60, alpha=0.7, label="Recomendaciones")

    # Resaltar la #1
    if top_recommendations:
        ax.annotate(top_recommendations[0]["title"], rec_coords[0], fontsize=9, weight="bold", color="#1B5E20")

    ax.legend()
    title_suffix = ""
    if extra_data.get('selected_provider') and extra_data.get('selected_provider') != "Todas las plataformas":
        title_suffix += f" (Plataforma: {extra_data.get('selected_provider')})"
    if extra_data.get('selected_genre') and extra_data.get('selected_genre') != "Todos los géneros":
        title_suffix += f" (Género: {extra_data.get('selected_genre')})"

    ax.set_title(f"Movie Match - Mapa de Gustos y Recomendaciones en Espacio de Embeddings (PCA 2D){title_suffix}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle="--", alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Grafico guardado en: {save_path}")

    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Movie Match - Recomendador por embeddings")
    parser.add_argument("--save-plot", type=str, help="Ruta para guardar el gráfico PCA generado")
    parser.add_argument("--demo", action="store_true", help="Ejecutar con datos de prueba predeterminados")
    parser.add_argument("--provider", type=str, default="Todas las plataformas", help="Filtrar por plataforma (ej: Netflix, Disney Plus, Max)")
    parser.add_argument("--genre", type=str, default="Todos los géneros", help="Filtrar por género (ej: Acción, Comedia, Ciencia ficción)")
    parser.add_argument("--region", type=str, default="AR", help="Código de país para disponibilidad (ej: AR, ES, MX, US)")
    args = parser.parse_args()

    print("Movie Match - Recomendador por Embeddings Semánticos\n")

    if args.demo:
        p1 = ["Interstellar", "Eternal Sunshine of the Spotless Mind", "Whiplash"]
        p2 = ["Coco", "La La Land", "Amelie"]
    else:
        print("Ingresá títulos de películas separadas por comas (ejemplo: Matrix, Inception, Avatar)\n")
        raw_p1 = input("Persona 1 - ¿Qué películas te gustan?: ")
        raw_p2 = input("Persona 2 - ¿Qué películas te gustan?: ")

        p1 = [t.strip() for t in raw_p1.split(",") if t.strip()]
        p2 = [t.strip() for t in raw_p2.split(",") if t.strip()]

        if not p1 or not p2:
            print("\nUsando datos de prueba por defecto ya que no ingresaste suficientes películas...\n")
            p1 = ["Interstellar", "Eternal Sunshine of the Spotless Mind", "Whiplash"]
            p2 = ["Coco", "La La Land", "Amelie"]

    print(f"\nProcesando gustos y consultando TMDB (Plataforma: {args.provider}, Género: {args.genre}, País: {args.region})...")
    df_recs, extra = recommend([p1, p2], selected_provider=args.provider, selected_genre=args.genre, region=args.region)
    
    print("\nTop Recomendaciones:")
    print(df_recs.to_string(index=False))

    save_plot_path = args.save_plot or "mapa_gustos.png"
    plot_taste_map(extra, save_path=save_plot_path)


