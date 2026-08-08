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
                        "release_date": m.get("release_date", ""),
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
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    min_vote_average: float = 0.0,
    region: str = "AR",
    people_names: Optional[list[str]] = None,
    people_weights: Optional[list[float]] = None,
    disliked_movies: Optional[list[str]] = None,
    disliked_genres: Optional[list[str]] = None,
    dislike_penalty_weight: float = 0.4,
) -> tuple[pd.DataFrame, dict]:
    """
    Calcula recomendaciones cruzando los gustos de N personas (con nombres y pesos personalizados)
    y aplicando filtros de plataforma, género, año, nota mínima y veto/preferencias negativas.
    """
    n_people = len(people_movies)
    if n_people == 0:
        raise ValueError("Debes ingresar al menos los gustos de una persona.")

    names = (
        people_names
        if (people_names and len(people_names) == n_people)
        else [f"Persona {i+1}" for i in range(n_people)]
    )
    raw_weights = (
        people_weights
        if (people_weights and len(people_weights) == n_people)
        else [1.0] * n_people
    )
    total_w = sum(raw_weights) if sum(raw_weights) > 0 else 1.0
    weights = [w / total_w for w in raw_weights]

    taste_vectors = []
    all_found_movies = []
    seen_ids = set()

    for idx, (titles, name) in enumerate(zip(people_movies, names), 1):
        v, found = build_taste_vector(titles, region=region)
        taste_vectors.append(v)
        all_found_movies.append(found)
        for m in found:
            seen_ids.add(m["id"])
        print(f"[{name}] encontró: {[m['title'] for m in found]}")

    # Vector combinado con pesos por persona
    weighted_vecs = [w * v for w, v in zip(weights, taste_vectors)]
    combined_vector = np.sum(weighted_vecs, axis=0)
    combined_vector /= np.linalg.norm(combined_vector)

    # Procesar películas vetadas / desaprobadas (preferencias negativas)
    disliked_found = []
    disliked_vectors = []
    disliked_ids = set()

    if disliked_movies:
        for title in disliked_movies:
            if not title.strip():
                continue
            movie = search_movie(title.strip(), region=region)
            if movie and movie.get("overview"):
                disliked_found.append(movie)
                disliked_ids.add(movie["id"])
                disliked_vectors.append(embed_text(movie["overview"]))
        if disliked_found:
            print(f"[Preferencias Negativas] Películas vetadas cargadas: {[m['title'] for m in disliked_found]}")

    # Catálogo candidatas
    pool = get_candidate_pool(n_pages=n_pages_pool, region=region)
    pool_filtered = [m for m in pool if m["id"] not in seen_ids and m["id"] not in disliked_ids]

    # Excluir películas vetadas por género si aplica
    if disliked_genres:
        genre_map = get_movie_genres()
        disliked_genre_ids = set()
        for dg in disliked_genres:
            gid = genre_map.get(dg)
            if gid is not None:
                disliked_genre_ids.add(gid)

        def _has_disliked_genre(m):
            m_gids = set(m.get("genre_ids", []))
            if m_gids.intersection(disliked_genre_ids):
                return True
            m_gnames = set(m.get("genres", []))
            if any(dg in m_gnames for dg in disliked_genres):
                return True
            return False

        pool_filtered = [m for m in pool_filtered if not _has_disliked_genre(m)]

    # Filtrar por plataforma si fue seleccionada
    if selected_provider and selected_provider != "Todas las plataformas":
        pool_filtered = [m for m in pool_filtered if matches_provider(selected_provider, m.get("providers", []))]

    # Filtrar por género preferido si fue seleccionado
    if selected_genre and selected_genre != "Todos los géneros":
        genre_map = get_movie_genres()
        target_id = genre_map.get(selected_genre)
        if target_id is not None:
            pool_filtered = [m for m in pool_filtered if target_id in m.get("genre_ids", [])]
        else:
            pool_filtered = [m for m in pool_filtered if selected_genre in m.get("genres", [])]

    # Filtrar por calificación mínima TMDB
    if min_vote_average > 0.0:
        pool_filtered = [m for m in pool_filtered if (m.get("vote_average") or 0.0) >= min_vote_average]

    # Filtrar por rango de años de estreno
    if min_year is not None or max_year is not None:
        def _year_ok(m):
            rd = m.get("release_date", "")
            if rd and len(rd) >= 4 and rd[:4].isdigit():
                y = int(rd[:4])
                if min_year is not None and y < min_year:
                    return False
                if max_year is not None and y > max_year:
                    return False
            return True
        pool_filtered = [m for m in pool_filtered if _year_ok(m)]

    if not pool_filtered:
        filters_desc = []
        if selected_provider and selected_provider != "Todas las plataformas":
            filters_desc.append(f"plataforma '{selected_provider}'")
        if selected_genre and selected_genre != "Todos los géneros":
            filters_desc.append(f"género '{selected_genre}'")
        if min_vote_average > 0.0:
            filters_desc.append(f"puntuación mínima ⭐ {min_vote_average}")
        if min_year or max_year:
            filters_desc.append(f"rango de años {min_year or 'inicio'}-{max_year or 'actual'}")
        if disliked_genres:
            filters_desc.append(f"géneros vetados '{', '.join(disliked_genres)}'")
        f_str = ", ".join(filters_desc) if filters_desc else "seleccionados"
        raise ValueError(
            f"No hay películas disponibles en el catálogo para los filtros de {f_str}. "
            "Probá flexibilizar los filtros."
        )

    # Calcular embeddings y similitud coseno (aplicando penalización por disgustos)
    pool_embeddings = np.array([embed_text(m["overview"]) for m in pool_filtered])
    pos_sims = cosine_similarity(combined_vector.reshape(1, -1), pool_embeddings)[0]

    if disliked_vectors:
        dis_embeddings = np.array(disliked_vectors)
        dis_sims_matrix = cosine_similarity(dis_embeddings, pool_embeddings) # (n_dislikes, n_candidates)
        max_dis_sims = np.max(dis_sims_matrix, axis=0) # max sim a cualquier película vetada
        final_scores = pos_sims - (dislike_penalty_weight * np.maximum(0.0, max_dis_sims))
    else:
        final_scores = pos_sims

    for m, s, pos_s in zip(pool_filtered, final_scores, pos_sims):
        m["score"] = float(s)
        m["raw_pos_score"] = float(pos_s)

    ranking = sorted(pool_filtered, key=lambda x: x["score"], reverse=True)
    top_recommendations = ranking[:n_recommendations]

    # Calcular a qué película ingresada se parece más cada recomendación
    input_movies_flat = []
    for p_idx, (name, found_list) in enumerate(zip(names, all_found_movies), 1):
        for m_inp in found_list:
            if m_inp.get("overview"):
                input_movies_flat.append({
                    "title": m_inp["title"],
                    "person": name,
                    "vec": embed_text(m_inp["overview"])
                })

    for m in top_recommendations:
        m_vec = embed_text(m["overview"])
        best_title = "N/A"
        best_person = ""
        best_sim = -1.0
        for inp in input_movies_flat:
            sim = float(cosine_similarity(m_vec.reshape(1, -1), inp["vec"].reshape(1, -1))[0][0])
            if sim > best_sim:
                best_sim = sim
                best_title = inp["title"]
                best_person = inp["person"]
        m["closest_movie"] = best_title
        m["closest_person"] = best_person
        m["closest_similarity_str"] = f"{best_title} ({best_person})"

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

    if "release_date" in df_results.columns:
        df_results["year_str"] = df_results["release_date"].apply(
            lambda d: d[:4] if isinstance(d, str) and len(d) >= 4 else "N/A"
        )
    else:
        df_results["year_str"] = "N/A"

    df_results["closest_str"] = [m.get("closest_similarity_str", "N/A") for m in top_recommendations]

    df_results = df_results[["title", "year_str", "score", "closest_str", "vote_average", "genres_str", "providers_str", "overview"]]
    df_results.rename(columns={"providers_str": "providers", "genres_str": "genres", "year_str": "Año", "closest_str": "Más parecida a"}, inplace=True)
    df_results["score"] = df_results["score"].round(3)

    extra_data = {
        "taste_vectors": taste_vectors,
        "combined_vector": combined_vector,
        "all_found_movies": all_found_movies,
        "top_recommendations": top_recommendations,
        "selected_provider": selected_provider,
        "selected_genre": selected_genre,
        "min_year": min_year,
        "max_year": max_year,
        "min_vote_average": min_vote_average,
        "region": region,
        "people_names": names,
        "people_weights": weights,
        "disliked_found": disliked_found,
        "disliked_genres": disliked_genres,
    }

    return df_results, extra_data

def plot_taste_map(extra_data: dict, save_path: str = None, ax: plt.Axes = None):
    """Genera el gráfico 2D PCA con el mapa de gustos N-personas, películas vetadas y recomendaciones."""
    taste_vectors = extra_data["taste_vectors"]
    combined_vector = extra_data["combined_vector"]
    all_found_movies = extra_data["all_found_movies"]
    top_recommendations = extra_data["top_recommendations"]
    people_names = extra_data.get("people_names", [f"Persona {i+1}" for i in range(len(all_found_movies))])
    disliked_found = extra_data.get("disliked_found", [])

    all_vecs = []
    labels_info = []

    for idx, found in enumerate(all_found_movies):
        p_name = people_names[idx] if idx < len(people_names) else f"Persona {idx+1}"
        for m in found:
            all_vecs.append(embed_text(m["overview"]))
            labels_info.append((p_name, m["title"]))

    # Películas vetadas / disgustos
    for m in disliked_found:
        all_vecs.append(embed_text(m["overview"]))
        labels_info.append(("VETO", m["title"]))

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
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf"]

    for p_idx, found in enumerate(all_found_movies):
        color = colors[p_idx % len(colors)]
        p_name = people_names[p_idx] if p_idx < len(people_names) else f"Persona {p_idx+1}"
        n_m = len(found)
        p_coords = coords[curr_idx : curr_idx + n_m]
        ax.scatter(p_coords[:, 0], p_coords[:, 1], color=color, s=110, label=p_name, zorder=3)
        for i, m in enumerate(found):
            ax.annotate(m["title"], p_coords[i], fontsize=8, alpha=0.9, color=color, weight="bold")
        curr_idx += n_m

    # Dislikes / Vetadas
    if disliked_found:
        n_dis = len(disliked_found)
        dis_coords = coords[curr_idx : curr_idx + n_dis]
        ax.scatter(dis_coords[:, 0], dis_coords[:, 1], color="#d62728", marker="v", s=130, label="Películas Vetadas", zorder=4)
        for i, m in enumerate(disliked_found):
            ax.annotate(f"🚫 {m['title']}", dis_coords[i], fontsize=8, alpha=0.9, color="#b71c1c", weight="bold")
        curr_idx += n_dis

    # Combined vector
    comb_coord = coords[curr_idx : curr_idx + 1]
    ax.scatter(comb_coord[:, 0], comb_coord[:, 1], color="black", marker="X", s=240, label="Gusto Combinado", zorder=5)
    curr_idx += 1

    # Recommendations
    n_recs = len(top_recommendations)
    rec_coords = coords[curr_idx : curr_idx + n_recs]
    ax.scatter(rec_coords[:, 0], rec_coords[:, 1], color="#2ca02c", s=70, alpha=0.7, label="Recomendaciones", zorder=2)

    # Resaltar la #1
    if top_recommendations:
        ax.annotate(top_recommendations[0]["title"], rec_coords[0], fontsize=9, weight="bold", color="#1b5e20")

    ax.legend(loc="best")
    title_suffix = ""
    if extra_data.get('selected_provider') and extra_data.get('selected_provider') != "Todas las plataformas":
        title_suffix += f" (Plataforma: {extra_data.get('selected_provider')})"
    if extra_data.get('selected_genre') and extra_data.get('selected_genre') != "Todos los géneros":
        title_suffix += f" (Género: {extra_data.get('selected_genre')})"

    ax.set_title(f"Movie Match - Mapa de Gustos Grupo & Veto (PCA 2D){title_suffix}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle="--", alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Gráfico guardado en: {save_path}")

    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Movie Match - Recomendador por embeddings (Grupo & Veto)")
    parser.add_argument("--save-plot", type=str, help="Ruta para guardar el gráfico PCA generado")
    parser.add_argument("--demo", action="store_true", help="Ejecutar con datos de prueba predeterminados (3 personas)")
    parser.add_argument("--provider", type=str, default="Todas las plataformas", help="Filtrar por plataforma")
    parser.add_argument("--genre", type=str, default="Todos los géneros", help="Filtrar por género")
    parser.add_argument("--min-year", type=int, default=None, help="Año mínimo de estreno")
    parser.add_argument("--max-year", type=int, default=None, help="Año máximo de estreno")
    parser.add_argument("--min-vote", type=float, default=0.0, help="Puntuación mínima de TMDB (0 a 10)")
    parser.add_argument("--region", type=str, default="AR", help="Código de país para disponibilidad")
    args = parser.parse_args()

    print("Movie Match - Recomendador por Embeddings Semánticos (Modo Grupo & Preferencias Negativas)\n")

    if args.demo:
        people_movies = [
            ["Interstellar", "Whiplash"],
            ["Coco", "Amelie"],
            ["Inception", "The Dark Knight"],
        ]
        people_names = ["Ana", "Carlos", "Sofia"]
        disliked_movies = ["Twilight"]
        disliked_genres = ["Terror"]
    else:
        p1 = ["Interstellar", "Eternal Sunshine of the Spotless Mind", "Whiplash"]
        p2 = ["Coco", "La La Land", "Amelie"]
        people_movies = [p1, p2]
        people_names = ["Persona 1", "Persona 2"]
        disliked_movies = []
        disliked_genres = []

    print(f"\nProcesando gustos y consultando TMDB (Participantes: {people_names}, Veto: {disliked_movies + disliked_genres})...")
    df_recs, extra = recommend(
        people_movies,
        people_names=people_names,
        disliked_movies=disliked_movies,
        disliked_genres=disliked_genres,
        selected_provider=args.provider,
        selected_genre=args.genre,
        min_year=args.min_year,
        max_year=args.max_year,
        min_vote_average=args.min_vote,
        region=args.region,
    )
    
    print("\nTop Recomendaciones:")
    print(df_recs.to_string(index=False))

    save_plot_path = args.save_plot or "mapa_gustos.png"
    plot_taste_map(extra, save_path=save_plot_path)




