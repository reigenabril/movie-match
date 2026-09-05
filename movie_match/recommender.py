"""
Motor de Recomendación de Películas basado en Embeddings Semánticos y Similitud Coseno.
"""
from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from movie_match.tmdb import (
    search_movie,
    get_candidate_pool,
    matches_provider,
    get_movie_genres,
)
from movie_match.embeddings import embed_text, embed_texts


def build_taste_vector(movie_titles: list[str], region: str = "AR") -> tuple[np.ndarray, list[dict]]:
    """Genera el vector de gusto normalizado para una lista de títulos."""
    found = []
    vectors = []
    for title in movie_titles:
        movie = search_movie(title, region=region)
        if movie and movie.get("overview"):
            found.append(movie)
            vectors.append(embed_text(movie["overview"]))

    if not vectors:
        raise ValueError(f"No se pudo generar embedding para ninguna película de la lista: {movie_titles}")

    taste_vector = np.mean(vectors, axis=0)
    taste_vector /= np.linalg.norm(taste_vector)
    return taste_vector, found


def calculate_combined_taste_vector(taste_vectors: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Calcula el vector ponderado resultante de combinar los gustos de varios participantes."""
    weighted_vecs = [w * v for w, v in zip(weights, taste_vectors)]
    combined_vector = np.sum(weighted_vecs, axis=0)
    norm = np.linalg.norm(combined_vector)
    if norm > 0:
        combined_vector /= norm
    return combined_vector


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
    candidate_pool: Optional[list[dict]] = None,
    candidate_embeddings: Optional[np.ndarray] = None,
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

    combined_vector = calculate_combined_taste_vector(taste_vectors, weights)

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
    pool = candidate_pool if candidate_pool is not None else get_candidate_pool(n_pages=n_pages_pool, region=region)
    if candidate_embeddings is not None and len(candidate_embeddings) == len(pool):
        for idx, m in enumerate(pool):
            m["_embedding"] = candidate_embeddings[idx]

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

    # Calcular embeddings y similitud coseno
    if pool_filtered and all("_embedding" in m for m in pool_filtered):
        pool_embeddings = np.array([m["_embedding"] for m in pool_filtered])
    else:
        pool_embeddings = embed_texts([m["overview"] for m in pool_filtered])
        for m, emb in zip(pool_filtered, pool_embeddings):
            m["_embedding"] = emb

    pos_sims = cosine_similarity(combined_vector.reshape(1, -1), pool_embeddings)[0]

    if disliked_vectors:
        dis_embeddings = np.array(disliked_vectors)
        dis_sims_matrix = cosine_similarity(dis_embeddings, pool_embeddings)
        max_dis_sims = np.max(dis_sims_matrix, axis=0)
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
        m_vec = m.get("_embedding")
        if m_vec is None:
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
