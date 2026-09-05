"""
Movie Match — AI Movie Recommender & Data Pipeline
"""
from movie_match.config import (
    POPULAR_PROVIDERS,
    DEFAULT_GENRES,
    PROJECT_ROOT,
    DATA_DIR,
    OUTPUT_DIR,
)
from movie_match.tmdb import (
    tmdb_get,
    get_movie_genres,
    get_genre_id_to_name_map,
    get_movie_watch_providers,
    matches_provider,
    search_movie,
    get_candidate_pool,
    fetch_and_save_catalog,
)
from movie_match.embeddings import (
    get_embedding_model,
    embed_text,
    embed_texts,
)
from movie_match.recommender import (
    build_taste_vector,
    calculate_combined_taste_vector,
    recommend,
)
from movie_match.visualization import (
    plot_taste_map,
)

__all__ = [
    "POPULAR_PROVIDERS",
    "DEFAULT_GENRES",
    "PROJECT_ROOT",
    "DATA_DIR",
    "OUTPUT_DIR",
    "tmdb_get",
    "get_movie_genres",
    "get_genre_id_to_name_map",
    "get_movie_watch_providers",
    "matches_provider",
    "search_movie",
    "get_candidate_pool",
    "fetch_and_save_catalog",
    "get_embedding_model",
    "embed_text",
    "embed_texts",
    "build_taste_vector",
    "calculate_combined_taste_vector",
    "recommend",
    "plot_taste_map",
]
