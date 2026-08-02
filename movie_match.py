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

# Modelo de embeddings de sinopsis
_model = None

def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

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
    response = requests.get(f"{TMDB_BASE}{endpoint}", params=params, headers=HEADERS)
    response.raise_for_status()
    return response.json()

def search_movie(title: str) -> dict | None:
    """Busca una película por título y devuelve datos básicos + overview."""
    data = tmdb_get("/search/movie", {"query": title})
    results = data.get("results", [])
    if not results:
        print(f"⚠️  No se encontraron resultados para: {title!r}")
        return None
    top = results[0]
    return {
        "id": top["id"],
        "title": top["title"],
        "overview": top.get("overview", ""),
        "release_date": top.get("release_date", ""),
        "vote_average": top.get("vote_average"),
        "poster_path": top.get("poster_path"),
    }

def build_taste_vector(movie_titles: list[str]) -> tuple[np.ndarray, list[dict]]:
    """Genera el vector de gusto normalizado para una lista de títulos."""
    found = []
    vectors = []
    for title in movie_titles:
        movie = search_movie(title)
        if movie and movie["overview"]:
            found.append(movie)
            vectors.append(embed_text(movie["overview"]))
    
    if not vectors:
        raise ValueError(f"No se pudo generar embedding para ninguna película de la lista: {movie_titles}")
    
    taste_vector = np.mean(vectors, axis=0)
    taste_vector /= np.linalg.norm(taste_vector)
    return taste_vector, found

def get_candidate_pool(n_pages: int = 5) -> list[dict]:
    """Obtiene un catálogo de películas populares y top rated desde TMDB."""
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
                    }
    return list(candidates.values())

def recommend(
    people_movies: list[list[str]],
    n_recommendations: int = 10,
    n_pages_pool: int = 5
) -> tuple[pd.DataFrame, dict]:
    """
    Calcula recomendaciones cruzando los gustos de N personas.
    """
    taste_vectors = []
    all_found_movies = []
    seen_ids = set()

    for idx, titles in enumerate(people_movies, 1):
        v, found = build_taste_vector(titles)
        taste_vectors.append(v)
        all_found_movies.append(found)
        for m in found:
            seen_ids.add(m["id"])
        print(f"Persona {idx} encontró: {[m['title'] for m in found]}")

    # Punto medio entre todos los vectores de gusto
    combined_vector = np.mean(taste_vectors, axis=0)
    combined_vector /= np.linalg.norm(combined_vector)

    # Catálogo candidatas
    pool = get_candidate_pool(n_pages=n_pages_pool)
    pool_filtered = [m for m in pool if m["id"] not in seen_ids]

    # Calcular embeddings y similitud coseno
    pool_embeddings = np.array([embed_text(m["overview"]) for m in pool_filtered])
    sims = cosine_similarity(combined_vector.reshape(1, -1), pool_embeddings)[0]

    for m, s in zip(pool_filtered, sims):
        m["score"] = float(s)

    ranking = sorted(pool_filtered, key=lambda x: x["score"], reverse=True)
    top_recommendations = ranking[:n_recommendations]

    df_results = pd.DataFrame(top_recommendations)[["title", "score", "vote_average", "overview"]]
    df_results["score"] = df_results["score"].round(3)

    extra_data = {
        "taste_vectors": taste_vectors,
        "combined_vector": combined_vector,
        "all_found_movies": all_found_movies,
        "top_recommendations": top_recommendations,
    }

    return df_results, extra_data

def plot_taste_map(extra_data: dict, save_path: str = None):
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

    plt.figure(figsize=(10, 7))

    curr_idx = 0
    colors = ["#4C72B0", "#DD8452", "#9370DB", "#E6550D"]

    for p_idx, found in enumerate(all_found_movies):
        color = colors[p_idx % len(colors)]
        n_m = len(found)
        p_coords = coords[curr_idx : curr_idx + n_m]
        plt.scatter(p_coords[:, 0], p_coords[:, 1], color=color, s=100, label=f"Persona {p_idx+1}")
        for i, m in enumerate(found):
            plt.annotate(m["title"], p_coords[i], fontsize=8, alpha=0.9)
        curr_idx += n_m

    # Combined vector
    comb_coord = coords[curr_idx : curr_idx + 1]
    plt.scatter(comb_coord[:, 0], comb_coord[:, 1], color="black", marker="X", s=220, label="Gusto combinado")
    curr_idx += 1

    # Recommendations
    n_recs = len(top_recommendations)
    rec_coords = coords[curr_idx : curr_idx + n_recs]
    plt.scatter(rec_coords[:, 0], rec_coords[:, 1], color="#55A868", s=60, alpha=0.7, label="Recomendaciones")

    # Resaltar la #1
    plt.annotate(top_recommendations[0]["title"], rec_coords[0], fontsize=9, weight="bold", color="#1B5E20")

    plt.legend()
    plt.title("Movie Match — Mapa de Gustos y Recomendaciones en Espacio de Embeddings (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"📷 Gráfico guardado en: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Movie Match — Recomendador por embeddings")
    parser.add_argument("--save-plot", type=str, help="Ruta para guardar el gráfico PCA generado")
    parser.add_argument("--demo", action="store_true", help="Ejecutar con datos de prueba predeterminados")
    args = parser.parse_args()

    print("🎬 Movie Match — Recomendador por Embeddings Semánticos\n")

    if args.demo:
        p1 = ["Interstellar", "Eternal Sunshine of the Spotless Mind", "Whiplash"]
        p2 = ["Coco", "La La Land", "Amelie"]
    else:
        print("💡 Ingresá títulos de películas separadas por comas (ejemplo: Matrix, Inception, Avatar)\n")
        raw_p1 = input("👤 Persona 1 — ¿Qué películas te gustan?: ")
        raw_p2 = input("👤 Persona 2 — ¿Qué películas te gustan?: ")

        p1 = [t.strip() for t in raw_p1.split(",") if t.strip()]
        p2 = [t.strip() for t in raw_p2.split(",") if t.strip()]

        if not p1 or not p2:
            print("\n⚠️ Usando datos de prueba por defecto ya que no ingresaste suficientes películas...\n")
            p1 = ["Interstellar", "Eternal Sunshine of the Spotless Mind", "Whiplash"]
            p2 = ["Coco", "La La Land", "Amelie"]

    print("\n🔍 Procesando gustos y consultando TMDB...")
    df_recs, extra = recommend([p1, p2])
    
    print("\n🏆 Top Recomendaciones:")
    print(df_recs.to_string(index=False))

    save_plot_path = args.save_plot or "mapa_gustos.png"
    plot_taste_map(extra, save_path=save_plot_path)

