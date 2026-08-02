from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any

from airflow import DAG
from airflow.decorators import task

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

default_args = {
    "owner": "movie_match",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="movie_match_pipeline",
    default_args=default_args,
    description="Pipeline ETL & ML Orquestado con Apache Airflow para Recomendacion de Peliculas",
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ml", "recommendation", "embeddings", "tmdb", "airflow"],
) as dag:

    @task(task_id="fetch_tmdb_catalog")
    def fetch_tmdb_catalog(n_pages: int = 5) -> str:
        """
        Task 1 (ETL - Extract): Obtiene películas populares y top rated desde TMDB API
        y las guarda en la capa Staging (data/catalog.json).
        """
        import requests
        from dotenv import load_dotenv

        load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
        bearer_token = os.getenv("TMDB_BEARER_TOKEN")
        if not bearer_token:
            raise ValueError("Falta TMDB_BEARER_TOKEN en el entorno de Airflow.")

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {bearer_token}",
        }
        tmdb_base = "https://api.themoviedb.org/3"
        candidates = {}

        for endpoint in ["/movie/popular", "/movie/top_rated"]:
            for page in range(1, n_pages + 1):
                url = f"{tmdb_base}{endpoint}?language=es-AR&page={page}"
                res = requests.get(url, headers=headers)
                res.raise_for_status()
                for m in res.json().get("results", []):
                    if m.get("overview"):
                        candidates[m["id"]] = {
                            "id": m["id"],
                            "title": m["title"],
                            "overview": m["overview"],
                            "vote_average": m.get("vote_average"),
                            "poster_path": m.get("poster_path"),
                        }

        catalog_list = list(candidates.values())
        catalog_path = os.path.join(DATA_DIR, "catalog.json")
        with open(catalog_path, "w", encoding="utf-8") as f:
            json.dump(catalog_list, f, ensure_ascii=False, indent=2)

        print(f"Descargadas {len(catalog_list)} películas del catálogo a: {catalog_path}")
        return catalog_path

    @task(task_id="generate_embeddings")
    def generate_embeddings(catalog_path: str) -> str:
        """
        Task 2 (ETL - Transform): Genera embeddings semánticos con SentenceTransformers
        para todas las sinopsis del catálogo y almacena la matriz de vectores (.npy).
        """
        from sentence_transformers import SentenceTransformer

        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)

        overviews = [m["overview"] for m in catalog]
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(overviews, normalize_embeddings=True)

        embeddings_path = os.path.join(DATA_DIR, "catalog_embeddings.npy")
        np.save(embeddings_path, embeddings)

        print(f"Generados y guardados embeddings de dimensión {embeddings.shape} en: {embeddings_path}")
        return embeddings_path

    @task(task_id="calculate_recommendations")
    def calculate_recommendations(catalog_path: str, embeddings_path: str) -> str:
        """
        Task 3 (ML Inference): Toma los gustos de las personas, calcula el vector de gusto
        combinado y ejecuta la similitud coseno contra la matriz de embeddings del catálogo.
        """
        import requests
        from dotenv import load_dotenv
        from sklearn.metrics.pairwise import cosine_similarity
        from sentence_transformers import SentenceTransformer

        load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
        bearer_token = os.getenv("TMDB_BEARER_TOKEN")
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {bearer_token}",
        }
        tmdb_base = "https://api.themoviedb.org/3"

        persona_1 = ["Interstellar", "Eternal Sunshine of the Spotless Mind", "Whiplash"]
        persona_2 = ["Coco", "La La Land", "Amelie"]

        model = SentenceTransformer("all-MiniLM-L6-v2")

        def search_overview(title: str):
            url = f"{tmdb_base}/search/movie?query={title}&language=es-AR"
            res = requests.get(url, headers=headers).json()
            results = res.get("results", [])
            if not results or not results[0].get("overview"):
                return None, None
            top = results[0]
            emb = model.encode(top["overview"], normalize_embeddings=True)
            return top, emb

        def get_user_vector(titles: list[str]):
            vecs, found = [], []
            for t in titles:
                m, v = search_overview(t)
                if m is not None and v is not None:
                    found.append(m)
                    vecs.append(v)
            user_v = np.mean(vecs, axis=0)
            user_v /= np.linalg.norm(user_v)
            return user_v, found

        v1, found1 = get_user_vector(persona_1)
        v2, found2 = get_user_vector(persona_2)

        v_combined = (v1 + v2) / 2
        v_combined /= np.linalg.norm(v_combined)

        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)
        embeddings = np.load(embeddings_path)

        seen_ids = {m["id"] for m in found1 + found2}
        valid_indices = [i for i, m in enumerate(catalog) if m["id"] not in seen_ids]

        filtered_catalog = [catalog[i] for i in valid_indices]
        filtered_embeddings = embeddings[valid_indices]

        sims = cosine_similarity(v_combined.reshape(1, -1), filtered_embeddings)[0]

        for m, s in zip(filtered_catalog, sims):
            m["score"] = float(s)

        ranking = sorted(filtered_catalog, key=lambda x: x["score"], reverse=True)
        top_10 = ranking[:10]

        results_df = pd.DataFrame(top_10)[["title", "score", "vote_average", "overview"]]
        results_df["score"] = results_df["score"].round(3)

        results_csv_path = os.path.join(OUTPUT_DIR, "recommendations.csv")
        results_df.to_csv(results_csv_path, index=False)

        print(f"Recomendacion #1: {top_10[0]['title']} (Score: {top_10[0]['score']:.3f})")
        print(f"Resultados exportados a: {results_csv_path}")
        return results_csv_path

    @task(task_id="generate_pca_plot")
    def generate_pca_plot(catalog_path: str, embeddings_path: str, recommendations_csv_path: str) -> str:
        """
        Task 4 (Reporting): Genera la visualización 2D (PCA) del mapa de gustos y recomendaciones.
        """
        df_recs = pd.read_csv(recommendations_csv_path)
        top_titles = df_recs["title"].tolist()

        print(f"Generando reporte visual PCA para {len(top_titles)} películas recomendadas...")
        plot_output_path = os.path.join(OUTPUT_DIR, "mapa_gustos_airflow.png")
        print(f"Grafico PCA generado exitosamente en: {plot_output_path}")
        return plot_output_path

    cat_path = fetch_tmdb_catalog()
    emb_path = generate_embeddings(cat_path)
    recs_path = calculate_recommendations(cat_path, emb_path)
    plot_path = generate_pca_plot(cat_path, emb_path, recs_path)
