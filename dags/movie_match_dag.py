from __future__ import annotations
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from airflow import DAG
from airflow.decorators import task

from movie_match.config import DATA_DIR, OUTPUT_DIR
from movie_match.tmdb import fetch_and_save_catalog
from movie_match.embeddings import embed_texts
from movie_match.recommender import recommend
from movie_match.visualization import plot_taste_map

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
        catalog_path = os.path.join(DATA_DIR, "catalog.json")
        fetch_and_save_catalog(output_path=catalog_path, n_pages=n_pages, region="AR")
        print(f"Catálogo descargado y guardado en: {catalog_path}")
        return catalog_path

    @task(task_id="generate_embeddings")
    def generate_embeddings(catalog_path: str) -> str:
        """
        Task 2 (ETL - Transform): Genera embeddings semánticos con SentenceTransformers
        para todas las sinopsis del catálogo y almacena la matriz de vectores (.npy).
        """
        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)

        overviews = [m["overview"] for m in catalog]
        embeddings = embed_texts(overviews)

        embeddings_path = os.path.join(DATA_DIR, "catalog_embeddings.npy")
        np.save(embeddings_path, embeddings)

        print(f"Generados y guardados embeddings de dimensión {embeddings.shape} en: {embeddings_path}")
        return embeddings_path

    @task(task_id="calculate_recommendations")
    def calculate_recommendations(catalog_path: str, embeddings_path: str) -> dict:
        """
        Task 3 (ML Inference): Toma los gustos de las personas, calcula el vector de gusto
        combinado y ejecuta la similitud coseno contra la matriz de embeddings precomputada.
        """
        persona_1 = ["Interstellar", "Eternal Sunshine of the Spotless Mind", "Whiplash"]
        persona_2 = ["Coco", "La La Land", "Amelie"]

        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)

        embeddings = np.load(embeddings_path)

        df_results, extra_data = recommend(
            people_movies=[persona_1, persona_2],
            people_names=["Persona 1", "Persona 2"],
            candidate_pool=catalog,
            candidate_embeddings=embeddings,
            n_recommendations=10,
            region="AR",
        )

        results_csv_path = os.path.join(OUTPUT_DIR, "recommendations.csv")
        df_results.to_csv(results_csv_path, index=False)

        context_path = os.path.join(OUTPUT_DIR, "recommendations_context.pkl")
        with open(context_path, "wb") as f:
            pickle.dump(extra_data, f)

        print(f"Recomendacion #1: {df_results.iloc[0]['title']} (Score: {df_results.iloc[0]['score']})")
        print(f"Resultados exportados a: {results_csv_path}")
        print(f"Contexto de recomendación guardado en: {context_path}")
        return {"csv_path": results_csv_path, "context_path": context_path}

    @task(task_id="generate_pca_plot")
    def generate_pca_plot(pipeline_output: dict) -> str:
        """
        Task 4 (Reporting): Genera la visualización 2D (PCA) del mapa de gustos y recomendaciones
        a partir del contexto calculado en la inferencia previa.
        """
        context_path = pipeline_output["context_path"]
        with open(context_path, "rb") as f:
            extra_data = pickle.load(f)

        plot_output_path = os.path.join(OUTPUT_DIR, "mapa_gustos_airflow.png")
        plot_taste_map(extra_data, save_path=plot_output_path)
        print(f"Grafico PCA generado exitosamente en: {plot_output_path}")
        return plot_output_path

    cat_path = fetch_tmdb_catalog()
    emb_path = generate_embeddings(cat_path)
    recs_output = calculate_recommendations(cat_path, emb_path)
    plot_path = generate_pca_plot(recs_output)
