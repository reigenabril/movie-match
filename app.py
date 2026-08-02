from __future__ import annotations
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from movie_match import recommend, plot_taste_map, search_movie

load_dotenv()

st.set_page_config(
    page_title="Movie Match - AI Movie Recommender",
    page_icon="film",
    layout="wide",
)

# Estilos CSS personalizados
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 3em;
    }
    .movie-card {
        background-color: #1e222d;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #2e3440;
    }
    .score-badge {
        background-color: #43a047;
        color: white;
        padding: 4px 8px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 0.9em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Movie Match - AI Movie Recommender")
st.write(
    "Ingresá las películas favoritas de dos personas para encontrar el punto medio de gusto y recomendar la película ideal."
)

st.sidebar.header("Configuración")
tmdb_token = os.getenv("TMDB_BEARER_TOKEN")

if not tmdb_token:
    st.sidebar.error("Falta TMDB_BEARER_TOKEN en el archivo .env")
else:
    st.sidebar.success("Token de TMDB conectado correctamente")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Persona 1")
    p1_input = st.text_area(
        "Ingresá películas (separadas por coma)",
        value="Interstellar, Eternal Sunshine of the Spotless Mind, Whiplash",
        height=100,
        key="p1",
    )

with col2:
    st.subheader("Persona 2")
    p2_input = st.text_area(
        "Ingresá películas (separadas por coma)",
        value="Coco, La La Land, Amelie",
        height=100,
        key="p2",
    )

p1_list = [t.strip() for t in p1_input.split(",") if t.strip()]
p2_list = [t.strip() for t in p2_input.split(",") if t.strip()]

st.write("---")

if st.button("Buscar Recomendaciones"):
    if not p1_list or not p2_list:
        st.warning("Ingresá al menos una película para cada persona.")
    else:
        with st.spinner("Analizando sinopsis y generando embeddings semánticos..."):
            try:
                df_recs, extra = recommend([p1_list, p2_list])
                top_recs = extra["top_recommendations"]

                st.subheader("Top Recomendaciones Combinadas")

                # Mostrar las mejores 4 en columnas con poster
                rec_cols = st.columns(min(4, len(top_recs)))
                for idx, col in enumerate(rec_cols):
                    m = top_recs[idx]
                    with col:
                        poster_url = (
                            f"https://image.tmdb.org/t/p/w500{m['poster_path']}"
                            if m.get("poster_path")
                            else "https://via.placeholder.com/500x750?text=No+Poster"
                        )
                        st.image(poster_url, use_column_width=True)
                        st.markdown(f"**{m['title']}**")
                        match_pct = int(m['score'] * 100)
                        st.markdown(f"<span class='score-badge'>Match: {match_pct}%</span>", unsafe_allow_html=True)
                        st.caption(f"Rating: {m.get('vote_average', 'N/A')} / 10")

                st.write("---")
                st.subheader("Tabla Completa de Resultados")
                st.dataframe(
                    df_recs.rename(
                        columns={
                            "title": "Título",
                            "score": "Score de Coincidencia",
                            "vote_average": "Calificación TMDB",
                            "overview": "Sinopsis",
                        }
                    ),
                    use_container_width=True,
                )

                st.write("---")
                st.subheader("Mapa de Gustos en Espacio Vectorial (PCA 2D)")

                fig, ax = plt.subplots(figsize=(10, 6))
                plot_taste_map(extra)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Ocurrió un error al procesar las recomendaciones: {e}")
