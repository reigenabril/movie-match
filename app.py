from __future__ import annotations
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from movie_match import recommend, plot_taste_map, search_movie, POPULAR_PROVIDERS, get_movie_genres

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
    .provider-badge {
        background-color: #1e88e5;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.8em;
        margin-top: 5px;
        display: inline-block;
    }
    .genre-badge {
        background-color: #ab47bc;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.8em;
        margin-top: 3px;
        display: inline-block;
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

st.sidebar.markdown("---")
st.sidebar.subheader("🍿 Plataforma de Streaming")
selected_provider = st.sidebar.selectbox(
    "Filtrar por Plataforma (opcional)",
    options=POPULAR_PROVIDERS,
    index=0,
    help="Si seleccionás una plataforma, solo se recomendarán películas disponibles en ella."
)

region_options = {
    "Argentina (AR)": "AR",
    "España (ES)": "ES",
    "México (MX)": "MX",
    "Estados Unidos (US)": "US",
}
selected_region_label = st.sidebar.selectbox(
    "País de disponibilidad",
    options=list(region_options.keys()),
    index=0
)
selected_region = region_options[selected_region_label]

@st.cache_data(show_spinner=False, ttl=3600)
def cached_get_genres():
    return get_movie_genres()

@st.cache_data(show_spinner=False, ttl=3600)
def cached_recommend(p1_tuple: tuple[str, ...], p2_tuple: tuple[str, ...], selected_provider: str, selected_genre: str, region: str):
    return recommend(
        [list(p1_tuple), list(p2_tuple)],
        selected_provider=selected_provider,
        selected_genre=selected_genre,
        region=region,
    )

st.sidebar.markdown("---")
st.sidebar.subheader("🎭 Género de Película")
genres_dict = cached_get_genres()
selected_genre = st.sidebar.selectbox(
    "Filtrar por Género (opcional)",
    options=list(genres_dict.keys()),
    index=0,
    help="Si seleccionás un género, solo se recomendarán películas pertenecientes a esa categoría."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Año de Estreno")
min_year, max_year = st.sidebar.slider(
    "Rango de Años de Estreno",
    min_value=1950,
    max_value=2026,
    value=(1970, 2026),
    step=1,
    help="Filtra películas estrenadas dentro de este rango de años."
)

st.sidebar.markdown("---")
st.sidebar.subheader("⭐ Calificación Mínima (TMDB)")
min_vote = st.sidebar.slider(
    "Puntuación Mínima (0 a 10)",
    min_value=0.0,
    max_value=10.0,
    value=6.0,
    step=0.5,
    help="Solo se recomendarán películas con una calificación media mayor o igual a este valor en TMDB."
)

@st.cache_data(show_spinner=False, ttl=3600)
def cached_recommend(
    p1_tuple: tuple[str, ...],
    p2_tuple: tuple[str, ...],
    selected_provider: str,
    selected_genre: str,
    min_year: int,
    max_year: int,
    min_vote: float,
    region: str,
):
    return recommend(
        [list(p1_tuple), list(p2_tuple)],
        selected_provider=selected_provider,
        selected_genre=selected_genre,
        min_year=min_year,
        max_year=max_year,
        min_vote_average=min_vote,
        region=region,
    )

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
        with st.spinner("Analizando sinopsis y filtrando catálogo..."):
            try:
                df_recs, extra = cached_recommend(
                    tuple(p1_list),
                    tuple(p2_list),
                    selected_provider=selected_provider,
                    selected_genre=selected_genre,
                    min_year=min_year,
                    max_year=max_year,
                    min_vote=min_vote,
                    region=selected_region,
                )
                top_recs = extra["top_recommendations"]

                filters_info = []
                if selected_provider != "Todas las plataformas":
                    filters_info.append(f"Plataforma: **{selected_provider}** ({selected_region_label})")
                if selected_genre != "Todos los géneros":
                    filters_info.append(f"Género: **{selected_genre}**")
                if min_year > 1950 or max_year < 2026:
                    filters_info.append(f"Años: **{min_year} - {max_year}**")
                if min_vote > 0.0:
                    filters_info.append(f"Nota mín.: **⭐ {min_vote}**")

                if filters_info:
                    st.info("🍿 **Filtros activos:** " + " | ".join(filters_info))

                st.subheader("Top Recomendaciones Combinadas")

                # Mostrar las mejores en columnas con poster, géneros y plataformas
                n_display = min(4, len(top_recs))
                if n_display > 0:
                    rec_cols = st.columns(n_display)
                    for idx, col in enumerate(rec_cols):
                        m = top_recs[idx]
                        with col:
                            poster_url = (
                                f"https://image.tmdb.org/t/p/w500{m['poster_path']}"
                                if m.get("poster_path")
                                else "https://via.placeholder.com/500x750?text=No+Poster"
                            )
                            st.image(poster_url, use_container_width=True)
                            year_badge = f" ({m['release_date'][:4]})" if m.get("release_date") and len(m["release_date"]) >= 4 else ""
                            st.markdown(f"**{m['title']}**{year_badge}")
                            match_pct = int(m['score'] * 100)
                            st.markdown(f"<span class='score-badge'>Match: {match_pct}%</span>", unsafe_allow_html=True)
                            st.caption(f"⭐ Rating: **{m.get('vote_average', 'N/A')}** / 10")
                            
                            g_list = m.get("genres", [])
                            if g_list:
                                st.caption(f"🎭 **Géneros:** {', '.join(g_list)}")

                            provs = m.get("providers", [])
                            prov_text = ", ".join(provs) if provs else "Sin streaming"
                            st.caption(f"📺 **Plataformas:** {prov_text}")

                st.write("---")
                st.subheader("Tabla Completa de Resultados")
                st.dataframe(
                    df_recs.rename(
                        columns={
                            "title": "Título",
                            "score": "Score de Coincidencia",
                            "vote_average": "Calificación TMDB",
                            "genres": "Géneros",
                            "providers": "Plataforma(s)",
                            "overview": "Sinopsis",
                        }
                    ),
                    use_container_width=True,
                )

                st.write("---")
                st.subheader("Mapa de Gustos en Espacio Vectorial (PCA 2D)")

                fig = plot_taste_map(extra)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Ocurrió un error al procesar las recomendaciones: {e}")


