"""
Componente del sidebar lateral para filtros y configuración de la búsqueda.
"""
from __future__ import annotations
import os
import streamlit as st
from movie_match.config import POPULAR_PROVIDERS, TMDB_BEARER_TOKEN
from movie_match.tmdb import get_movie_genres


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_genres():
    return get_movie_genres()


def render_sidebar() -> dict:
    """Renderiza la barra lateral con filtros y devuelve las opciones seleccionadas."""
    st.sidebar.header("⚙️ Configuración & Filtros")

    if not TMDB_BEARER_TOKEN:
        st.sidebar.error("⚠️ Falta TMDB_BEARER_TOKEN en el archivo .env")
    else:
        st.sidebar.success("🟢 Token de TMDB conectado")

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

    genres_dict = _cached_genres()

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎭 Género Preferido")
    selected_genre = st.sidebar.selectbox(
        "Filtrar por Género (opcional)",
        options=list(genres_dict.keys()),
        index=0,
        help="Solo películas pertenecientes a esta categoría."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📅 Año de Estreno")
    min_year, max_year = st.sidebar.slider(
        "Rango de Años",
        min_value=1950,
        max_value=2026,
        value=(1970, 2026),
        step=1
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("⭐ Calificación Mínima (TMDB)")
    min_vote = st.sidebar.slider(
        "Puntuación Mínima (0 a 10)",
        min_value=0.0,
        max_value=10.0,
        value=6.0,
        step=0.5
    )

    return {
        "provider": selected_provider,
        "region": selected_region,
        "region_label": selected_region_label,
        "genre": selected_genre,
        "min_year": min_year,
        "max_year": max_year,
        "min_vote": min_vote,
        "genres_dict": genres_dict,
    }
