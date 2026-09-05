"""
Componentes de presentación para visualización de recomendaciones, tablas y gráficos.
"""
from __future__ import annotations
from typing import List, Dict, Any
import pandas as pd
import streamlit as st
from movie_match.visualization import plot_taste_map


def render_active_filters_banner(filters_info: list[str]):
    """Muestra un banner resumen de los filtros activos."""
    if filters_info:
        st.info("🍿 **Filtros y Configuración Activa:**\n- " + "\n- ".join(filters_info))


def render_recommendation_cards(top_recs: list[dict]):
    """Muestra tarjetas enriquecidas con pósters, badges y detalles de las mejores recomendaciones."""
    st.subheader("🎉 Top Recomendaciones para el Grupo")
    n_display = min(4, len(top_recs))
    if n_display <= 0:
        st.warning("No se encontraron recomendaciones con los filtros seleccionados.")
        return

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
            year_badge = (
                f" ({m['release_date'][:4]})"
                if m.get("release_date") and len(m["release_date"]) >= 4
                else ""
            )
            st.markdown(f"**{m['title']}**{year_badge}")
            match_pct = int(m.get("score", 0.0) * 100)
            st.markdown(
                f"<span class='score-badge'>Match Grupo: {match_pct}%</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"⭐ Rating TMDB: **{m.get('vote_average', 'N/A')}** / 10")

            g_list = m.get("genres", [])
            if g_list:
                st.caption(f"🎭 **Géneros:** {', '.join(g_list)}")

            provs = m.get("providers", [])
            prov_text = ", ".join(provs) if provs else "Sin streaming"
            st.caption(f"📺 **Plataformas:** {prov_text}")

            if m.get("closest_movie"):
                st.caption(f"💡 **Afinidad:** {m['closest_movie']} *({m.get('closest_person', '')})*")

    st.write("---")


def render_results_table(df_recs: pd.DataFrame):
    """Muestra la tabla interactiva de datos completa con nombres amigables."""
    st.subheader("📋 Tabla Completa de Resultados")
    column_renames = {
        "title": "Título",
        "score": "Match Grupo",
        "vote_average": "Calificación TMDB",
        "genres": "Géneros",
        "providers": "Plataforma(s)",
        "overview": "Sinopsis",
    }
    st.dataframe(
        df_recs.rename(columns=column_renames),
        use_container_width=True,
    )
    st.write("---")


def render_pca_plot(extra_data: dict):
    """Renderiza el gráfico 2D PCA con el mapa de gustos y recomendaciones."""
    st.subheader("📊 Mapa del Espacio Vectorial de Gustos (PCA 2D)")
    fig = plot_taste_map(extra_data)
    st.pyplot(fig)
