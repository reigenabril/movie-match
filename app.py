from __future__ import annotations
import streamlit as st
from dotenv import load_dotenv

from movie_match.recommender import recommend
from ui import (
    load_css,
    render_sidebar,
    render_group_inputs,
    render_veto_inputs,
    render_active_filters_banner,
    render_recommendation_cards,
    render_results_table,
    render_pca_plot,
)

load_dotenv()

st.set_page_config(
    page_title="Movie Match - AI Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 1. Cargar estilos visuales
load_css()

# 2. Encabezado principal
st.title("🎬 Movie Match — AI Recommender (Modo Grupo & Veto)")
st.write(
    "Encontrá la película perfecta combinando los gustos de **2 o más personas** y descartando lo que no quieran ver."
)

# 3. Barra lateral con filtros de plataforma, género, año y calificación
filters = render_sidebar()

# 4. Formulario de integrantes del grupo (nombres, películas y pesos)
people_names, people_movies_list, people_weights = render_group_inputs()

# 5. Sección de veto / preferencias negativas
available_genres = [g for g in filters["genres_dict"].keys() if g != "Todos los géneros"]
disliked_movies, disliked_genres = render_veto_inputs(available_genres=available_genres)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_recommend_group(
    people_movies_tuple: tuple[tuple[str, ...], ...],
    people_names_tuple: tuple[str, ...],
    people_weights_tuple: tuple[float, ...],
    disliked_movies_tuple: tuple[str, ...],
    disliked_genres_tuple: tuple[str, ...],
    selected_provider: str,
    selected_genre: str,
    min_year: int,
    max_year: int,
    min_vote: float,
    region: str,
):
    return recommend(
        people_movies=[list(pm) for pm in people_movies_tuple],
        people_names=list(people_names_tuple),
        people_weights=list(people_weights_tuple),
        disliked_movies=list(disliked_movies_tuple),
        disliked_genres=list(disliked_genres_tuple),
        selected_provider=selected_provider,
        selected_genre=selected_genre,
        min_year=min_year,
        max_year=max_year,
        min_vote_average=min_vote,
        region=region,
    )


# 6. Acción de recomendación y presentación de resultados
if st.button("🚀 Buscar Recomendaciones de Grupo"):
    empty_people = [people_names[idx] for idx, m in enumerate(people_movies_list) if not m]
    if empty_people:
        st.warning(f"Ingresá al menos una película favorita para: {', '.join(empty_people)}")
    else:
        with st.spinner("🧠 Calculando vector de gusto combinado y filtrando catálogo..."):
            try:
                pm_tuple = tuple(tuple(m) for m in people_movies_list)
                pn_tuple = tuple(people_names)
                pw_tuple = tuple(people_weights)
                dm_tuple = tuple(disliked_movies)
                dg_tuple = tuple(disliked_genres)

                df_recs, extra = cached_recommend_group(
                    pm_tuple,
                    pn_tuple,
                    pw_tuple,
                    dm_tuple,
                    dg_tuple,
                    selected_provider=filters["provider"],
                    selected_genre=filters["genre"],
                    min_year=filters["min_year"],
                    max_year=filters["max_year"],
                    min_vote=filters["min_vote"],
                    region=filters["region"],
                )

                # Construir resumen de filtros activos
                filters_info = [
                    f"Grupo: **{len(people_names)} integrantes** ({', '.join(people_names)})"
                ]
                if filters["provider"] != "Todas las plataformas":
                    filters_info.append(f"Plataforma: **{filters['provider']}** ({filters['region_label']})")
                if filters["genre"] != "Todos los géneros":
                    filters_info.append(f"Género: **{filters['genre']}**")
                if filters["min_year"] > 1950 or filters["max_year"] < 2026:
                    filters_info.append(f"Años: **{filters['min_year']} - {filters['max_year']}**")
                if filters["min_vote"] > 0.0:
                    filters_info.append(f"Nota mín.: **⭐ {filters['min_vote']}**")
                if disliked_movies:
                    filters_info.append(f"Vetadas: **{', '.join(disliked_movies)}**")
                if disliked_genres:
                    filters_info.append(f"Géneros vetados: **{', '.join(disliked_genres)}**")

                render_active_filters_banner(filters_info)
                render_recommendation_cards(extra["top_recommendations"])
                render_results_table(df_recs)
                render_pca_plot(extra)

            except Exception as e:
                st.error(f"Ocurrió un error al procesar las recomendaciones: {e}")



