"""
Componentes de formulario para captura de gustos de participantes y preferencias negativas.
"""
from __future__ import annotations
from typing import List, Tuple
import streamlit as st

DEFAULT_PARTICIPANTS_DATA = [
    ("Persona 1", "Interstellar, Eternal Sunshine of the Spotless Mind, Whiplash"),
    ("Persona 2", "Coco, La La Land, Amelie"),
    ("Persona 3", "Inception, The Dark Knight, Fight Club"),
    ("Persona 4", "Pulp Fiction, Goodfellas, Scarface"),
    ("Persona 5", "Spirited Away, Princess Mononoke, Your Name"),
    ("Persona 6", "Toy Story, Finding Nemo, Shrek"),
]


def render_group_inputs() -> tuple[list[str], list[list[str]], list[float]]:
    """Renderiza los inputs para N personas con sus películas favoritas y pesos."""
    st.subheader("👥 Integrantes del Grupo")
    n_people = st.number_input("Número de participantes", min_value=2, max_value=8, value=2, step=1)

    people_names = []
    people_movies_list = []
    people_weights = []

    grid_cols = st.columns(min(3, int(n_people)))

    for i in range(int(n_people)):
        col = grid_cols[i % len(grid_cols)]
        def_name, def_movies = (
            DEFAULT_PARTICIPANTS_DATA[i]
            if i < len(DEFAULT_PARTICIPANTS_DATA)
            else (f"Persona {i+1}", "")
        )

        with col:
            st.markdown(f"#### 👤 Participante {i+1}")
            name = st.text_input("Nombre / Apodo", value=def_name, key=f"name_{i}")
            movies_str = st.text_area(
                "Películas favoritas (separadas por coma)",
                value=def_movies,
                height=100,
                key=f"movies_{i}",
            )
            weight = st.slider(
                f"Importancia / Peso ({name})",
                min_value=0.5,
                max_value=3.0,
                value=1.0,
                step=0.5,
                key=f"w_{i}",
            )

            m_list = [t.strip() for t in movies_str.split(",") if t.strip()]
            people_names.append(name)
            people_movies_list.append(m_list)
            people_weights.append(weight)

    st.write("---")
    return people_names, people_movies_list, people_weights


def render_veto_inputs(available_genres: list[str]) -> tuple[list[str], list[str]]:
    """Renderiza el bloque expandible de veto / preferencias negativas."""
    with st.expander("🚫 Preferencias Negativas & Veto (Opcional)", expanded=False):
        st.markdown("Agregá películas o géneros que **NO** les gusten o que quieran **evitar** para esta noche.")
        v_col1, v_col2 = st.columns(2)

        with v_col1:
            disliked_input = st.text_area(
                "Películas desaprobadas / vetadas (separadas por coma)",
                value="",
                placeholder="Ej: Twilight, Transformers, Saw",
                height=90,
                key="disliked_movies_input",
            )
            disliked_movies = [t.strip() for t in disliked_input.split(",") if t.strip()]

        with v_col2:
            disliked_genres = st.multiselect(
                "Géneros a excluir / vetar",
                options=available_genres,
                default=[],
                help="Cualquier película de estos géneros será descartada completamente del resultado.",
            )

    st.markdown("---")
    return disliked_movies, disliked_genres
