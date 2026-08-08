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
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS with modern UI aesthetics
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #ff4b4b 0%, #ff6b6b 100%);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 3.2em;
        font-size: 1.05rem;
        border: none;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(255, 75, 75, 0.4);
    }
    .person-card {
        background: #1e222d;
        border-radius: 12px;
        padding: 18px;
        border: 1px solid #2d3342;
        margin-bottom: 15px;
    }
    .score-badge {
        background-color: #2e7d32;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.95em;
        display: inline-block;
    }
    .veto-card {
        background: #2b1b1f;
        border-radius: 12px;
        padding: 18px;
        border: 1px solid #5c2429;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎬 Movie Match — AI Recommender (Modo Grupo & Veto)")
st.write(
    "Encontrá la película perfecta combinando los gustos de **2 o más personas** y descartando lo que no quieran ver."
)

st.sidebar.header("⚙️ Configuración & Filtros")
tmdb_token = os.getenv("TMDB_BEARER_TOKEN")

if not tmdb_token:
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

@st.cache_data(show_spinner=False, ttl=3600)
def cached_get_genres():
    return get_movie_genres()

genres_dict = cached_get_genres()

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
    people_movies = [list(pm) for pm in people_movies_tuple]
    people_names = list(people_names_tuple)
    people_weights = list(people_weights_tuple)
    disliked_movies = list(disliked_movies_tuple)
    disliked_genres = list(disliked_genres_tuple)

    return recommend(
        people_movies,
        people_names=people_names,
        people_weights=people_weights,
        disliked_movies=disliked_movies,
        disliked_genres=disliked_genres,
        selected_provider=selected_provider,
        selected_genre=selected_genre,
        min_year=min_year,
        max_year=max_year,
        min_vote_average=min_vote,
        region=region,
    )

# --- SECCIÓN MODO GRUPO ---
st.subheader("👥 Integrantes del Grupo")
n_people = st.number_input("Número de participantes", min_value=2, max_value=8, value=2, step=1)

default_data = [
    ("Persona 1", "Interstellar, Eternal Sunshine of the Spotless Mind, Whiplash"),
    ("Persona 2", "Coco, La La Land, Amelie"),
    ("Persona 3", "Inception, The Dark Knight, Fight Club"),
    ("Persona 4", "Pulp Fiction, Goodfellas, Scarface"),
    ("Persona 5", "Spirited Away, Princess Mononoke, Your Name"),
    ("Persona 6", "Toy Story, Finding Nemo, Shrek"),
]

people_names = []
people_movies_list = []
people_weights = []

# Distribuir inputs en columnas
grid_cols = st.columns(min(3, int(n_people)))

for i in range(int(n_people)):
    col = grid_cols[i % len(grid_cols)]
    def_name, def_movies = default_data[i] if i < len(default_data) else (f"Persona {i+1}", "")
    
    with col:
        st.markdown(f"#### 👤 Participante {i+1}")
        name = st.text_input(f"Nombre / Apodo", value=def_name, key=f"name_{i}")
        movies_str = st.text_area(
            f"Películas favoritas (separadas por coma)",
            value=def_movies,
            height=100,
            key=f"movies_{i}",
        )
        weight = st.slider(f"Importancia / Peso ({name})", min_value=0.5, max_value=3.0, value=1.0, step=0.5, key=f"w_{i}")
        
        m_list = [t.strip() for t in movies_str.split(",") if t.strip()]
        people_names.append(name)
        people_movies_list.append(m_list)
        people_weights.append(weight)

st.write("---")

# --- SECCIÓN PREFERENCIAS NEGATIVAS / VETO ---
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
        disliked_movies_list = [t.strip() for t in disliked_input.split(",") if t.strip()]

    with v_col2:
        available_genres_list = [g for g in genres_dict.keys() if g != "Todos los géneros"]
        disliked_genres_selected = st.multiselect(
            "Géneros a excluir / vetar",
            options=available_genres_list,
            default=[],
            help="Cualquier película de estos géneros será descartada completamente del resultado."
        )

st.markdown("---")

if st.button("🚀 Buscar Recomendaciones de Grupo"):
    # Validaciones
    empty_people = [people_names[idx] for idx, m in enumerate(people_movies_list) if not m]
    if empty_people:
        st.warning(f"Ingresá al menos una película favorita para: {', '.join(empty_people)}")
    else:
        with st.spinner("🧠 Calculando vector de gusto combinado y filtrando catálogo..."):
            try:
                # Formatear estructuras para caché (tuplas inmutables)
                pm_tuple = tuple(tuple(m) for m in people_movies_list)
                pn_tuple = tuple(people_names)
                pw_tuple = tuple(people_weights)
                dm_tuple = tuple(disliked_movies_list)
                dg_tuple = tuple(disliked_genres_selected)

                df_recs, extra = cached_recommend_group(
                    pm_tuple,
                    pn_tuple,
                    pw_tuple,
                    dm_tuple,
                    dg_tuple,
                    selected_provider=selected_provider,
                    selected_genre=selected_genre,
                    min_year=min_year,
                    max_year=max_year,
                    min_vote=min_vote,
                    region=selected_region,
                )
                top_recs = extra["top_recommendations"]

                filters_info = []
                filters_info.append(f"Grupo: **{len(people_names)} integrantes** ({', '.join(people_names)})")
                if selected_provider != "Todas las plataformas":
                    filters_info.append(f"Plataforma: **{selected_provider}** ({selected_region_label})")
                if selected_genre != "Todos los géneros":
                    filters_info.append(f"Género: **{selected_genre}**")
                if min_year > 1950 or max_year < 2026:
                    filters_info.append(f"Años: **{min_year} - {max_year}**")
                if min_vote > 0.0:
                    filters_info.append(f"Nota mín.: **⭐ {min_vote}**")
                if disliked_movies_list:
                    filters_info.append(f"Vetadas: **{', '.join(disliked_movies_list)}**")
                if disliked_genres_selected:
                    filters_info.append(f"Géneros vetados: **{', '.join(disliked_genres_selected)}**")

                st.info("🍿 **Filtros y Configuración Activa:**\n- " + "\n- ".join(filters_info))

                st.subheader("🎉 Top Recomendaciones para el Grupo")

                # Mostrar las mejores películas recomendadas
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
                            st.markdown(f"<span class='score-badge'>Match Grupo: {match_pct}%</span>", unsafe_allow_html=True)
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
                st.subheader("📋 Tabla Completa de Resultados")
                st.dataframe(
                    df_recs.rename(
                        columns={
                            "title": "Título",
                            "score": "Match Grupo",
                            "vote_average": "Calificación TMDB",
                            "genres": "Géneros",
                            "providers": "Plataforma(s)",
                            "overview": "Sinopsis",
                        }
                    ),
                    use_container_width=True,
                )

                st.write("---")
                st.subheader("📊 Mapa del Espacio Vectorial de Gustos (PCA 2D)")

                fig = plot_taste_map(extra)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Ocurrió un error al procesar las recomendaciones: {e}")



