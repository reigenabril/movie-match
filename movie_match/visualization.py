"""
Visualización 2D mediante PCA del espacio vectorial de gustos, vetos y recomendaciones.
"""
from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from movie_match.embeddings import embed_text


def plot_taste_map(extra_data: dict, save_path: Optional[str] = None, ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Genera el gráfico 2D PCA con el mapa de gustos N-personas, películas vetadas y recomendaciones."""
    combined_vector = extra_data["combined_vector"]
    all_found_movies = extra_data["all_found_movies"]
    top_recommendations = extra_data["top_recommendations"]
    people_names = extra_data.get("people_names", [f"Persona {i+1}" for i in range(len(all_found_movies))])
    disliked_found = extra_data.get("disliked_found", [])

    all_vecs = []

    for idx, found in enumerate(all_found_movies):
        for m in found:
            all_vecs.append(embed_text(m["overview"]))

    # Películas vetadas / disgustos
    for m in disliked_found:
        all_vecs.append(embed_text(m["overview"]))

    all_vecs.append(combined_vector)

    for m in top_recommendations:
        all_vecs.append(embed_text(m["overview"]))

    all_vecs = np.array(all_vecs)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_vecs)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = ax.get_figure()

    curr_idx = 0
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf"]

    for p_idx, found in enumerate(all_found_movies):
        color = colors[p_idx % len(colors)]
        p_name = people_names[p_idx] if p_idx < len(people_names) else f"Persona {p_idx+1}"
        n_m = len(found)
        p_coords = coords[curr_idx : curr_idx + n_m]
        ax.scatter(p_coords[:, 0], p_coords[:, 1], color=color, s=110, label=p_name, zorder=3)
        for i, m in enumerate(found):
            ax.annotate(m["title"], p_coords[i], fontsize=8, alpha=0.9, color=color, weight="bold")
        curr_idx += n_m

    # Dislikes / Vetadas
    if disliked_found:
        n_dis = len(disliked_found)
        dis_coords = coords[curr_idx : curr_idx + n_dis]
        ax.scatter(dis_coords[:, 0], dis_coords[:, 1], color="#d62728", marker="v", s=130, label="Películas Vetadas", zorder=4)
        for i, m in enumerate(disliked_found):
            ax.annotate(f"🚫 {m['title']}", dis_coords[i], fontsize=8, alpha=0.9, color="#b71c1c", weight="bold")
        curr_idx += n_dis

    # Gusto combinado
    comb_coord = coords[curr_idx : curr_idx + 1]
    ax.scatter(comb_coord[:, 0], comb_coord[:, 1], color="black", marker="X", s=240, label="Gusto Combinado", zorder=5)
    curr_idx += 1

    # Recomendaciones
    n_recs = len(top_recommendations)
    rec_coords = coords[curr_idx : curr_idx + n_recs]
    ax.scatter(rec_coords[:, 0], rec_coords[:, 1], color="#2ca02c", s=70, alpha=0.7, label="Recomendaciones", zorder=2)

    # Resaltar la #1
    if top_recommendations:
        ax.annotate(top_recommendations[0]["title"], rec_coords[0], fontsize=9, weight="bold", color="#1b5e20")

    ax.legend(loc="best")
    title_suffix = ""
    if extra_data.get("selected_provider") and extra_data.get("selected_provider") != "Todas las plataformas":
        title_suffix += f" (Plataforma: {extra_data.get('selected_provider')})"
    if extra_data.get("selected_genre") and extra_data.get("selected_genre") != "Todos los géneros":
        title_suffix += f" (Género: {extra_data.get('selected_genre')})"

    ax.set_title(f"Movie Match - Mapa de Gustos Grupo & Veto (PCA 2D){title_suffix}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle="--", alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Gráfico guardado en: {save_path}")

    return fig
