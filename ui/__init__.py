"""
Paquete UI para la aplicación Movie Match en Streamlit.
"""
from ui.theme import load_css
from ui.sidebar import render_sidebar
from ui.group_inputs import render_group_inputs, render_veto_inputs
from ui.results_view import (
    render_active_filters_banner,
    render_recommendation_cards,
    render_results_table,
    render_pca_plot,
)

__all__ = [
    "load_css",
    "render_sidebar",
    "render_group_inputs",
    "render_veto_inputs",
    "render_active_filters_banner",
    "render_recommendation_cards",
    "render_results_table",
    "render_pca_plot",
]
