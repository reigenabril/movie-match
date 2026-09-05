"""
Gestión de estilos CSS y tema visual para Streamlit.
"""
import os
import streamlit as st

UI_DIR = os.path.dirname(os.path.abspath(__file__))
CSS_FILE = os.path.join(UI_DIR, "styles.css")


def load_css():
    """Carga e inyecta el archivo CSS externo en la app de Streamlit."""
    if os.path.exists(CSS_FILE):
        with open(CSS_FILE, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
