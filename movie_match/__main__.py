"""
Punto de entrada CLI ejecutable: python -m movie_match
"""
import os
import argparse
from movie_match.config import OUTPUT_DIR
from movie_match.recommender import recommend
from movie_match.visualization import plot_taste_map


def main():
    parser = argparse.ArgumentParser(description="Movie Match - Recomendador por embeddings (Grupo & Veto)")
    parser.add_argument("--save-plot", type=str, help="Ruta para guardar el gráfico PCA generado")
    parser.add_argument("--demo", action="store_true", help="Ejecutar con datos de prueba predeterminados (3 personas)")
    parser.add_argument("--provider", type=str, default="Todas las plataformas", help="Filtrar por plataforma")
    parser.add_argument("--genre", type=str, default="Todos los géneros", help="Filtrar por género")
    parser.add_argument("--min-year", type=int, default=None, help="Año mínimo de estreno")
    parser.add_argument("--max-year", type=int, default=None, help="Año máximo de estreno")
    parser.add_argument("--min-vote", type=float, default=0.0, help="Puntuación mínima de TMDB (0 a 10)")
    parser.add_argument("--region", type=str, default="AR", help="Código de país para disponibilidad")
    args = parser.parse_args()

    print("Movie Match - Recomendador por Embeddings Semánticos (Modo Grupo & Preferencias Negativas)\n")

    if args.demo:
        people_movies = [
            ["Interstellar", "Whiplash"],
            ["Coco", "Amelie"],
            ["Inception", "The Dark Knight"],
        ]
        people_names = ["Ana", "Carlos", "Sofia"]
        disliked_movies = ["Twilight"]
        disliked_genres = ["Terror"]
    else:
        p1 = ["Interstellar", "Eternal Sunshine of the Spotless Mind", "Whiplash"]
        p2 = ["Coco", "La La Land", "Amelie"]
        people_movies = [p1, p2]
        people_names = ["Persona 1", "Persona 2"]
        disliked_movies = []
        disliked_genres = []

    print(f"\nProcesando gustos y consultando TMDB (Participantes: {people_names}, Veto: {disliked_movies + disliked_genres})...")
    df_recs, extra = recommend(
        people_movies,
        people_names=people_names,
        disliked_movies=disliked_movies,
        disliked_genres=disliked_genres,
        selected_provider=args.provider,
        selected_genre=args.genre,
        min_year=args.min_year,
        max_year=args.max_year,
        min_vote_average=args.min_vote,
        region=args.region,
    )

    print("\nTop Recomendaciones:")
    print(df_recs.to_string(index=False))

    save_plot_path = args.save_plot or os.path.join(OUTPUT_DIR, "mapa_gustos.png")
    plot_taste_map(extra, save_path=save_plot_path)


if __name__ == "__main__":
    main()
