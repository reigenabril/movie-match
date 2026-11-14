# Movie Match

Una aplicación estática en HTML, CSS y JavaScript para encontrar películas para un grupo usando TMDB. No necesita Streamlit, Python ni un servidor propio.

## Uso

1. Abrí el sitio e ingresá una **API key v3 de TMDB** (no el token Bearer v4). Podés obtenerla en [TMDB](https://www.themoviedb.org/settings/api). La clave se guarda en el almacenamiento local del navegador y se envía directamente a TMDB. No ingreses una clave que no quieras usar en el navegador.
2. Agregá al menos una película por integrante, separada por comas. Ajustá los pesos y filtros si querés.
3. Buscá el match. Los títulos se resuelven usando el primer resultado de búsqueda de TMDB.

Las recomendaciones se basan en las listas de recomendaciones de TMDB para cada favorita, combinadas por posición y peso. Los títulos vetados se excluyen; los géneros vetados también. El filtro de plataforma cruza los resultados con las primeras cinco páginas populares disponibles por suscripción en la región, por lo que puede dejar fuera películas menos populares. Esta implementación no reproduce los embeddings semánticos ni el gráfico PCA de la versión Python.

## Publicar en GitHub Pages

Subí `index.html`, `style.css` y `app.js` a la raíz del repositorio. En **Settings → Pages**, elegí **Deploy from a branch**, la rama `main` y la carpeta `/ (root)`. Después de que termine el despliegue, la URL será `https://reigenabril.github.io/movie-match/`. Para probar localmente, ejecutá `python3 -m http.server` en esta carpeta y abrí `http://localhost:8000`.

Este producto usa la API de TMDB, pero no está avalado ni certificado por TMDB.
