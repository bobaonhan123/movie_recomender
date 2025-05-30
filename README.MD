# Movie Recommendation App

A FastAPI web application that lists movies from The Movie Database (TMDB) API by genre. The application fetches data from the TMDB API and displays movie information including posters, ratings, and other details.

## Features

- Browse movies by genre
- View movie details including images, descriptions, and production information
- Pagination for movie listings
- Responsive design

## Technologies Used

- FastAPI - Web framework
- Jinja2 - Templating engine
- TMDB API - Movie data source
- Python-dotenv - Environment variables management
- Bootstrap 5 - Frontend styling

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/movie_recomender_app.git
cd movie_recomender_app
```

2. Create and activate a virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

3. Install dependencies:
```powershell
pip install fastapi uvicorn python-dotenv requests jinja2
```

4. Create a `.env` file in the root directory and add your TMDB API key:
```
TMDB_API_KEY=your_tmdb_api_key_here
```

You can obtain a TMDB API key by registering at [https://www.themoviedb.org/settings/api](https://www.themoviedb.org/settings/api)

## Running the Application

Run the application using the following command from the project root:

```powershell
python .\src\main.py
```

Access the application in your browser at `http://localhost:8000`

## Project Structure

```
movie_recomender_app/
│
├── .env                # Environment variables (not in version control)
├── .gitignore          # Git ignore file
├── README.md           # This file
│
└── src/                # Source code
    ├── main.py         # Entry point for the application
    │
    ├── app/            # Application code
    │   ├── __init__.py
    │   ├── app.py      # FastAPI application
    │   ├── settings.py # Application settings
    │   └── tmdb_client.py # TMDB API client
    │
    └── templates/      # Jinja2 templates
        ├── base.html   # Base template
        ├── index.html  # Home page with genres
        ├── genre.html  # Movies by genre page
        └── movie.html  # Movie details page
```

## License

This project is licensed under the MIT License.#   m o v i e _ r e c o m e n d e r  
 