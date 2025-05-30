from fastapi import FastAPI, Request, HTTPException, Depends, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import logging
from typing import Optional, List, Dict, Any
import numpy as np
import joblib

from .settings import APP_NAME, TMDB_API_KEY
from .tmdb_client import TMDBClient
from .database import init_database, save_user_profile, get_user_profile, get_all_profiles
from ml_engine.knn import knn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if TMDB API key is available
if not TMDB_API_KEY:
    logger.error("TMDB_API_KEY is not set. Please add it to your .env file.")

# Create FastAPI instance
app = FastAPI(title=APP_NAME)

# Get the directory of this file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to get to the src directory
src_dir = os.path.dirname(current_dir)
# Get project root directory
project_root = os.path.dirname(src_dir)

# Set up templates
templates = Jinja2Templates(directory=os.path.join(src_dir, "templates"))

# KNN model is loaded by the knn instance from ml_engine.knn
# knn_model_path = os.path.join(project_root, "parameters", "knn_model.pkl")
# knn_model = None
# try:
#     knn_model = joblib.load(knn_model_path)
#     logger.info("KNN model loaded successfully")
# except Exception as e:
#     logger.error(f"Error loading KNN model: {str(e)}")

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    await init_database()

# Create TMDB client
def get_tmdb_client():
    return TMDBClient()

# User profile vectorization function
def vectorize_user_profile(profile_data: Dict[str, Any]) -> np.ndarray:
    """
    Convert user profile data to a 43-dimensional feature vector
    matching the KNN model requirements
    """
    # Initialize vector with zeros
    vector = np.zeros(43)
    
    # Gender encoding (first 3 features)
    gender = profile_data.get('gender', '')
    if gender == 'male':
        vector[0] = 1
    elif gender == 'female':
        vector[1] = 1
    elif gender == 'other':
        vector[2] = 1
    
    # Address encoding (next 10 features, positions 3-12)
    address = profile_data.get('address', '')
    address_mapping = {
        'hanoi': 3, 'hcmc': 4, 'danang': 5, 'cantho': 6, 'haiphong': 7,
        'nhatrang': 8, 'dalat': 9, 'hue': 10, 'vungtau': 11, 'other_city': 12
    }
    if address in address_mapping:
        vector[address_mapping[address]] = 1
    
    # Job encoding (next 15 features, positions 13-27)
    job = profile_data.get('job', '')
    job_mapping = {
        'student': 13, 'teacher': 14, 'engineer': 15, 'doctor': 16, 'nurse': 17,
        'manager': 18, 'sales': 19, 'marketing': 20, 'finance': 21, 'hr': 22,
        'designer': 23, 'writer': 24, 'artist': 25, 'freelancer': 26, 'other_job': 27
    }
    if job in job_mapping:
        vector[job_mapping[job]] = 1
    
    # Industry encoding (next 10 features, positions 28-37)
    industry = profile_data.get('industry', '')
    industry_mapping = {
        'technology': 28, 'healthcare': 29, 'education': 30, 'finance': 31, 'retail': 32,
        'manufacturing': 33, 'media': 34, 'government': 35, 'nonprofit': 36, 'other_industry': 37
    }
    if industry in industry_mapping:
        vector[industry_mapping[industry]] = 1
    
    # Age group encoding (last 5 features, positions 38-42)
    age_group = profile_data.get('age_group', '')
    age_mapping = {
        '18-25': 38, '26-35': 39, '36-45': 40, '46-55': 41, '56+': 42
    }
    if age_group in age_mapping:
        vector[age_mapping[age_group]] = 1
    
    return vector

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, tmdb_client: TMDBClient = Depends(get_tmdb_client)):
    """Home page that displays movie genres"""
    try:
        genres_data = tmdb_client.get_genres()
        genres = genres_data.get("genres", [])
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "genres": genres}
        )
    except Exception as e:
        logger.error(f"Error getting genres: {str(e)}")
        raise HTTPException(status_code=500, detail="Error getting movie genres")

@app.get("/genre/{genre_id}", response_class=HTMLResponse)
async def movies_by_genre(
    request: Request, 
    genre_id: int, 
    page: int = 1, 
    tmdb_client: TMDBClient = Depends(get_tmdb_client)
):
    """Page that displays movies for a specific genre"""
    try:
        # Get genre information
        genres_data = tmdb_client.get_genres()
        genres = {g["id"]: g["name"] for g in genres_data.get("genres", [])}
        
        # Get movies for this genre
        movies_data = tmdb_client.get_movies_by_genre(genre_id, page)
        movies = movies_data.get("results", [])

        # Add full image URLs to movies
        for movie in movies:
            if movie.get("poster_path"):
                movie["poster_url"] = tmdb_client.get_image_url(movie["poster_path"])
            else:
                movie["poster_url"] = None
                
        # Get genre name
        genre_name = genres.get(genre_id, "Unknown Genre")
        
        # Pagination details
        total_pages = movies_data.get("total_pages", 1)
        start_page = max(1, page - 2)
        end_page = min(total_pages + 1, page + 3)
        page_range = list(range(start_page, end_page))

        return templates.TemplateResponse(
            "genre.html",
            {
                "request": request, 
                "genre_id": genre_id,
                "genre_name": genre_name,
                "movies": movies,
                "current_page": page,
                "total_pages": total_pages,
                "page_range": page_range
            }
        )
    except Exception as e:
        logger.error(f"Error getting movies for genre {genre_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting movies for genre {genre_id}")


@app.get("/movie/{movie_id}", response_class=HTMLResponse)
async def movie_detail(
    request: Request, 
    movie_id: int, 
    tmdb_client: TMDBClient = Depends(get_tmdb_client)
):
    """Page that displays detailed information about a movie"""
    try:
        # Get movie details
        movie = tmdb_client.get_movie_details(movie_id)
        # print(f"Movie details: {movie}")  # Debugging line
        # Add full image URLs
        if movie.get("poster_path"):
            movie["poster_url"] = tmdb_client.get_image_url(movie["poster_path"])
        else:
            movie["poster_url"] = None
            
        if movie.get("backdrop_path"):
            movie["backdrop_url"] = tmdb_client.get_image_url(movie["backdrop_path"])
        else:
            movie["backdrop_url"] = None
        
        return templates.TemplateResponse(
            "movie.html",
            {"request": request, "movie": movie}
        )
    except Exception as e:
        logger.error(f"Error getting details for movie {movie_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting details for movie {movie_id}")

@app.get("/profile", response_class=HTMLResponse)
async def profile_form(request: Request):
    """Display user profile form"""
    return templates.TemplateResponse("profile.html", {"request": request})

@app.post("/profile")
async def save_profile(
    request: Request,
    user_id: str = Form(...),
    gender: str = Form(...),
    address: str = Form(...),
    job: str = Form(...),
    industry: str = Form(...),
    age_group: str = Form(...)
):
    """Save user profile and redirect to recommendations"""
    try:
        logger.info(f"=== PROFILE SUBMISSION ===")
        logger.info(f"Received profile data: user_id={user_id}, gender={gender}, address={address}, job={job}, industry={industry}, age_group={age_group}")
        
        # Validate required fields
        if not all([user_id, gender, address, job, industry, age_group]):
            logger.error("Missing required fields")
            raise HTTPException(status_code=422, detail="All fields are required")
        
        # Create profile data
        profile_data = {
            'user_id': user_id,
            'gender': gender,
            'address': address,
            'job': job,
            'industry': industry,
            'age_group': age_group
        }
        
        logger.info("Starting vectorization...")
        # Vectorize the profile
        profile_vector = vectorize_user_profile(profile_data)
        profile_data['vector'] = profile_vector
        logger.info(f"Vectorization successful: {profile_vector.shape}")
        
        logger.info("Saving to database...")
        # Save to database
        success = await save_user_profile(user_id, profile_data)
        
        if success:
            logger.info(f"Profile saved successfully for user {user_id}")
            # Detect if request is AJAX/fetch (for JS form)
            if request.headers.get("x-requested-with") == "XMLHttpRequest" or request.headers.get("accept") == "application/json":
                return {"status": "success", "redirect_url": f"/recommend/{user_id}"}
            else:
                # Redirect to recommendations page for normal form
                return RedirectResponse(url=f"/recommend/{user_id}", status_code=303)
        else:
            logger.error("Failed to save profile to database")
            raise HTTPException(status_code=500, detail="Failed to save profile")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error saving profile: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error saving profile: {str(e)}")

@app.get("/recommend/{user_id}", response_class=HTMLResponse)
async def get_recommendations(
    request: Request,
    user_id: str,
    tmdb_client: TMDBClient = Depends(get_tmdb_client)
):
    """Get movie recommendations for a user"""
    try:
        # Get user profile from database
        profile = await get_user_profile(user_id)
        if not profile:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        # Get user vector
        user_vector = np.array(profile['vector'])
        
        # Define genre mapping based on the order provided
        GERNES_MAPPING = {
            0: 'Action', 1: 'Adventure', 2: 'Animation', 3: 'Comedy', 4: 'Crime',
            5: 'Drama', 6: 'Family', 7: 'Fantasy', 8: 'Horror', 9: 'Musical',
            10: 'Mystery', 11: 'Romance', 12: 'Sci-Fi', 13: 'Thriller'
        }
        
        # Get nearest neighbors (each neighbor label is a genre distribution vector)
        raw_neighbors = knn.query(user_vector, k=5)

        # Weighted average of neighbor genre vectors by similarity
        labels = np.array([y for y, sim in raw_neighbors])
        sims = np.array([sim for y, sim in raw_neighbors])
        if np.sum(sims) > 0:
            weighted_avg = np.average(labels, axis=0, weights=sims)
        else:
            weighted_avg = np.mean(labels, axis=0)
        # Normalize to probability distribution
        weighted_avg = weighted_avg / (np.sum(weighted_avg) + 1e-8)

        # Build sorted list of genres with probabilities
        recommended_genres_with_probs = []
        for idx, score in sorted(enumerate(weighted_avg), key=lambda x: x[1], reverse=True):
            genre_name = GERNES_MAPPING.get(idx)
            if genre_name:
                recommended_genres_with_probs.append({"name": genre_name, "probability": float(score)})

        # Sort genres by probability in descending order
        recommended_genres_with_probs.sort(key=lambda x: x["probability"], reverse=True)
        
        # Fetch movies for the top N genres (e.g., top 3)
        top_n_genres = 3
        movies_for_recommendation = []
        
        # Get TMDB genre list to map names to IDs
        tmdb_genres_data = tmdb_client.get_genres()
        tmdb_genre_name_to_id = {g['name']: g['id'] for g in tmdb_genres_data.get("genres", [])}

        for i, genre_info in enumerate(recommended_genres_with_probs[:top_n_genres]):
            genre_name = genre_info["name"]
            genre_id = tmdb_genre_name_to_id.get(genre_name)
            
            if genre_id:
                try:
                    # Fetch a few movies for this genre
                    movies_data = tmdb_client.get_movies_by_genre(genre_id, page=1)
                    genre_movies = movies_data.get("results", [])
                    
                    for movie in genre_movies[:3]: # Take top 3 movies for this genre
                        if movie.get("poster_path"):

                            movie["poster_url"] = tmdb_client.get_image_url(movie["poster_path"])

                        else:

                            movie["poster_url"] = None
                        movie["recommended_genre"] = genre_name # Add recommended genre to movie
                        movie["genre_probability"] = genre_info["probability"] # Add probability
                        movies_for_recommendation.append(movie)
                except Exception as e:
                    logger.error(f"Error fetching movies for genre {genre_name} (ID: {genre_id}): {str(e)}")
        
        # Remove duplicate movies if any, keeping the one with higher probability or from a higher-ranked genre
        # This is a simple way to handle duplicates; more sophisticated logic might be needed
        seen_movie_ids = set()
        unique_movies = []
        for movie in movies_for_recommendation:
            if movie['id'] not in seen_movie_ids:
                unique_movies.append(movie)
                seen_movie_ids.add(movie['id'])
        
        logger.info(f"Recommendations for user {user_id}: {recommended_genres_with_probs}")
        logger.info(f"Movies for recommendation: {len(unique_movies)} movies")

        return templates.TemplateResponse(
            "recommend.html",
            {
                "request": request,
                "user_id": user_id,
                "profile": profile,
                "recommended_genres": recommended_genres_with_probs, # Pass sorted genres with probs
                "movies": unique_movies # Pass fetched movies
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendations for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

@app.get("/test-form", response_class=HTMLResponse)
async def test_form(request: Request):
    """Test form page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Profile Form</title>
    </head>
    <body>
        <h1>Test Profile Form</h1>
        <form method="POST" action="/profile">
            <p>
                <label>User ID:</label>
                <input type="text" name="user_id" value="test123" required>
            </p>
            <p>
                <label>Gender:</label>
                <select name="gender" required>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                </select>
            </p>
            <p>
                <label>Address:</label>
                <select name="address" required>
                    <option value="hanoi">Hanoi</option>
                    <option value="hcmc">HCMC</option>
                </select>
            </p>
            <p>
                <label>Job:</label>
                <select name="job" required>
                    <option value="engineer">Engineer</option>
                    <option value="student">Student</option>
                </select>
            </p>
            <p>
                <label>Industry:</label>
                <select name="industry" required>
                    <option value="technology">Technology</option>
                    <option value="education">Education</option>
                </select>
            </p>
            <p>
                <label>Age Group:</label>
                <select name="age_group" required>
                    <option value="26-35">26-35</option>
                    <option value="18-25">18-25</option>
                </select>
            </p>
            <button type="submit">Submit</button>
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)