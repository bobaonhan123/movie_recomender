import os
from dotenv import load_dotenv
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the project root directory (2 levels up from this file)
project_root = Path(__file__).parent.parent.parent


# Try to load from .env file
dotenv_path = os.path.join(project_root, '.env')
logger.info(f"Looking for .env file at: {dotenv_path}")
logger.info(f"File exists: {os.path.isfile(dotenv_path)}")

# Try to load environment variables from .env
load_dotenv(dotenv_path=dotenv_path)

# TMDB API settings - use environment variable if available, otherwise use hardcoded key
TMDB_API_KEY = os.getenv("TMDB_API_KEY", 'default')
logger.info(f"Using API key: {TMDB_API_KEY[:4]}****{TMDB_API_KEY[-4:]}")

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Application settings
APP_NAME = "Movie Recommendation App"