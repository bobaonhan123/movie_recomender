import requests
from .settings import TMDB_API_KEY, TMDB_BASE_URL, TMDB_IMAGE_BASE_URL

class TMDBClient:
    """
    Client for The Movie Database (TMDB) API
    """
    def __init__(self):
        self.api_key = TMDB_API_KEY
        self.base_url = TMDB_BASE_URL
        self.image_base_url = TMDB_IMAGE_BASE_URL
        
    def _make_request(self, endpoint, params=None):
        """Make a request to the TMDB API"""
        import logging
        logger = logging.getLogger(__name__)
        
        if params is None:
            params = {}
        
        # Add API key to parameters
        params['api_key'] = self.api_key
        
        # Build the full URL
        url = f"{self.base_url}/{endpoint}"
        
        logger.info(f"Making request to {url} with params: {params}")
        
        try:
            # Make the request
            response = requests.get(url, params=params)
            
            # Raise exception for bad responses
            response.raise_for_status()
            
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {e}")
            logger.error(f"URL: {url}, API Key: {self.api_key[:4]}{'*' * (len(self.api_key) - 8)}{self.api_key[-4:]}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status Code: {e.response.status_code}")
                logger.error(f"Response Text: {e.response.text}")
            raise
    
    def get_genres(self):
        """Get list of movie genres"""
        return self._make_request('genre/movie/list')
    
    def get_movies_by_genre(self, genre_id, page=1):
        """Get movies by genre ID"""
        return self._make_request('discover/movie', {
            'with_genres': genre_id,
            'page': page,
            'sort_by': 'popularity.desc'
        })
    
    def get_movie_details(self, movie_id):
        """Get detailed information about a movie"""
        return self._make_request(f'movie/{movie_id}')
    
    def get_image_url(self, image_path):
        """Get the full URL for an image"""
        if not image_path:
            return None
        return f"{self.image_base_url}{image_path}"
