import uvicorn
import os
import sys
from pathlib import Path

# Add the parent directory to the Python path so we can import from src
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Import the app after setting up sys.path
from app.app import app

if __name__ == "__main__":
    print(f"Starting FastAPI app with TMDB API")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
