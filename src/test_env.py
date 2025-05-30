import os
from dotenv import load_dotenv
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent

# Load environment variables from .env file with an absolute path
dotenv_path = os.path.join(project_root, '.env')
print(f"Looking for .env file at: {dotenv_path}")
print(f"File exists: {os.path.isfile(dotenv_path)}")

load_dotenv(dotenv_path=dotenv_path)

# Check if TMDB API key is loaded
api_key = os.getenv("TMDB_API_KEY")
if api_key:
    print(f"API Key found: {api_key[:4]}{'*' * (len(api_key) - 8)}{api_key[-4:]}")
else:
    print("API Key not found!")

# Print all environment variables (be careful with sensitive data)
print("\nAll environment variables:")
for key, value in os.environ.items():
    if key == "TMDB_API_KEY":
        print(f"{key}: {value[:4]}{'*' * (len(value) - 8)}{value[-4:]}")
    else:
        print(f"{key}: {value}")
