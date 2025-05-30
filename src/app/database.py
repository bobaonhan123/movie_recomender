import aiosqlite
import json
import os
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Database file path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
DB_PATH = os.path.join(project_root, "data", "users.db")

# Ensure data directory exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

async def init_database():
    """Initialize the SQLite database and create tables if they don't exist"""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                gender TEXT,
                address TEXT,
                job TEXT,
                industry TEXT,
                age_group TEXT,
                vector_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                movie_id INTEGER,
                recommendation_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
            )
        """)
        
        await db.commit()
        logger.info("Database initialized successfully")

async def save_user_profile(user_id: str, profile_data: Dict[str, Any]) -> bool:
    """Save user profile to SQLite database"""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            # Check if user exists
            cursor = await db.execute(
                "SELECT user_id FROM user_profiles WHERE user_id = ?", 
                (user_id,)
            )
            existing_user = await cursor.fetchone()
            
            # Serialize vector data if it exists
            vector_data = None
            if 'vector' in profile_data:
                vector_data = json.dumps(profile_data['vector'].tolist() if hasattr(profile_data['vector'], 'tolist') else profile_data['vector'])
            
            if existing_user:
                # Update existing user
                await db.execute("""
                    UPDATE user_profiles 
                    SET gender = ?, address = ?, job = ?, industry = ?, age_group = ?, 
                        vector_data = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                """, (
                    profile_data.get('gender'),
                    profile_data.get('address'),
                    profile_data.get('job'),
                    profile_data.get('industry'),
                    profile_data.get('age_group'),
                    vector_data,
                    user_id
                ))
                logger.info(f"Updated profile for user {user_id}")
            else:
                # Insert new user
                await db.execute("""
                    INSERT INTO user_profiles 
                    (user_id, gender, address, job, industry, age_group, vector_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    profile_data.get('gender'),
                    profile_data.get('address'),
                    profile_data.get('job'),
                    profile_data.get('industry'),
                    profile_data.get('age_group'),
                    vector_data
                ))
                logger.info(f"Created new profile for user {user_id}")
            
            await db.commit()
            return True
            
    except Exception as e:
        logger.error(f"Error saving user profile: {str(e)}")
        return False

async def get_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """Get user profile from SQLite database"""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            cursor = await db.execute("""
                SELECT user_id, gender, address, job, industry, age_group, vector_data, created_at, updated_at
                FROM user_profiles WHERE user_id = ?
            """, (user_id,))
            
            row = await cursor.fetchone()
            if row:
                profile = {
                    'user_id': row[0],
                    'gender': row[1],
                    'address': row[2],
                    'job': row[3],
                    'industry': row[4],
                    'age_group': row[5],
                    'created_at': row[7],
                    'updated_at': row[8]
                }
                
                # Deserialize vector data if it exists
                if row[6]:
                    profile['vector'] = json.loads(row[6])
                
                return profile
            return None
            
    except Exception as e:
        logger.error(f"Error getting user profile: {str(e)}")
        return None

async def get_all_profiles() -> List[Dict[str, Any]]:
    """Get all user profiles from SQLite database"""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            cursor = await db.execute("""
                SELECT user_id, gender, address, job, industry, age_group, vector_data, created_at, updated_at
                FROM user_profiles
                ORDER BY created_at DESC
            """)
            
            profiles = []
            async for row in cursor:
                profile = {
                    'user_id': row[0],
                    'gender': row[1],
                    'address': row[2],
                    'job': row[3],
                    'industry': row[4],
                    'age_group': row[5],
                    'created_at': row[7],
                    'updated_at': row[8]
                }
                
                # Deserialize vector data if it exists
                if row[6]:
                    profile['vector'] = json.loads(row[6])
                
                profiles.append(profile)
            
            return profiles
            
    except Exception as e:
        logger.error(f"Error getting all profiles: {str(e)}")
        return []

async def save_user_recommendations(user_id: str, recommendations: List[Dict[str, Any]]) -> bool:
    """Save user recommendations to database"""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            # Clear existing recommendations for this user
            await db.execute(
                "DELETE FROM user_recommendations WHERE user_id = ?", 
                (user_id,)
            )
            
            # Insert new recommendations
            for rec in recommendations:
                await db.execute("""
                    INSERT INTO user_recommendations (user_id, movie_id, recommendation_score)
                    VALUES (?, ?, ?)
                """, (
                    user_id,
                    rec.get('movie_id'),
                    rec.get('score', 0.0)
                ))
            
            await db.commit()
            logger.info(f"Saved {len(recommendations)} recommendations for user {user_id}")
            return True
            
    except Exception as e:
        logger.error(f"Error saving recommendations: {str(e)}")
        return False

async def get_user_recommendations(user_id: str) -> List[Dict[str, Any]]:
    """Get user recommendations from database"""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            cursor = await db.execute("""
                SELECT movie_id, recommendation_score, created_at
                FROM user_recommendations 
                WHERE user_id = ?
                ORDER BY recommendation_score DESC
            """, (user_id,))
            
            recommendations = []
            async for row in cursor:
                recommendations.append({
                    'movie_id': row[0],
                    'score': row[1],
                    'created_at': row[2]
                })
            
            return recommendations
            
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        return []
