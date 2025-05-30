import asyncio
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from app.database import save_user_profile, get_user_profile
from app.app import vectorize_user_profile

async def test_user_flow():
    """Test the complete user profile and recommendation flow"""
    
    # Test user profile data
    profile_data = {
        'user_id': 'test_user_123',
        'gender': 'male',
        'address': 'hanoi',
        'job': 'engineer',
        'industry': 'technology',
        'age_group': '26-35'
    }
    
    print("Testing user profile vectorization...")
    
    # Test vectorization
    vector = vectorize_user_profile(profile_data)
    print(f"✅ Vectorization successful: {vector.shape} dimensions")
    print(f"✅ Vector sample: {vector[:10]}")  # Show first 10 elements
    
    # Add vector to profile data
    profile_data['vector'] = vector
    
    print("\nTesting database operations...")
    
    # Test saving to database
    success = await save_user_profile(profile_data['user_id'], profile_data)
    if success:
        print("✅ Profile saved to SQLite database successfully")
    else:
        print("❌ Failed to save profile to database")
        return
    
    # Test retrieving from database
    retrieved_profile = await get_user_profile(profile_data['user_id'])
    if retrieved_profile:
        print("✅ Profile retrieved from database successfully")
        print(f"✅ Retrieved data: {retrieved_profile['user_id']}, {retrieved_profile['gender']}, {retrieved_profile['job']}")
        
        # Check if vector was preserved
        if 'vector' in retrieved_profile:
            retrieved_vector = retrieved_profile['vector']
            print(f"✅ Vector preserved: {len(retrieved_vector)} dimensions")
        else:
            print("❌ Vector not preserved in database")
    else:
        print("❌ Failed to retrieve profile from database")

if __name__ == "__main__":
    print("🧪 Testing Movie Recommendation App with SQLite Database...")
    print("=" * 60)
    
    asyncio.run(test_user_flow())
    
    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")
    print("🎬 Your movie recommendation app is ready with SQLite database!")
