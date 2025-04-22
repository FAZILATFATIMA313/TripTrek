from database import get_user, save_trip, get_past_trips

# Test 1: Create a fake user
print("🐶 Creating test user...")
user = get_user("test_device_123")
print(f"User ID: {user['_id']}")

# Test 2: Save a fake trip
print("\n✈️ Saving test trip...")
save_trip("test_device_123", {"destination": "Paris", "date": "2023-12-25"})
print("Trip saved!")

# Test 3: Get past trips
print("\n📜 Fetching past trips...")
trips = get_past_trips("test_device_123")
print("Past trips:", trips)