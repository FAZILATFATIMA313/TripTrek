from pymongo import MongoClient
from datetime import datetime

# 🌟 Connect to your FREE MongoDB database
db = MongoClient("mongodb+srv://triptrekuser:TrikTrek3!@cluster0.seub3ya.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0").triptrek

# ✨ Automatic account creator
def get_user(device_id):
    user = db.users.find_one({"device_id": device_id})
    if not user:
        user = {"device_id": device_id, "past_trips": [], "created_at": datetime.now()}
        db.users.insert_one(user)
    return user

# 💾 Save trips
def save_trip(device_id, trip_data):
    db.users.update_one(
        {"device_id": device_id},
        {"$push": {"past_trips": trip_data}}
    )

# 📜 Get past trips
def get_past_trips(device_id):
    user = db.users.find_one({"device_id": device_id})
    return user["past_trips"] if user else []