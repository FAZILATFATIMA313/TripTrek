from flask import Flask, request, jsonify
from database import get_user, save_trip, get_past_trips  # 👈 Our magic database functions!

app = Flask(__name__)

# 🎪 When user opens website
@app.route('/login', methods=['POST'])
def login():
    device_id = request.json.get('device_id')
    user = get_user(device_id)
    return jsonify({"user_id": str(user["_id"]), "past_trips": user["past_trips"]})

# 💌 Save a trip
@app.route('/save-trip', methods=['POST'])
def save_trip_route():
    save_trip(request.json.get('device_id'), request.json.get('trip_data'))
    return jsonify({"success": True})

# 🗂️ Get past trips
@app.route('/past-trips', methods=['POST'])
def past_trips():
    return jsonify(get_past_trips(request.json.get('device_id')))

if __name__ == '__main__':
    app.run(debug=True)