import requests
import datetime

API_KEY = "BqpUHnfUfwKsLqgbPUNjeiKdoOIbamhw"     
API_SECRET = "SGVxUAMkfpxSA2pA"                 
EXCHANGE_RATES = {
    "USD": 83.0,  # Example: 1 USD = 83 INR
    "EUR": 90.0,  # Example: 1 EUR = 90 INR
    "GBP": 105.0, # Example: 1 GBP = 105 INR
    "AED": 22.5,  # Example: 1 AED = 22.5 INR
    "INR": 1.0    # INR to INR remains the same
}

# Allowed seat types
SEAT_TYPES = ["ECONOMY", "BUSINESS", "FIRST"]

def get_access_token():
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "client_credentials",
        "client_id": API_KEY,
        "client_secret": API_SECRET
    }
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        print("❌ Error getting API token:", response.json())
        exit()

def convert_to_inr(price, currency):
    return round(price * EXCHANGE_RATES.get(currency, 1.0), 2)

def get_flights(origin, destination, departure_date, max_price, seat_type, adults, children, airline=None):
    access_token = get_access_token()
    url = (f"https://test.api.amadeus.com/v2/shopping/flight-offers?"
           f"originLocationCode={origin}&destinationLocationCode={destination}&departureDate={departure_date}" 
           f"&adults={adults}&children={children}&travelClass={seat_type}&nonStop=false")
    
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)
    data = response.json()
    
    if "errors" in data:
        print("❌ Error:", data["errors"])
        return
    
    flights = data.get("data", [])
    if not flights:
        print("❌ No flights found.")
        return
    
    valid_flights = []
    for flight in flights:
        base_price = float(flight["price"]["total"])
        currency = flight["price"]["currency"]
        converted_price = convert_to_inr(base_price, currency)
        total_price = converted_price * (adults + children)
        
        airline_code = flight["validatingAirlineCodes"][0] if "validatingAirlineCodes" in flight else "Unknown"
        duration = flight["itineraries"][0]["duration"]
        booking_link = flight.get("deepLink", f"https://www.google.com/travel/flights?q=flights%20from%20{origin}%20to%20{destination}%20on%20{departure_date}")
        
        if total_price <= max_price:
            valid_flights.append({
                "airline": airline_code,
                "duration": duration,
                "price": total_price,
                "booking_link": booking_link
            })
    
    if not valid_flights:
        print(f"\n❌ No flights found within ₹{max_price} budget. Increasing budget by ₹5000 and retrying...")
        max_price += 5000
        get_flights(origin, destination, departure_date, max_price, seat_type, adults, children, airline)
        return
    
    print(f"\n✈️ Available Flights (within ₹{max_price} budget):\n")
    for i, flight in enumerate(valid_flights, 1):
        print(f"🔹 Flight {i}: {origin} → {destination}")
        print(f"   ✈️ Airline: {flight['airline']}")
        print(f"   🕒 Duration: {flight['duration']}")
        print(f"   💰 Total Price for {adults} Adults & {children} Children: ₹{flight['price']}")
        print(f"   🔗 Book here: {flight['booking_link']}\n")

# Get user input
origin = input("Enter origin airport code (e.g., DEL for Delhi): ").upper()
destination = input("Enter destination airport code (e.g., BOM for Mumbai): " ).upper()
departure_date = input("Enter departure date (YYYY-MM-DD): ")

try:
    datetime.datetime.strptime(departure_date, "%Y-%m-%d")
except ValueError:
    print("❌ Invalid date format. Please use YYYY-MM-DD.")
    exit()

while True:
    seat_type = input("Enter seat type (Economy, Business, First): ").upper()
    if seat_type in SEAT_TYPES:
        break
    print("❌ Invalid seat type. Choose from: Economy, Business, First")

while True:
    try:
        adults = int(input("Enter number of adults: "))
        if adults >= 1:
            break
        print("❌ Must be at least 1 adult.")
    except ValueError:
        print("❌ Invalid input. Enter a number.")

while True:
    try:
        children = int(input("Enter number of children (0 if none): "))
        if children >= 0:
            break
        print("❌ Cannot be negative.")
    except ValueError:
        print("❌ Invalid input. Enter a number.")

while True:
    try:
        max_price = float(input("Enter maximum budget (in INR): "))
        if max_price > 0:
            break
        print("❌ Budget must be greater than 0.")
    except ValueError:
        print("❌ Invalid budget. Enter a number.")

airline = input("Enter preferred airline code (or press Enter to skip): ").upper() or None

# Fetch flights
get_flights(origin, destination, departure_date, max_price, seat_type, adults, children, airline)
