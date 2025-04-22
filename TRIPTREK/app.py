from flask import Flask, render_template, request, redirect, url_for, jsonify
import google.generativeai as genai
import requests
import urllib.parse
import random
from math import ceil
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
from functools import wraps, lru_cache
import time
from flask_caching import Cache
import math

app = Flask(__name__)
# Configure caching
cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})
cache.init_app(app)
DEFAULT_CACHE_TIMEOUT = 3600 
load_dotenv()  # This must come BEFORE accessing os.getenv()

# Now you can safely access your environment variables
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)  

# --- Decorator for caching API calls ---
def cache_api_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
        result = cache.get(cache_key)
        if result is None:
            result = func(*args, **kwargs)
            cache.set(cache_key, result, timeout=DEFAULT_CACHE_TIMEOUT)
        return result
    return wrapper

# --- Helper Functions ---
def format_budget_split(split):
    """Format the budget split for display in prompt"""
    if not split:
        return ""
    items = []
    for category, amount in split.items():
        percentage = (amount / sum(split.values())) * 100
        items.append(f"- {category.capitalize()}: ₹{amount:,} ({percentage:.1f}%)")
    return "Budget Allocation:\n" + "\n".join(items)

def get_gemini_plan(source, destination, people, budget, split, days):
    ai_plan = ""
    try:
        # First validate if budget is feasible
        distance = get_distance(source, destination)
        min_budget = calculate_minimum_budget(people, distance, days)
        
        if budget < min_budget:
            warning_html = (f"""
            <div class='budget-warning'>
                ⚠️ <strong>Budget Warning</strong><br>
                The budget of ₹{budget:,} is too low for this {days}-day trip for {people} people.<br>
                Minimum required budget is ₹{min_budget:,}.<br>
                Please increase your budget or adjust your trip parameters.
            </div>
            """)
            return warning_html, None, min_budget

        # Generate the travel plan
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        
        # Enhanced prompt with clear structure
        prompt = f"""
        Create a concise {days}-day itinerary from {source} to {destination} for {people} people with a total budget of ₹{budget:,} (₹{budget//people:,} per person).
        
        **Important Note:**If the user's original budget of ₹{budget:,} was too low for this trip. This plan shows what's possible with the minimum required budget.
        **Format Requirements:**
        - Use this exact structure for each day:
        
        ### Day [X]: [Title]
        **Travel:** [Mode] from [Location] to [Location] (₹Cost)
        **Stay:** [Hotel] (₹Cost/night)
        **Meals:** 
          - Breakfast: [Place] (₹Cost)
          - Lunch: [Place] (₹Cost)
          - Dinner: [Place] (₹Cost)
        **Activities:**
          - [Activity 1] (₹Cost)
          - [Activity 2] (₹Cost)
        **Daily Total:** ₹XXXX
        
        **Additional Requirements:**
        - Keep each day's description under 150 words
        - Only include 2-3 key activities per day
        - Show exact costs for everything
        - Total all costs at the end
        - Budget allocation: {format_budget_split(split)}
        
        **Example:**
        ### Day 1: Arrival & Exploration
        **Travel:** Flight from Delhi to Goa (₹4,500/person)
        **Stay:** Taj Holiday Village (₹6,000/night)
        **Meals:**
          - Breakfast: Cafe Bodega (₹350)
          - Lunch: Fisherman's Wharf (₹600)
          - Dinner: Pousada (₹800)
        **Activities:**
          - Beach walk (Free)
          - Chapora Fort visit (₹50)
        **Daily Total:** ₹11,800
        """

        response = model.generate_content(prompt)
        ai_plan = clean_ai_response(response.text)
        
        warning_html = (f"""
            <div class='budget-warning'>
                ⚠️ <strong>Budget Warning</strong><br>
                Your original budget of ₹{budget:,} was too low for this {days}-day trip for {people} people.<br>
                We've generated a plan using the minimum required budget of ₹{min_budget:,}.<br>
                You may want to consider increasing your budget for more comfort/options.
            </div>
            """)
        if budget < min_budget:
            return warning_html, None, min_budget
        else:
            ai_plan = clean_ai_response(response.text)
            ai_plan += f"\n\n**Minimum Required Budget:** ₹{min_budget:,}\n**Estimated Cost:** ₹[TOTAL]\n**Remaining:** ₹[REMAINING]"
            return ai_plan, min_budget, min_budget


    except Exception as e:
        error_html = (f"""
        <div class='error-message'>
            ❌ <strong>Error generating plan</strong><br>
            {str(e)}
        </div>
        """)
        return error_html, None, None

    
def clean_ai_response(text):
    # Remove markdown headers and convert to HTML
    text = text.replace('### ', '<h3>').replace('\n\n', '</h3>\n')
    text = text.replace('**', '<strong>').replace('**', '</strong>')
    
    # Format lists properly
    text = text.replace('- ', '<li>')
    text = text.replace('\n', '<br>')
    
    # Ensure proper HTML structure
    text = f"<div class='itinerary-container'>{text}</div>"
    
    return text

def calculate_minimum_budget(people_count, distance_km, days):
    """Calculate minimum required budget with per-day costs"""
    # Transport costs
    if distance_km <= 300:
        min_travel = 1000 * (1 if people_count <= 2 else 1.5)  # Car/bus
    elif distance_km <= 700:
        min_travel = 2000 * people_count  # Train
    else:
        min_travel = 5000 * people_count  # Flight
    
    # Accommodation (per night)
    min_hotel = 1500 * people_count * days  # ₹1500 per person per night
    
    # Food (3 meals per day)
    min_food = 600 * people_count * days  # ₹200 per meal
    
    # Basic activities
    min_activities = 500 * people_count * days
    
    return ceil(min_travel + min_hotel + min_food + min_activities)

def create_maps_link(place):
    escaped = urllib.parse.quote_plus(place)
    return f'<a href="https://www.google.com/maps/search/?api=1&query={escaped}" target="_blank">{place}</a>'

@cache_api_call
def get_coordinates(place):
    geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": place,
        "key": GOOGLE_MAPS_API_KEY
    }
    response = requests.get(geocode_url, params=params).json()
    if response["results"]:
        location = response["results"][0]["geometry"]["location"]
        return location["lat"], location["lng"]
    else:
        raise ValueError(f"Could not find coordinates for: {place}")

def get_distance(source, destination):
    try:
        source_coords = get_coordinates(source)
        dest_coords = get_coordinates(destination)
        distance_km, _ = get_distance_and_duration(source_coords, dest_coords)
        return distance_km
    except Exception:
        return 300  # Default distance if API fails

@cache_api_call
def get_distance_and_duration(source_coords, dest_coords):
    matrix_url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": f"{source_coords[0]},{source_coords[1]}",
        "destinations": f"{dest_coords[0]},{dest_coords[1]}",
        "units": "metric",
        "key": GOOGLE_MAPS_API_KEY
    }
    response = requests.get(matrix_url, params=params).json()
    if response["rows"] and response["rows"][0]["elements"][0]["status"] == "OK":
        element = response["rows"][0]["elements"][0]
        distance_km = float(element["distance"]["value"]) / 1000  # Convert meters to km
        duration = element["duration"]["text"]
        return round(distance_km, 2), duration
    else:
        raise ValueError("Could not calculate distance or duration")
    
@cache_api_call
def get_nearest_airport(coords):
    try:
        places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "location": f"{coords[0]},{coords[1]}",
            "radius": 50000,  # 50km radius
            "type": "airport",
            "key": GOOGLE_MAPS_API_KEY
        }
        response = requests.get(places_url, params=params).json()
        
        if not response.get('results'):
            return None
            
        # Get the closest airport
        closest = min(
            response['results'],
            key=lambda x: x.get('distance', float('inf')))
            
        return {
            'name': closest['name'],
            'code': closest['name'].split()[-1],  # Assumes last word is code (e.g., "Delhi DEL")
            'distance': closest.get('distance', 'N/A')
        }
    except Exception:
        return None
        
def calculate_cab_fare(distance_km):
    """Estimate cab fare based on distance"""
    base_fare = 100  # Minimum fare
    per_km = 15      # ₹15 per km
    return base_fare + (distance_km * per_km)
    
def get_primary_recommendation(distance_km, duration, budget):
    """Determine the best travel mode based on distance and budget"""
    if distance_km <= 300:  # Short distance
        if budget >= 5000:
            return {
                'mode': 'cab',
                'reason': 'Comfortable private transport for short distances',
                'links': get_cab_links("Current Location", "Destination", distance_km)
            }
        else:
            return {
                'mode': 'bus',
                'reason': 'Economical option for short distances',
                'links': get_bus_links("Current Location", "Destination")
            }
    elif distance_km <= 700:  # Medium distance
        return {
            'mode': 'train',
            'reason': 'Best balance of comfort and cost for medium distances',
            'links': get_train_links("Current Location", "Destination")
        }
    else:  # Long distance
        if budget >= 10000:
            return {
                'mode': 'flight',
                'reason': 'Fastest option for long distances',
                'links': get_flight_links("Current Location", "Destination", None, None)
            }
        else:
            return {
                'mode': 'train',
                'reason': 'Most economical for long distances',
                'links': get_train_links("Current Location", "Destination")
            }
            

def suggest_travel_mode(source, destination, travel_budget):
    try:
        # Get coordinates for both locations
        source_coords = get_coordinates(source)
        dest_coords = get_coordinates(destination)
        
        # Get distance and duration
        distance_km, duration = get_distance_and_duration(source_coords, dest_coords)
        
        # Generate all travel options
        options = {
            'flights': get_flight_links(source, destination, source_coords, dest_coords),
            'trains': get_train_links(source, destination),
            'buses': get_bus_links(source, destination),
            'cabs': get_cab_links(source, destination, distance_km),
            'rentals': get_rental_links(source)
        }
        
        # Determine primary recommendation
        recommendation = get_primary_recommendation(distance_km, duration, travel_budget)
        
        return format_travel_recommendation(
            source, 
            destination,
            distance_km,
            duration,
            recommendation,
            options,
            travel_budget
        )
        
    except Exception as e:
        return f"""<div class="error">
            <p>⚠️ Could not fetch travel options</p>
            <p><small>{str(e)}</small></p>
        </div>"""

# --- Transportation Link Functions ---

def get_flight_links(source, destination, src_coords, dest_coords):
    """Generate flight search links with nearby airports"""
    airports = {
        'source': get_nearest_airport(src_coords) if src_coords else {
            'name': f"{source} Airport",
            'code': source[:3].upper()  # Create a fake code from first 3 letters
        },
        'destination': get_nearest_airport(dest_coords) if dest_coords else {
            'name': f"{destination} Airport", 
            'code': destination[:3].upper()
        }
    }
    links = []
    
    # Main flight search links
    if airports['source'] and airports['destination']:
        flight_query = f"{airports['source']['code']}-{airports['destination']['code']}"
        links.extend([
            {
                'name': f'Google Flights ({airports["source"]["code"]}→{airports["destination"]["code"]})',
                'url': f"https://www.google.com/travel/flights?q=Flights+to+{flight_query}",
                'type': 'search'
            },
            {
                'name': 'MakeMyTrip',
                'url': f"https://www.makemytrip.com/flight/search?itinerary={flight_query}",
                'type': 'booking'
            },
            {
                'name': 'Skyscanner',
                'url': f"https://www.skyscanner.co.in/transport/flights/{flight_query}/",
                'type': 'compare'
            }
        ])
    
    # Alternative flight options
    links.extend([
        {
            'name': 'Air India',
            'url': "https://www.airindia.in/",
            'type': 'airline'
        },
        {
            'name': 'IndiGo',
            'url': "https://www.goindigo.in/",
            'type': 'airline'
        }
    ])
    
    return links

def get_train_links(source, destination):
    """Generate train booking links"""
    return [
        {
            'name': 'IRCTC Official',
            'url': f"https://www.irctc.co.in/nget/train-search?src={source}&dest={destination}",
            'type': 'booking'
        },
        {
            'name': 'ConfirmTkt',
            'url': f"https://www.confirmtkt.com/train/{source}-to-{destination}-trains",
            'type': 'search'
        },
        {
            'name': 'RailYatri',
            'url': f"https://www.railyatri.in/trains/{source}-to-{destination}",
            'type': 'info'
        }
    ]

def get_bus_links(source, destination):
    """Generate bus booking links"""
    return [
        {
            'name': 'RedBus',
            'url': f"https://www.redbus.in/bus-tickets/{source}-to-{destination}",
            'type': 'booking'
        },
        {
            'name': 'Abhibus',
            'url': f"https://www.abhibus.com/{source}-to-{destination}-bus-tickets",
            'type': 'booking'
        },
        {
            'name': 'State Transport',
            'url': f"https://www.google.com/search?q={source}+to+{destination}+govt+bus",
            'type': 'info'
        }
    ]

def get_cab_links(source, destination, distance_km):
    """Generate cab booking links with fare estimates"""
    base_fare = calculate_cab_fare(distance_km)
    return [
        {
            'name': f'Uber (₹{base_fare}-₹{base_fare*1.5})',
            'url': f"https://m.uber.com/ul/?action=setPickup&pickup=my_location&dropoff[formatted_address]={destination}",
            'type': 'booking'
        },
        {
            'name': f'Ola (₹{base_fare-500}-₹{base_fare*1.3})',
            'url': f"https://book.olacabs.com/?drop={destination}",
            'type': 'booking'
        },
        {
            'name': 'Local Taxi Services',
            'url': f"https://www.google.com/search?q=taxi+{source}+to+{destination}",
            'type': 'info'
        }
    ]

def get_rental_links(location):
    """Generate car rental links"""
    return [
        {
            'name': 'Zoomcar',
            'url': f"https://www.zoomcar.com/search?city={location}",
            'type': 'rental'
        },
        {
            'name': 'Revv',
            'url': f"https://www.revv.co.in/{location.replace(' ', '-')}-self-drive-cars/",
            'type': 'rental'
        },
        {
            'name': 'Local Rentals',
            'url': f"https://www.google.com/search?q=car+rental+{location}",
            'type': 'info'
        }
    ]

# --- Helper Functions ---
def format_recommendations(recommendations):
    """Format recommendations into HTML sections"""
    html = ""
    for rec in recommendations:
        # Format places if they exist
        places_html = ""
        if 'places' in rec and rec['places']:
            places_html = f"""
            <div class="places-container">
                {"".join(rec['places'])}
            </div>
            """
        
        # Format links if they exist
        links_html = ""
        if 'links' in rec and rec['links']:
            links_html = format_links(rec['links'])
        
        html += f"""
        <div class="recommendation-section">
            <h3 class="section-title">
                {rec['emoji']} {rec['category'].capitalize()} (Budget: {rec['budget']})
            </h3>
            {places_html}
            {links_html}
        </div>
        """
    return html

def format_travel_recommendation(source, dest, distance, duration, recommendation, options, budget):
    """Format all options into HTML"""
    html = f"""
    <div class="travel-recommendations">
        <h3>Travel Options from {source} to {dest}</h3>
        <div class="meta-info">
            <span>📏 {distance} km</span>
            <span>⏱️ {duration}</span>
            <span>💰 Budget: ₹{budget}</span>
        </div>
        
        <div class="primary-recommendation">
            <h4>🌟 Recommended: {recommendation['mode'].capitalize()}</h4>
            <p>{recommendation['reason']}</p>
            {format_links(recommendation['links'], 'primary')}
        </div>
    """
    
    # Add each transportation section
    for transport_type in ['flights', 'trains', 'buses', 'cabs', 'rentals']:
        if options[transport_type]:
            html += f"""
            <div class="option-type {transport_type}">
                <h4>{get_transport_icon(transport_type)} {transport_type.capitalize()}</h4>
                {format_links(options[transport_type])}
            </div>
            """
    
    html += "</div>"
    return html

def format_links(links, style='normal'):
    """Format links list into HTML"""
    if not links:
        return ""
        
    html = "<div class='links-container'>"
    for link in links:
        html += f"""
        <a href="{link['url']}" 
           target="_blank" 
           class="travel-link {link.get('type', '')} {style}"
           title="{link.get('description', '')}">
            {link['name']}
        </a>
        """
    html += "</div>"
    return html

def get_transport_icon(transport_type):
    icons = {
        'flights': '✈️',
        'trains': '🚆',
        'buses': '🚌',
        'cabs': '🚖',
        'rentals': '🚗'
    }
    return icons.get(transport_type, '🔗')

def divide_budget(budget, priorities, people_count, distance_km=300):
    """
    Allocate budget including shopping and a combined 'explore' category.
    """
    if budget <= 0 or people_count <= 0 or distance_km < 0:
        raise ValueError("Invalid input values")

    base_allocations = {
        'travel': allocate_travel_budget(distance_km),
        'hotel': 0.25,
        'food': 0.20,
        'Other Activities': 0.15,
        'shopping': 0.10  # Explicitly added
    }

    remaining_percentage = 1 - sum(base_allocations.values())
    remaining_budget = budget * remaining_percentage

    priority_boost_percent = 0.05
    for i, priority in enumerate(priorities[:4]):
        priority = priority.lower()
        if priority in base_allocations:
            base_allocations[priority] += priority_boost_percent
        else:
            base_allocations[priority] = priority_boost_percent

    allocation = {k: budget * v for k, v in base_allocations.items()}

    if 'hotel' in allocation:
        allocation['hotel'] *= people_count

    allocation = {k: ceil(v) for k, v in allocation.items()}

    min_allocation = ceil(0.05 * budget)
    for category in allocation:
        if allocation[category] < min_allocation:
            allocation[category] = min_allocation

    total_allocated = sum(allocation.values())
    if total_allocated > budget:
        adjust_budget(allocation, total_allocated - budget)

    return allocation

def allocate_travel_budget(distance_km):
    """Determine travel budget percentage based on distance"""
    if distance_km <= 300:       # Short distance (road)
        return 0.15
    elif distance_km <= 700:     # Medium distance (train)
        return 0.25
    else:                        # Long distance (flight)
        return 0.35

def adjust_budget(allocation, excess_amount):
    """
    Reduce allocation amounts to remove excess, starting with largest non-priority categories
    """
    # Sort categories by amount (descending) excluding priority categories
    adjustable_cats = sorted(
        [(k, v) for k, v in allocation.items() if k not in ['travel', 'hotel']],
        key=lambda x: x[1],
        reverse=True
    )
    
    # Reduce amounts until we've covered the excess
    remaining_excess = excess_amount
    for cat, amount in adjustable_cats:
        reduction = min(remaining_excess, allocation[cat] - ceil(0.05 * sum(allocation.values())))
        allocation[cat] -= reduction
        remaining_excess -= reduction
        if remaining_excess <= 0:
            break
    
    # If still excess, adjust priority categories
    if remaining_excess > 0:
        for cat in ['hotel', 'travel']:
            if cat in allocation:
                reduction = min(remaining_excess, allocation[cat])
                allocation[cat] -= reduction
                remaining_excess -= reduction
                if remaining_excess <= 0:
                    break
def get_links_for_hotels(destination):
    """Get direct booking links for hotels in the destination"""
    return [
        {
            'name': 'Booking.com',
            'url': f"https://www.booking.com/searchresults.en-gb.html?ss={urllib.parse.quote_plus(destination)}",
            'type': 'booking'
        },
        {
            'name': 'MakeMyTrip Hotels',
            'url': f"https://www.makemytrip.com/hotels/hotel-listing/?checkin=date&checkout=date&city=CT{destination.replace(' ', '%20')}",
            'type': 'booking'
        },
        {
            'name': 'Agoda',
            'url': f"https://www.agoda.com/search?city={urllib.parse.quote_plus(destination)}",
            'type': 'booking'
        },
        {
            'name': 'OYO Rooms',
            'url': f"https://www.oyorooms.com/search?location={urllib.parse.quote_plus(destination)}",
            'type': 'budget'
        }
    ]

def get_links_for_restaurants(destination):
    """Get discovery links for restaurants and cafes"""
    return [
        {
            'name': 'Zomato',
            'url': f"https://www.zomato.com/{destination.lower().replace(' ', '-')}/restaurants",
            'type': 'discovery'
        },
        {
            'name': 'EazyDiner',
            'url': f"https://www.eazydiner.com/{destination.lower().replace(' ', '-')}",
            'type': 'reservation'
        },
        {
            'name': 'Dineout',
            'url': f"https://www.dineout.co.in/{destination.lower().replace(' ', '-')}-restaurants",
            'type': 'reservation'
        },
        {
            'name': 'Google Restaurants',
            'url': f"https://www.google.com/search?q=best+restaurants+in+{urllib.parse.quote_plus(destination)}",
            'type': 'discovery'
        }
    ]

def get_links_for_shopping(destination):
    """Get links for shopping centers and local markets"""
    return [
        {
            'name': 'Local Markets Guide',
            'url': f"https://www.google.com/search?q=best+shopping+markets+in+{urllib.parse.quote_plus(destination)}",
            'type': 'guide'
        },
        {
            'name': 'Shopping Malls',
            'url': f"https://www.google.com/maps/search/shopping+malls+in+{urllib.parse.quote_plus(destination)}",
            'type': 'maps'
        },
        {
            'name': 'Local Handicrafts',
            'url': f"https://www.google.com/search?q=local+handicrafts+in+{urllib.parse.quote_plus(destination)}",
            'type': 'specialty'
        },
        {
            'name': 'Fabindia Stores',
            'url': f"https://www.google.com/maps/search/fabindia+in+{urllib.parse.quote_plus(destination)}",
            'type': 'brand'
        }
    ]

def get_links_for_activities(destination):
    """Get links for activities and attractions"""
    return [
        {
            'name': 'TripAdvisor Things to Do',
            'url': f"https://www.tripadvisor.com/Attractions-g297604-Activities-{destination.replace(' ', '_')}.html",
            'type': 'discovery'
        },
        {
            'name': 'Thrillophilia',
            'url': f"https://www.thrillophilia.com/cities/{destination.lower().replace(' ', '-')}/trekking-activities",
            'type': 'adventure'
        },
        {
            'name': 'Local Parks & Gardens',
            'url': f"https://www.google.com/maps/search/parks+and+gardens+in+{urllib.parse.quote_plus(destination)}",
            'type': 'nature'
        },
        {
            'name': 'Cultural Sites',
            'url': f"https://www.google.com/search?q=cultural+sites+in+{urllib.parse.quote_plus(destination)}",
            'type': 'culture'
        }
    ]

def get_links_for_transport(destination):
    """Get links for local transport options"""
    return [
        {
            'name': 'Metro/Bus Routes',
            'url': f"https://www.google.com/search?q=public+transport+in+{urllib.parse.quote_plus(destination)}",
            'type': 'public'
        },
        {
            'name': 'Car Rentals',
            'url': f"https://www.google.com/search?q=car+rentals+in+{urllib.parse.quote_plus(destination)}",
            'type': 'rental'
        },
        {
            'name': 'Auto/Taxi Guide',
            'url': f"https://www.google.com/search?q=taxi+rates+in+{urllib.parse.quote_plus(destination)}",
            'type': 'guide'
        }
    ]
    
@cache_api_call
def get_places(destination, *place_types):
    """Get high-quality places with proper links and rating filtering"""
    try:
        # Get coordinates first
        geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {"address": destination, "key": GOOGLE_MAPS_API_KEY}
        response = requests.get(geocode_url, params=params).json()
        
        if not response.get('results'):
            return []
            
        location = response['results'][0]['geometry']['location']
        lat, lng = location['lat'], location['lng']
        
        # Search for places with strict quality filters
        places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        places = []
        
        for ptype in place_types:
            params = {
                "location": f"{lat},{lng}",
                "radius": 5000,  # 5km radius
                "type": ptype,
                "key": GOOGLE_MAPS_API_KEY,
                "rankby": "prominence",  # Get most popular places first
                "min_price": 2,          # Mid-range to high-end places
                "max_price": 4           # (1-4 scale)
            }
            
            response = requests.get(places_url, params=params).json()
            if 'results' in response:
                # Strict filtering - minimum 4.3 rating and 100+ reviews
                filtered = [
                    p for p in response['results'] 
                    if p.get('rating', 0) >= 4.3 and 
                    p.get('user_ratings_total', 0) >= 100
                ]
                places.extend(filtered[:5])  # Get top 5 per category
        
        # Get detailed info including website for each place
        detailed_places = []
        place_details_url = "https://maps.googleapis.com/maps/api/place/details/json"
        
        for place in sorted(places, key=lambda x: x.get('rating', 0), reverse=True):
            try:
                details_params = {
                    "place_id": place['place_id'],
                    "fields": "name,rating,user_ratings_total,vicinity,website,url,formatted_phone_number",
                    "key": GOOGLE_MAPS_API_KEY
                }
                details_response = requests.get(place_details_url, params=details_params).json()
                
                if details_response.get('result'):
                    detailed_places.append(details_response['result'])
            except Exception as e:
                print(f"Error getting details for place: {str(e)}")
                continue
        
        # Format results with proper links
        formatted = []
        seen_names = set()
        
        for place in detailed_places:
            name = place.get('name', '')
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            
            rating = place.get('rating', 'N/A')
            address = place.get('vicinity', '')
            website = place.get('website', '')
            maps_link = place.get('url', f"https://www.google.com/maps/place/?q=place_id:{place['place_id']}")
            phone = place.get('formatted_phone_number', '')
            
            # Create booking links for hotels
            booking_link = ""
            if 'lodging' in place_types or 'hotel' in str(place_types):
                booking_link = f"""<a href="https://www.booking.com/searchresults.en-gb.html?ss={urllib.parse.quote_plus(name + ' ' + destination)}" 
                    target="_blank" class="booking-link">Book on Booking.com</a>"""
            
            place_card = f"""
            <div class="place-card">
                <div class="place-header">
                    <a href="{maps_link}" target="_blank" class="place-link">
                        <strong>{name}</strong> ⭐ {rating} ({place.get('user_ratings_total', '?')} reviews)
                    </a>
                </div>
                <div class="place-body">
                    <div class="place-address">{address}</div>
                    {f'<div class="place-phone"><i class="fas fa-phone"></i> {phone}</div>' if phone else ''}
                    <div class="place-links">
                        {f'<a href="{website}" target="_blank" class="place-website">Official Website</a>' if website else ''}
                        {booking_link}
                    </div>
                </div>
            </div>
            """
            formatted.append(place_card)
        
        return formatted
        
    except Exception as e:
        print(f"Error getting places: {str(e)}")
        return []
    
def generate_budget_visualization(total_budget, split, days=1, people=1):
    """Generate complete budget visualization with daily and per-person breakdown"""
    if not split:
        return "<div>No budget allocation data available</div>"
    
    colors = ['#4361ee', '#4895ef', '#4cc9f0', '#560bad', '#7209b7', '#f72585']
    categories = list(split.keys())
    amounts = [split[cat] for cat in categories]
    percentages = [(amount / total_budget * 100) for amount in amounts]
    
    # Generate pie chart
    pie_chart = generate_pie_chart_svg(percentages, colors, amounts, total_budget)
    
    # Generate detailed breakdown with daily and per-person calculations
    breakdown_items = []
    for i, (category, amount) in enumerate(split.items()):
        daily_amount = amount / days if days > 0 else 0
        per_person = amount / people if people > 0 else 0
        
        breakdown_items.append(f"""
        <div class="budget-breakdown-item">
            <div class="category-color" style="background:{colors[i % len(colors)]};"></div>
            <div class="category-name">{get_category_emoji(category)} {category}</div>
            <div class="category-amount">₹{amount:,}</div>
            <div class="category-daily">₹{daily_amount:,.0f}/day</div>
            <div class="category-per-person">₹{per_person:,.0f}/person</div>
            <div class="category-percentage">{percentages[i]:.1f}%</div>
        </div>
        """)
    
    return f"""
    <div class="budget-visualization-container">
        <div class="budget-header">
            <h3>Total Budget: ₹{total_budget:,}</h3>
            <div class="budget-subheader">
                <span>For {days} days</span>
                <span>For {people} people (₹{total_budget//people:,}/person)</span>
            </div>
        </div>
        
        <div class="budget-charts">
            <div class="chart-container pie-chart">
                <h4>Budget Allocation</h4>
                {pie_chart}
            </div>
        </div>
        
        <div class="budget-breakdown">
            <h4>Detailed Budget Breakdown</h4>
            <div class="breakdown-header">
                <span>Category</span>
                <span>Total</span>
                <span>Daily</span>
                <span>Per Person</span>
                <span>%</span>
            </div>
            {"".join(breakdown_items)}
        </div>
    </div>
    """
    
def get_category_emoji(category):
    """Return appropriate emoji for each budget category"""
    emoji_map = {
        "travel": "✈️",
        "hotel": "🏨",
        "food": "🍽️",
        "activities": "🎡",
        "shopping": "🛍️",
        "transport": "🚗",
        "sightseeing": "🏛️",
        "other": "✨"
    }
    return emoji_map.get(category.lower(), "📌")


# Update generate_pie_chart_svg() function to accept budget parameter:
def generate_pie_chart_svg(percentages, colors, amounts, budget):
    """Generate SVG pie chart"""
    svg_size = 200
    center = svg_size // 2
    radius = svg_size // 2 - 10
    
    # Generate SVG path commands for each segment
    cumulative_percent = 0
    path_commands = []
    
    for i, percent in enumerate(percentages):
        if percent == 0:
            continue
            
        start_angle = cumulative_percent * 3.6  # Convert percentage to degrees
        end_angle = (cumulative_percent + percent) * 3.6
        cumulative_percent += percent
        
        # Calculate start and end coordinates
        x1 = center + radius * math.cos(math.radians(start_angle - 90))
        y1 = center + radius * math.sin(math.radians(start_angle - 90))
        x2 = center + radius * math.cos(math.radians(end_angle - 90))
        y2 = center + radius * math.sin(math.radians(end_angle - 90))
        
        # Large arc flag (1 if angle > 180, 0 otherwise)
        large_arc = 1 if (end_angle - start_angle) > 180 else 0
        
        path_commands.append(
            f'<path d="M {center} {center} L {x1} {y1} A {radius} {radius} 0 {large_arc} 1 {x2} {y2} Z" '
            f'fill="{colors[i % len(colors)]}" stroke="#ffffff" stroke-width="1" />'
        )
    
    return f"""
    <svg width="{svg_size}" height="{svg_size}" viewBox="0 0 {svg_size} {svg_size}">
        {"".join(path_commands)}
        <circle cx="{center}" cy="{center}" r="{radius * 0.4}" fill="#ffffff" />
        <text x="{center}" y="{center}" text-anchor="middle" dominant-baseline="middle" 
          font-size="24" font-weight="bold" fill="#4361ee">₹{budget:,}</text>
    </svg>
    """

def generate_bar_chart_svg(categories, amounts, colors, total_budget):
    """Generate SVG bar chart"""
    svg_width = 300
    svg_height = 200
    padding = 40
    bar_width = 30
    max_amount = max(amounts)
    scale_factor = (svg_height - 2 * padding) / max_amount
    
    # Generate bars
    bars = []
    for i, (category, amount) in enumerate(zip(categories, amounts)):
        x = padding + i * (bar_width + 10)
        height = amount * scale_factor
        y = svg_height - padding - height
        
        bars.append(f"""
        <g class="bar-group">
            <rect x="{x}" y="{y}" width="{bar_width}" height="{height}" 
                  fill="{colors[i % len(colors)]}" rx="3" />
            <text x="{x + bar_width/2}" y="{y - 5}" text-anchor="middle" 
                  font-size="10">{amount:,}</text>
            <text x="{x + bar_width/2}" y="{svg_height - padding + 15}" 
                  text-anchor="middle" font-size="10" transform="rotate(45 {x + bar_width/2} {svg_height - padding + 15})">
                {category[:8]}
            </text>
        </g>
        """)
    
    return f"""
    <svg width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">
        <line x1="{padding}" y1="{svg_height - padding}" x2="{svg_width - padding}" y2="{svg_height - padding}" 
              stroke="#cccccc" stroke-width="1" />
        {"".join(bars)}
    </svg>
    """

def format_section(title, items):
    if not items:
        return ""
    html = f"<h2 class='glow-heading'>{title}</h2><ul>"
    for item in items:
        html += f"<li>{item}</li>"
    html += "</ul>"
    return html

# --- Routes ---
db = MongoClient("mongodb+srv://triptrekuser:TrikTrek3!@cluster0.seub3ya.mongodb.net/triptrek?retryWrites=true&w=majority&appName=Cluster0").triptrek

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

@app.route('/')
def home():
    return render_template("triptrek.html")

@app.route('/manual')
def manual():
    return render_template("manual.html")

@app.route('/triptrek')
def triptrek():
    return render_template("triptrek.html")

@app.route('/Automated')
def Automated():
    return render_template("Automated.html")

@app.route('/plan')
def plan():
    return render_template("plan.html")

from flask import request, jsonify
from bson import ObjectId
import bcrypt
from datetime import datetime


@app.route('/login', methods=['POST'])
def login():
    try:
        # Get email and password from request
        if request.is_json:
            email = request.json.get('email')
            password = request.json.get('password')
            device_id = request.json.get('device_id')  # Keeping device_id as in original
        else:
            email = request.form.get('email')
            password = request.form.get('password')
            device_id = request.form.get('device_id')  # Keeping device_id as in original
        
        # Validate inputs - keeping device_id check from original
        if not device_id:
            return jsonify({"error": "Device ID is required"}), 400
        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400
            
        # Find user in database
        user = users_collection.find_one({"email": email})
        
        if user:
            # Existing user - verify password
            if not check_password_hash(user['password'], password):
                return jsonify({"error": "Invalid password"}), 401
        else:
            # New user - create account
            new_user = {
                "email": email,
                "password": generate_password_hash(password),
                "device_id": device_id,  # Keeping device_id as in original
                "past_trips": [],
                "created_at": datetime.utcnow()
            }
            users_collection.insert_one(new_user)
            user = users_collection.find_one({"email": email})
            
        # Return user data (excluding password) - keeping original response structure
        return jsonify({
            "user_id": str(user["_id"]),
            "device_id": user.get("device_id", device_id),  # Maintain device_id in response
            "past_trips": user.get("past_trips", []),  # Keeping past_trips as in original
            "message": "Login successful" if user else "New account created"
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# 💌 Save a trip
# Fix the save_trip_route function to properly handle JSON data:
@app.route('/save-trip', methods=['POST'])
def save_trip_route():
    try:
        if request.is_json:
            data = request.json
            device_id = data.get('device_id')
            trip_data = data.get('trip_data')
        else:
            device_id = request.form.get('device_id')
            trip_data = request.form.get('trip_data')
        
        if not device_id or not trip_data:
            return jsonify({"error": "Device ID and trip data are required"}), 400
            
        save_trip(device_id, trip_data)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🗂️ Get past trips
@app.route('/past-trips', methods=['POST'])
def past_trips():
    return jsonify(get_past_trips(request.form.get('device_id')))

@app.route('/recommend', methods=['POST'])
def recommend():
    start_time = time.time()
    
    try:
        # Validate required fields
        required_fields = ['source', 'destination', 'budget', 'people', 'days']
        for field in required_fields:
            if field not in request.form or not request.form[field].strip():
                return render_template("error.html", 
                    message=f"{field.capitalize()} is required",
                    title="Missing Information"), 400

        # Get form data
        source = request.form['source'].strip() + ", India"
        destination = request.form['destination'].strip() + ", India"
        
        try:
            budget = int(request.form['budget'])
            people = int(request.form['people'])
            days = int(request.form['days'])
            
            if budget <= 0 or people <= 0 or days <= 0:
                raise ValueError("Values must be positive")
        except ValueError:
            return render_template("error.html", 
                message="Please enter valid numbers for budget, people, and days",
                title="Invalid Input"), 400

        # Get priorities
        priorities = [
            request.form.get('priority1', '').lower(),
            request.form.get('priority2', '').lower(),
            request.form.get('priority3', '').lower(),
            request.form.get('priority4', '').lower()
        ]
        priorities = [p for p in priorities if p]

        # Get coordinates and distance (cached)
        src_coords = get_coordinates(source)
        dest_coords = get_coordinates(destination)
        distance_km, _ = get_distance_and_duration(src_coords, dest_coords)

        # Calculate budget split
        split = divide_budget(budget, priorities, people, distance_km)
        
        # Generate content in parallel where possible
        ai_plan, _, _ = get_gemini_plan(source, destination, people, budget, split, days)
        travel_suggestion = suggest_travel_mode(source, destination, split.get("travel", 0))
        
        # Generate recommendations
        recommendations = generate_all_recommendations(destination, split,days,people)

        # Calculate processing time
        processing_time = round(time.time() - start_time, 2)
        
        budget_visual_html = generate_budget_visualization(budget, split, days, people)

        # ✅ Do the formatting outside first:
        formatted_budget = f"{budget:,}"

        # ✅ Then return:
        return render_template("recommendations.html",
            ai_plan=ai_plan,
            rec=recommendations,
            travel_suggestion=travel_suggestion,
            source=source.replace(", India", ""),
            destination=destination.replace(", India", ""),
            budget=budget,  # keep as number for logic
            formatted_budget=formatted_budget,  # for display
            people=people,
            days=days,
            priorities=priorities,
            processing_time=processing_time,
            split=split,
            recommendations=recommendations,
            budget_visual_html=budget_visual_html
        )


    except Exception as e:
        app.logger.error(f"Error in recommend route: {str(e)}")
        return render_template("error.html", 
            message="We encountered an issue while planning your trip",
            title="Planning Error"), 500

def get_category_emoji(category):
    emoji_map = {
        "hotel": "🏨",
        "food": "🍽️",
        "shopping": "🛍️",
        "Other Activities": "🏛️",
        "travel": "✈️"
    }
    return emoji_map.get(category, "📍")

@cache_api_call
def get_tourist_places(destination, *place_types):
    """Get high-quality tourist places with proper links and rating filtering"""
    try:
        # Get coordinates first
        geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {"address": destination, "key": GOOGLE_MAPS_API_KEY}
        response = requests.get(geocode_url, params=params).json()
        
        if not response.get('results'):
            return []
            
        location = response['results'][0]['geometry']['location']
        lat, lng = location['lat'], location['lng']
        
        # Search for places with strict tourist-focused filters
        places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        places = []
        
        for ptype in place_types:
            params = {
                "location": f"{lat},{lng}",
                "radius": 10000,  # 10km radius
                "type": ptype,
                "key": GOOGLE_MAPS_API_KEY,
                "rankby": "prominence",  # Get most popular places first
                "min_price": 2,          # Mid-range to high-end places
                "max_price": 4           # (1-4 scale)
            }
            
            response = requests.get(places_url, params=params).json()
            if 'results' in response:
                # Strict filtering - minimum 4.0 rating and 50+ reviews for tourist spots
                filtered = [
                    p for p in response['results'] 
                    if p.get('rating', 0) >= 4.0 and 
                    p.get('user_ratings_total', 0) >= 50
                ]
                places.extend(filtered[:8])  # Get top 8 per category
        
        # Get detailed info including website for each place
        detailed_places = []
        place_details_url = "https://maps.googleapis.com/maps/api/place/details/json"
        
        for place in sorted(places, key=lambda x: x.get('rating', 0), reverse=True):
            try:
                details_params = {
                    "place_id": place['place_id'],
                    "fields": "name,rating,user_ratings_total,vicinity,website,url,formatted_phone_number,photos,types",
                    "key": GOOGLE_MAPS_API_KEY
                }
                details_response = requests.get(place_details_url, params=details_params).json()
                
                if details_response.get('result'):
                    detailed_places.append(details_response['result'])
            except Exception as e:
                print(f"Error getting details for place: {str(e)}")
                continue
        
        # Format results with proper links and tourist-friendly info
        formatted = []
        seen_names = set()
        
        for place in detailed_places:
            name = place.get('name', '')
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            
            rating = place.get('rating', 'N/A')
            address = place.get('vicinity', '')
            website = place.get('website', '')
            maps_link = place.get('url', f"https://www.google.com/maps/place/?q=place_id:{place['place_id']}")
            phone = place.get('formatted_phone_number', '')
            types = place.get('types', [])
            
            # Get photo if available
            photo_html = ""
            if 'photos' in place and place['photos']:
                photo_ref = place['photos'][0]['photo_reference']
                photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_ref}&key={GOOGLE_MAPS_API_KEY}"
                photo_html = f'<img src="{photo_url}" alt="{name}" class="place-photo">'
            
            # Create category tags
            categories = [t.replace('_', ' ').title() for t in types if t not in ['point_of_interest', 'establishment']]
            category_html = "".join([f'<span class="place-category">{c}</span>' for c in categories[:3]])
            
            place_card = f"""
            <div class="place-card">
                <div class="place-header">
                    <a href="{maps_link}" target="_blank" class="place-link">
                        <strong>{name}</strong> ⭐ {rating} ({place.get('user_ratings_total', '?')} reviews)
                    </a>
                </div>
                {photo_html}
                <div class="place-body">
                    <div class="place-categories">{category_html}</div>
                    <div class="place-address">{address}</div>
                    {f'<div class="place-phone"><i class="fas fa-phone"></i> {phone}</div>' if phone else ''}
                    <div class="place-links">
                        {f'<a href="{website}" target="_blank" class="place-website">Official Website</a>' if website else ''}
                        <a href="{maps_link}" target="_blank" class="place-maps">View on Map</a>
                    </div>
                </div>
            </div>
            """
            formatted.append(place_card)
        
        return formatted
        
    except Exception as e:
        print(f"Error getting tourist places: {str(e)}")
        return []
    
# --- New Helper Function ---
def generate_all_recommendations(destination, split, days, people):
    """Generate recommendations for all budget categories with tourist focus"""
    print(f"Generating recommendations for {destination} with split: {split}")
    
    # Enhanced category mapping with tourist-focused place types
    category_mapping = {
        "hotel": {
            "place_types": ["lodging", "hotel"],
            "links": get_links_for_hotels(destination),
            "description": "Recommended accommodations"
        },
        "food": {
            "place_types": ["restaurant", "cafe", "bakery", "meal_delivery", "meal_takeaway"],
            "links": get_links_for_restaurants(destination),
            "description": "Local cuisine and dining options"
        },
        "shopping": {
            "place_types": ["shopping_mall", "department_store", "clothing_store", "jewelry_store"],
            "links": get_links_for_shopping(destination),
            "description": "Shopping destinations and markets"
        },
        "Other Activities": {
            "place_types": ["tourist_attraction", "museum", "park", "amusement_park", "aquarium", "zoo"],
            "links": get_links_for_activities(destination),
            "description": "Popular attractions and activities"
        },
        "travel": {
            "place_types": [],
            "links": get_links_for_transport(destination),
            "description": "Local transportation options"
        }
    }
    
    recommendations = []
    for category, data in category_mapping.items():
        if category in split and split[category] > 0:
            # Get high-quality tourist places
            places = get_tourist_places(destination, *data["place_types"]) if data["place_types"] else []
            
            recommendations.append({
                'category': category,
                'emoji': get_category_emoji(category),
                'budget': f"₹{split[category]:,}",
                'daily_budget': f"₹{split[category]//days:,}/day" if days > 0 else "₹0",
                'per_person': f"₹{split[category]//people:,}/person" if people > 0 else "₹0",
                'places': places,
                'links': data["links"],
                'description': data["description"]
            })
    
    return recommendations


if __name__ == '__main__':
    app.run(debug=False)