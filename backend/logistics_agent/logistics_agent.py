import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LogisticsAgent:
    def __init__(self):
        # 1. Try fetching from Environment
        env_key = os.getenv("GEOAPIFY_API_KEY")
        
        # 2. If Env fails, use the backup key (Hackathon Fix)
        # This ensures the agent NEVER thinks it's offline if the key exists.
        if env_key:
            self.api_key = env_key
        else:
            # Using the key you provided in config.py
            self.api_key = "e752819c238c4864a4be5df774a62240"
            print("   [Logistics] Using Hardcoded Backup Key")

        # Simple offline database for fallbacks
        self.location_db = {
            "SKIPPER_KOLKATA": {"lat": 22.5726, "lon": 88.3639},
            "TATA_JAMSHEDPUR": {"lat": 22.8046, "lon": 86.2029},
            "SAIL_BHILAI": {"lat": 21.1938, "lon": 81.3509},
            "JSW_MUMBAI": {"lat": 19.0760, "lon": 72.8777},
            "MUMBAI_SITE": {"lat": 19.0760, "lon": 72.8777},
            "NAGPUR_SITE": {"lat": 21.1458, "lon": 79.0882},
        }

    def calculate_delivery(self, supplier_name, project_site_key, quantity_tonnes):
        """
        Calculates delivery metrics.
        """
        # 1. Try Online API First (Most Accurate)
        if self.api_key:
            try:
                # Clean inputs: "Kolkata, India" is good, "SKIPPER_KOLKATA" needs cleaning
                origin_query = supplier_name.replace("_", " ") if "India" not in supplier_name else supplier_name
                dest_query = project_site_key.replace("_", " ") if "India" not in project_site_key else project_site_key
                
                return self._get_route_from_api(origin_query, dest_query, quantity_tonnes)
            except Exception as e:
                print(f"   [Logistics] API Call Failed: {e}")
                # Don't crash, just print error and try DB

        # 2. Offline Fallback
        return self._get_route_from_db(supplier_name, project_site_key, quantity_tonnes)

    def _get_route_from_api(self, origin, destination, qty):
        # A. Geocode Origin
        origin_coords = self._geocode(origin)
        if not origin_coords: raise ValueError(f"Could not geocode Origin: {origin}")
        
        # B. Geocode Destination
        dest_coords = self._geocode(destination)
        if not dest_coords: raise ValueError(f"Could not geocode Destination: {destination}")
        
        # C. Calculate Route (Routing API)
        url = f"https://api.geoapify.com/v1/routing?waypoints={origin_coords['lat']},{origin_coords['lon']}|{dest_coords['lat']},{dest_coords['lon']}&mode=truck&apiKey={self.api_key}"
        
        resp = requests.get(url)
        if resp.status_code != 200:
            raise Exception(f"Routing API Error ({resp.status_code}): {resp.text}")
            
        data = resp.json()
        if 'features' not in data or not data['features']:
             raise Exception("No route found by API")

        route_props = data['features'][0]['properties']
        
        dist_km = route_props['distance'] / 1000.0
        time_hours = route_props['time'] / 3600.0
        
        # Commercial Adjustments 
        transit_days = (time_hours / 12.0) + 1.0  
        
        # Cost Logic (₹6.5 per Ton per Km)
        rate_per_ton_km = 6.5
        cost = dist_km * qty * rate_per_ton_km
        
        arrival_date = datetime.now() + timedelta(days=transit_days)
        
        return {
            "origin_supplier": origin,
            "destination_project": destination,
            "quantity_tonnes": qty,
            "distance_km": round(dist_km, 2),
            "transit_time_days": round(transit_days, 1),
            "est_arrival_date": arrival_date.strftime("%Y-%m-%d"),
            "transport_cost_inr": round(cost, 2),
            "source": "Live API"
        }

    def _geocode(self, location_name):
        url = "https://api.geoapify.com/v1/geocode/search"
        params = {"text": location_name, "apiKey": self.api_key, "limit": 1}
        
        try:
            resp = requests.get(url, params=params)
            if resp.status_code == 200 and resp.json()['features']:
                props = resp.json()['features'][0]['properties']
                return {"lat": props['lat'], "lon": props['lon']}
        except Exception:
            return None
        return None

    def _get_route_from_db(self, origin, dest, qty):
        # Check if keys exist
        if origin not in self.location_db:
            # If we are here, API failed AND DB failed.
            raise ValueError(f"Location Key '{origin}' not found in LOCATION_DB and API failed.")
            
        if dest not in self.location_db:
             # Try to find a generic site key if specific one fails
            if "SITE" in dest and "MUMBAI" in dest: dest = "MUMBAI_SITE"
            elif "SITE" in dest and "NAGPUR" in dest: dest = "NAGPUR_SITE"
            else: raise ValueError(f"Location Key '{dest}' not found in LOCATION_DB.")

        # Calculate "As the crow flies" distance
        lat1, lon1 = self.location_db[origin]['lat'], self.location_db[origin]['lon']
        lat2, lon2 = self.location_db[dest]['lat'], self.location_db[dest]['lon']
        
        # Approx: 1 degree lat = 111km
        dist_km = ((lat2 - lat1)**2 + (lon2 - lon1)**2)**0.5 * 100.0
        
        transit_days = (dist_km / 400.0) + 2 
        cost = dist_km * qty * 6.0 
        
        return {
            "origin_supplier": origin,
            "destination_project": dest,
            "quantity_tonnes": qty,
            "distance_km": round(dist_km, 2),
            "transit_time_days": round(transit_days, 1),
            "est_arrival_date": (datetime.now() + timedelta(days=transit_days)).strftime("%Y-%m-%d"),
            "transport_cost_inr": round(cost, 2),
            "source": "Offline DB"
        }