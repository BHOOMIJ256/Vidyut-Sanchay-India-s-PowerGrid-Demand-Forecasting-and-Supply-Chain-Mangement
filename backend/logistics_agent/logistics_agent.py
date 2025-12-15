# logistics_agent.py
from __future__ import annotations

import json # Added for clean output
from datetime import datetime, timedelta
from typing import Dict, Tuple
import requests

# Imports for data and math
# logistics_agent.py (Line 9 onwards)

# NOTE: We change these imports to direct imports for standalone execution.
# This works when running files inside the same directory.
from .config import LOCATION_DB, MASTER_SETTINGS
from .utils import (
    apply_road_curvature,
    calculate_cost,
    calculate_driving_time_hours,
    calculate_haversine_distance,
    calculate_rest_hours,
)

class LogisticsAgent:
    """
    The Logistics Agent: A deterministic calculator for a single bulk shipment.
    It provides Distance, Time, and Cost based on fixed parameters.
    """

    def __init__(self, settings: Dict[str, float] | None = None) -> None:
        self.location_db: Dict[str, Tuple[float, float]] = LOCATION_DB
        # Use MASTER_SETTINGS as the default instead of LOGISTICS_SETTINGS
        self.settings: Dict[str, float] = settings or MASTER_SETTINGS

    def get_coordinates(self, key: str) -> Tuple[float, float]:
        """Return coordinates for a known key, raising an error if not found."""
        try:
            return self.location_db[key]
        except KeyError as exc:
            # Crucial for Risk Agent integration: gives a clean error message
            raise ValueError(f"Location Key '{key}' not found in the LOCATION_DB.") from exc

    def calculate_route_metrics(self, origin_key: str, destination_key: str) -> Dict[str, float]:
        """
        Calculate road distance and duration using the Geoapify Routing API.
        
        Geoapify returns:
        - Distance (in meters)
        - Duration (in seconds, which is the raw driving time)
        """
      
        """
        Calculate road distance and duration using the Geoapify Routing API.
        Uses the standard query parameter structure (?waypoints=...) confirmed by documentation.
        """
        
        origin_coord = self.get_coordinates(origin_key)
        destination_coord = self.get_coordinates(destination_key)

        # Geoapify requires coordinates in the format: lat,lon (Latitude first)
        # The structure is: lat1,lon1|lat2,lon2
        start_location = f"{origin_coord[0]},{origin_coord[1]}" # lat,lon
        end_location = f"{destination_coord[0]},{destination_coord[1]}" # lat,lon
        
        waypoints_string = f"{start_location}|{end_location}"

        # --- Build the Query Parameters ---
        params = {
            # Use the correct waypoints format confirmed by API docs
            "waypoints": waypoints_string,
            
            # The mode should be 'truck' for heavy haulage (not 'drive')
            "mode": self.settings["TRANSPORT_MODE"], # Should be 'truck' from config
            
            "apiKey": self.settings["GEOAPIFY_KEY"],
            
            # Optional: Add details if needed, but remove to test basic functionality first
            # "details": "route_details", 
        }

        # --- CRITICAL: Use the base URL for the endpoint ---
        base_url = self.settings["GEOAPIFY_ROUTING_URL"] # e.g., https://api.geoapify.com/v1/routing

        # --- Execute API Call ---
        try:
            # requests.get handles building the full query string (?waypoints=...&mode=...)
            response = requests.get(base_url, params=params) 
            response.raise_for_status() # This will catch the 400 error if it persists

            data = response.json()
            
            # Check for valid route data
            if not data.get('features'):
                raise ValueError("Geoapify: Route could not be calculated (e.g., no road access).")

            # Data extraction logic (based on GeoJSON feature structure)
            properties = data['features'][0]['properties']
            
            # Time and distance are reported per leg or summarized for the route
            # Check for the key names based on typical Geoapify response:
            distance_km = properties['distance'] / 1000  # Distance is in meters, convert to km
            duration_seconds = properties['time']
            
            # Convert seconds to driving hours
            driving_hours = duration_seconds / 3600

            return {
                "distance_km": float(distance_km),
                "driving_hours": float(driving_hours),
            }

        except requests.exceptions.HTTPError as e:
            # Print the URL for final debugging if it fails again
            print(f"DEBUG URL: {response.url}")
            raise IOError(f"API HTTP Error: {e} - Response: {response.text}")
        except Exception as e:
            raise ValueError(f"Route Parsing Error: {e}")

    def calculate_eta(self, driving_hours: float) -> Dict[str, float | str]:
        """Compute final ETA (days) and estimated arrival date."""
        
        rest_hours = calculate_rest_hours(
            driving_hours,
            self.settings["driver_shift_hours"],
            self.settings["driver_rest_hours"],
        )
        
        # Total time = Driving + Rest + Fixed Buffer (Loading/Unloading)
        total_hours = driving_hours + rest_hours + self.settings["loading_buffer_hours"]
        
        transit_time_days = float(total_hours / 24)
        
        # Calculate arrival date based on today
        arrival_date = datetime.now().date() + timedelta(days=transit_time_days)
        
        return {
            "transit_time_days": transit_time_days,
            "est_arrival_date": arrival_date.strftime("%Y-%m-%d"),
        }

    def generate_report(
        self, supplier_key: str, project_site_key: str, quantity_tonnes: float
    ) -> Dict[str, object]:
        """Generate a full delivery report for a single material."""
        
        # 1. Route Metrics
        route_metrics = self.calculate_route_metrics(supplier_key, project_site_key)
        distance_km = route_metrics["distance_km"]
        driving_hours = route_metrics["driving_hours"]
        
        # 2. Time
        eta = self.calculate_eta(driving_hours)
        
        # 3. Cost
        rate = self.settings["cost_per_tonne_per_km"]
        total_cost = calculate_cost(distance_km, quantity_tonnes, rate)

        return {
            "origin_supplier": supplier_key,
            "destination_project": project_site_key,
            "quantity_tonnes": float(quantity_tonnes),
            "distance_km": float(distance_km),
            "transit_time_days": float(eta["transit_time_days"]),
            "est_arrival_date": eta["est_arrival_date"],
            "transport_cost_inr": float(total_cost),
            "rate_used_inr_t_km": rate,
        }

    def calculate_delivery(
        self, supplier_name: str, project_site_key: str, quantity_tonnes: float
    ) -> Dict[str, object]:
        """
        Public API (The function the Risk Agent will call)
        Computes the transport plan for one bulk material.
        """
        return self.generate_report(supplier_name, project_site_key, quantity_tonnes)


# --------------------------------------------------------------------------
# --- ENTRY POINT FOR TESTING AND DEMONSTRATION (Run this file directly) ---
# --------------------------------------------------------------------------
if __name__ == "__main__":
    agent = LogisticsAgent()

    print("--- Running Logistics Agent Test Scenario (Simulating Risk Agent Loop) ---")
    
    # 1. Set fixed project destination
    project_site = "MUMBAI_SITE"
    
    # --- Test 1: Steel from Tata (5000 tonnes) ---
    try:
        steel_tonnes = 5000.0
        # The Risk Agent's first call: 
        steel_report = agent.calculate_delivery(
            supplier_name="TATA_JAMSHEDPUR", 
            project_site_key=project_site, 
            quantity_tonnes=steel_tonnes
        )
        print("\n✅ Report 1: STEEL from TATA (Candidate 1):")
        print(json.dumps(steel_report, indent=4))
        
        # --- Test 2: Cement from SAIL (300 tonnes) ---
        cement_tonnes = 300.0
        # The Risk Agent's second call: 
        cement_report = agent.calculate_delivery(
            supplier_name="SAIL_BHILAI", 
            project_site_key=project_site, 
            quantity_tonnes=cement_tonnes
        )
        print("\n✅ Report 2: CEMENT from SAIL (Candidate 2):")
        print(json.dumps(cement_report, indent=4))

        # --- Aggregation Done by the Risk Agent ---
        total_transport_cost = steel_report['transport_cost_inr'] + cement_report['transport_cost_inr']
        print(f"\n--- RISK AGENT'S AGGREGATED TOTAL PROJECT TRANSPORT COST: ₹{total_transport_cost:,.0f} ---")

    except Exception as e:
        print(f"\n❌ FATAL ERROR DURING TEST EXECUTION. Check location_data.csv: {e}")