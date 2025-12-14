"""
Configuration for the logistics agent.
"""

# config.py (UPDATED)

# We use simpler, standardized keys for easy reference in the code
LOCATION_DB = {
    # SUPPLIERS
    "TATA_JAMSHEDPUR": (22.8046, 86.2029),     # Changed key
    "JSW_BALLARI": (15.1394, 76.9214),
    "SAIL_BOKARO": (23.6693, 86.1513),
    "SAIL_BHILAI": (21.2185, 81.3807),         # ADDED MISSING BHILAI LOCATION
    
    # PROJECT SITES / CITIES
    "MUMBAI_SITE": (19.0760, 72.8777),
    "DELHI_SITE": (28.7041, 77.1025),          # Renamed for consistency
    # ... include others like CHENNAI, KOLKATA, etc.
    "NAGPUR_SITE": (21.1458, 79.0882),         # Needed for a full test suite
}

# ... LOGISTICS_SETTINGS remains the same

LOGISTICS_SETTINGS = {
    "avg_truck_speed": 45,
    "cost_per_tonne_per_km": 6.5,
    "driver_rest_hours": 10,
    "driver_shift_hours": 14,
    "loading_buffer_hours": 48,
    "road_curvature_factor": 1.3,
}

# config.py (Snippet - Add this section)

# --- 3. API Settings ---
API_SETTINGS = {
    # Replace 'YOUR_GEOAPIFY_API_KEY_HERE' with the key you obtained
    "GEOAPIFY_KEY": "e752819c238c4864a4be5df774a62240", 
    "GEOAPIFY_ROUTING_URL": "https://api.geoapify.com/v1/routing",
    
    # We will use 'truck' as the mode for heavy goods vehicle (HGV) transport
    "TRANSPORT_MODE": "truck" 
}

# --- 4. MASTER SETTINGS DICTIONARY ---
# Combine all settings into one dictionary for the agent's convenience
MASTER_SETTINGS = {**LOGISTICS_SETTINGS, **API_SETTINGS} 
# The double asterisk (**) is Python's way of merging dictionaries.