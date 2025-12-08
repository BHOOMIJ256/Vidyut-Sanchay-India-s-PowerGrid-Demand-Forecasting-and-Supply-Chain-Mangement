"""
Configuration file for Price Agent
Contains API keys, base prices, and settings
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file (if exists)
load_dotenv()

# ==================== API KEYS ====================
# CRITICAL: Do not hardcode real keys here if pushing to GitHub. Use .env file.
API_KEYS = {
    # Alpha Vantage - Commodity prices (Aluminum, Copper, Oil)
    # Get key: https://www.alphavantage.co/support/#api-key
    'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY', 'YOUR_ALPHA_VANTAGE_KEY'),
    
    # ExchangeRate-API - USD to INR conversion
    # Get key: https://www.exchangerate-api.com/
    'exchangerate_api': os.getenv('EXCHANGERATE_API_KEY', 'YOUR_EXCHANGERATE_KEY'),
    
    # NewsAPI - (Optional) For future sentiment analysis
    'news_api': os.getenv('NEWS_API_KEY', 'YOUR_NEWS_API_KEY'),
}

# ==================== BASE PRICES ====================
# Reference prices as of 2025 (in INR)
# Used for fallbacks or items where live API is not available (like Steel/Hardware)
BASE_PRICES = {
    # --- RAW MATERIALS ---
    'steel_price_per_tonne': 52000,      # Structural Steel (Towers)
    'aluminum_price_per_tonne': 225000,  # Raw Aluminum (Fallback)
    'concrete_price_per_m3': 6000,       # M20/M25 Grade
    'insulator_price_per_unit': 5500,    # Polymer/Porcelain String
    
    # --- ELECTRICAL EQUIPMENT (Estimates for 132kV System) ---
    # These match your ML Model outputs (Count -> Cost)
    'transformer_price_unit': 25000000,    # ~2.5 Crores (132/33kV Power Transformer)
    'bus_reactor_price_unit': 8000000,     # ~80 Lakhs
    'circuit_breaker_price_unit': 1500000, # ~15 Lakhs (SF6 Breaker)
    
    # --- CONVERSION FACTORS ---
    # Needed to convert ML output (Length) to Market Unit (Weight)
    # ACSR Panther Conductor approx weight = 974 kg/km
    'conductor_weight_kg_per_km': 974,     
    
    # --- OTHER COSTS ---
    'fuel_price_per_liter': 96.50,       # Diesel (Transport/Logistics)
    'usd_to_inr': 84.00,                 # Fallback Exchange Rate
}

# ==================== CACHE SETTINGS ====================
CACHE_DURATION_SECONDS = 3600  # 1 hour cache to save API credits

# ==================== ALERT THRESHOLDS ====================
# If Live Price deviates from Base Price by this %, trigger alert
ALERT_THRESHOLDS = {
    'critical': 15,  # % change (e.g., War impacts oil)
    'high': 10,
    'medium': 5
}

# ==================== VOLATILITY FACTORS ====================
# Used only if API fails and we must simulate data
PRICE_VOLATILITY = {
    'steel_price_per_tonne': 0.08,       
    'aluminum_price_per_tonne': 0.12,    
    'conductor_price_per_km': 0.10,      
    'fuel_price_per_liter': 0.15,        
    'usd_to_inr': 0.06,                  
}

# ==================== API ENDPOINTS ====================
API_ENDPOINTS = {
    'alpha_vantage_base': 'https://www.alphavantage.co', # Removed /query to be flexible
    'exchangerate_api_base': 'https://v6.exchangerate-api.com/v6',
    'news_api_base': 'https://newsapi.org/v2',
}

# ==================== PRICE AGENT SETTINGS ====================
SETTINGS = {
    'use_live_apis': True,   # <--- ENABLED BY DEFAULT NOW
    'enable_caching': True,
    'log_level': 'INFO',
    'max_retries': 3,
    'request_timeout': 15,   # Increased to 15s for slower APIs
}

# ==================== DISPLAY SETTINGS ====================
DISPLAY = {
    'currency_symbol': '₹',
    'decimal_places': 2,
    'show_progress': True,
}