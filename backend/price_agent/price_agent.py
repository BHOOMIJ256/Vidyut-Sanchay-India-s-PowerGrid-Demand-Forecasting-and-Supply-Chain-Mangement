"""
Price Agent - Real-Time Price Monitoring and Analysis
Fetches commodity prices, analyzes trends, generates alerts, and calculates project costs.
"""

import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

from .config import (
    API_KEYS, BASE_PRICES, CACHE_DURATION_SECONDS,
    ALERT_THRESHOLDS, PRICE_VOLATILITY, API_ENDPOINTS, SETTINGS
)
from .utils import (
    calculate_percentage_change,
    format_currency,
    get_severity_level,
    generate_recommendation,
    is_cache_valid,
    simulate_price_fluctuation,
    validate_api_key
)

class PriceAgent:
    """
    AI Agent for monitoring commodity prices and providing procurement intelligence.
    Integrates with ML Quantity predictions to provide real-time cost estimates.
    """
    
    def __init__(self, use_live_apis: bool = None):
        """
        Initialize Price Agent
        Args:
            use_live_apis: Force True to use real APIs, False for simulation.
        """
        self.api_keys = API_KEYS
        self.base_prices = BASE_PRICES.copy()
        # Priority: Constructor Argument > Settings in Config
        self.use_live_apis = use_live_apis if use_live_apis is not None else SETTINGS['use_live_apis']
        
        # Cache for API responses
        self.cache = {}
        self.cache_duration = CACHE_DURATION_SECONDS
        
        # Statistics
        self.api_calls_made = 0
        self.cache_hits = 0
        
        print(f"🤖 Price Agent initialized")
        print(f"   Mode: {'Live APIs' if self.use_live_apis else 'Simulation'}")
        print(f"   Cache: {'Enabled' if SETTINGS['enable_caching'] else 'Disabled'}")
    
    # ==================== CORE AGENT METHODS ====================

    def get_current_prices(self) -> Dict[str, any]:
        """
        Get current market prices from all sources (Live or Simulated)
        """
        print("\n" + "="*70)
        print("💰 FETCHING CURRENT MARKET PRICES")
        print("="*70 + "\n")
        
        if self.use_live_apis:
            # 1. Fetch Exchange Rate first (needed for conversions)
            exchange_rate = self._fetch_exchange_rate()
            
            # 2. Fetch Commodities (Aluminum/Copper)
            commodity_prices = self._fetch_commodity_prices(exchange_rate)
            
            # 3. Fetch others
            steel_price = self._fetch_steel_price(exchange_rate)
            fuel_price = self._fetch_fuel_price()
        else:
            print("⚠️ Using simulated prices (set use_live_apis=True for real data)\n")
            exchange_rate = self._simulate_exchange_rate()
            commodity_prices = self._simulate_commodity_prices()
            steel_price = self._simulate_steel_price()
            fuel_price = self._simulate_fuel_price()
        
        # 4. Derive Conductor Price (Per KM) from Aluminum Price (Per Tonne)
        # Formula: (Aluminum Price * Processing Factor) * (Weight in Tonnes per KM)
        aluminum_price = commodity_prices.get('aluminum', self.base_prices['aluminum_price_per_tonne'])
        
        # Processing factor ~1.25 (Cost to turn raw ingot into cable)
        conductor_price_per_tonne = aluminum_price * 1.25
        
        # Weight conversion: kg/km -> tonnes/km
        weight_per_km_tonnes = self.base_prices.get('conductor_weight_kg_per_km', 974) / 1000.0
        
        conductor_price_per_km = conductor_price_per_tonne * weight_per_km_tonnes

        # Compile all current prices
        current_prices = {
            'steel_price_per_tonne': steel_price,
            'aluminum_price_per_tonne': aluminum_price,
            'copper_price_per_tonne': commodity_prices.get('copper', 0), # Added Copper
            
            # Derived/Base Prices
            'conductor_price_per_km': round(conductor_price_per_km, 2),
            'concrete_price_per_m3': self.base_prices['concrete_price_per_m3'],
            'insulator_price_per_unit': self.base_prices['insulator_price_per_unit'],
            
            # Equipment (From Base Prices)
            'transformer_price_unit': self.base_prices.get('transformer_price_unit', 25000000),
            'circuit_breaker_price_unit': self.base_prices.get('circuit_breaker_price_unit', 1500000),
            'bus_reactor_price_unit': self.base_prices.get('bus_reactor_price_unit', 8000000),
            
            'fuel_price_per_liter': fuel_price,
            'usd_to_inr': exchange_rate,
            'last_updated': datetime.now().isoformat()
        }
        
        print(f"\n✅ Price data updated: {current_prices['last_updated']}")
        print(f"📊 API calls made: {self.api_calls_made} | Cache hits: {self.cache_hits}")
        
        return current_prices

    # =========================================================================
    #  PASTE THIS INSIDE price_agent.py (Replace existing calculate_project_cost)
    # =========================================================================

    def compare_suppliers(self, material_type: str, quantity: float, market_base_price: float) -> Dict:
        """
        Compare suppliers based on Live Market Price + Supplier Premium.
        Returns the best supplier option (Dictionary).
        """
        # Define Supplier Database locally (or import from config)
        # In a real app, this comes from a database
        SUPPLIER_DB = {
            'steel': {
                'TATA_STEEL': {'premium': 0.04, 'transport': 1200, 'rating': 4.9},
                'JSW_STEEL':  {'premium': 0.02, 'transport': 800,  'rating': 4.7},
                'SAIL':       {'premium': 0.00, 'transport': 1500, 'rating': 4.5}
            },
            'conductor': {
                'STERLITE':   {'premium': 0.05, 'transport': 2000, 'rating': 4.8},
                'APAR_IND':   {'premium': 0.03, 'transport': 1800, 'rating': 4.6},
                'GUPTA_PWR':  {'premium': -0.02, 'transport': 1500, 'rating': 4.2}
            }
        }

        if material_type not in SUPPLIER_DB or quantity == 0:
            # If no supplier data, return standard market cost
            return {'supplier': 'Market Rate', 'total_quote': quantity * market_base_price}

        # Compare Candidates
        candidates = []
        for name, data in SUPPLIER_DB[material_type].items():
            # Formula: (Base Price * Premium) + Transport
            unit_price = market_base_price * (1 + data['premium'])
            material_cost = quantity * unit_price
            transport_cost = quantity * data['transport']
            
            total_cost = material_cost + transport_cost
            
            candidates.append({
                'supplier': name,
                'total_quote': total_cost,
                'unit_price': unit_price,
                'rating': data['rating']
            })
        
        # Select "Best" (Lowest Price for now)
        best_option = min(candidates, key=lambda x: x['total_quote'])
        return best_option

    def calculate_project_cost(self, ml_output: Dict) -> Dict:
        """
        1. Reads ML Quantities (All 7 items).
        2. Selects Best Supplier for Steel/Conductor.
        3. Applies Base Prices for Concrete/Equipment.
        4. Generates Final Budget.
        """
        # 1. Get Live Global Prices
        prices = self.get_current_prices()
        
        print("\n" + "="*70)
        print("🏗️  GENERATING COMPREHENSIVE COST REPORT")
        print("="*70)

        # 2. Extract Quantities (Handling your specific ML JSON format)
        def get_val(key):
            # Returns 0 if key is missing or empty
            return float(ml_output.get(key, {}).get("value", 0.0))

        q_steel = get_val("steel_tonnes")
        q_cond_km = get_val("conductor_km")
        q_insul = get_val("insulators_unit")
        q_concrete = get_val("concrete_cubic_meter")
        q_trans = get_val("transformers_count")
        q_break = get_val("circuit_breaker_count")
        q_react = get_val("bus_reactor_count")

        # 3. Calculate Costs & Select Suppliers

        # --- A. STEEL (Supplier Selection) ---
        steel_deal = self.compare_suppliers('steel', q_steel, prices['steel_price_per_tonne'])
        cost_steel = steel_deal['total_quote']

        # --- B. CONDUCTOR (Supplier Selection) ---
        # Note: prices['conductor_price_per_km'] is derived from Live Aluminum API
        cond_deal = self.compare_suppliers('conductor', q_cond_km, prices['conductor_price_per_km'])
        cost_conductor = cond_deal['total_quote']

        # --- C. OTHER MATERIALS (Standard Rates) ---
        cost_insul = q_insul * prices['insulator_price_per_unit']
        cost_concrete = q_concrete * prices['concrete_price_per_m3']
        
        # --- D. EQUIPMENT (Standard Rates) ---
        cost_trans = q_trans * prices['transformer_price_unit']
        cost_break = q_break * prices['circuit_breaker_price_unit']
        cost_react = q_react * prices['bus_reactor_price_unit']

        # 4. Total Calculation
        subtotal = (cost_steel + cost_conductor + cost_insul + cost_concrete + 
                    cost_trans + cost_break + cost_react)
        contingency = subtotal * 0.05
        grand_total = subtotal + contingency

        # 5. Print Detailed Table
        print(f"{'ITEM':<20} | {'QTY':<10} | {'SUPPLIER/SOURCE':<15} | {'COST (INR)':>15}")
        print("-" * 70)
        print(f"{'Steel':<20} | {q_steel:<8.1f} T | {steel_deal['supplier']:<15} | {format_currency(cost_steel):>15}")
        print(f"{'Conductor':<20} | {q_cond_km:<8.1f} km| {cond_deal['supplier']:<15} | {format_currency(cost_conductor):>15}")
        print(f"{'Insulators':<20} | {q_insul:<8.0f} U | {'Base Rate':<15} | {format_currency(cost_insul):>15}")
        print(f"{'Concrete':<20} | {q_concrete:<8.1f} m3| {'Local Mix':<15} | {format_currency(cost_concrete):>15}")
        print(f"{'Transformers':<20} | {q_trans:<8.0f} U | {'BHEL/CGL':<15} | {format_currency(cost_trans):>15}")
        print(f"{'Circuit Breakers':<20} | {q_break:<8.0f} U | {'Siemens/ABB':<15} | {format_currency(cost_break):>15}")
        print(f"{'Bus Reactors':<20} | {q_react:<8.0f} U | {'BHEL/CGL':<15} | {format_currency(cost_react):>15}")
        print("-" * 70)
        print(f"{'GRAND TOTAL':<49} | {format_currency(grand_total)}")
        print("=" * 70 + "\n")

        return {
            "grand_total": grand_total,
            "steel_supplier": steel_deal['supplier'],
            "conductor_supplier": cond_deal['supplier']
        }

    def analyze_price_changes(self, current_prices: Dict) -> Dict:
        """Analyze if current prices are significantly different from Base Prices"""
        alerts = []
        
        # Compare Steel
        steel_change = calculate_percentage_change(
            self.base_prices['steel_price_per_tonne'], 
            current_prices['steel_price_per_tonne']
        )
        if abs(steel_change) > ALERT_THRESHOLDS['medium']:
            alerts.append(f"Steel price is {steel_change:+.1f}% vs Baseline")

        # Compare Aluminum
        alu_change = calculate_percentage_change(
            self.base_prices['aluminum_price_per_tonne'],
            current_prices['aluminum_price_per_tonne']
        )
        if abs(alu_change) > ALERT_THRESHOLDS['medium']:
            alerts.append(f"Aluminum price is {alu_change:+.1f}% vs Baseline")

        return {
            "status": "VOLATILE" if len(alerts) > 0 else "STABLE",
            "alerts": alerts
        }

    # ==================== API FETCHING METHODS ====================
    
    def _fetch_exchange_rate(self) -> float:
        """Fetch real-time USD to INR exchange rate"""
        cache_key = 'exchange_rate'
        
        if SETTINGS['enable_caching'] and self._check_cache(cache_key):
            self.cache_hits += 1
            print("📦 Using cached exchange rate")
            return self.cache[cache_key]['data']
        
        if not validate_api_key(self.api_keys['exchangerate_api'], 'ExchangeRate API'):
            return self._simulate_exchange_rate()
        
        try:
            url = f"{API_ENDPOINTS['exchangerate_api_base']}/{self.api_keys['exchangerate_api']}/latest/USD"
            response = requests.get(url, timeout=SETTINGS['request_timeout'])
            data = response.json()
            
            if data['result'] == 'success':
                inr_rate = data['conversion_rates']['INR']
                self.api_calls_made += 1
                self._update_cache(cache_key, inr_rate)
                print(f"✅ Fetched live USD/INR rate: ₹{inr_rate:.2f}")
                return inr_rate
        except Exception as e:
            print(f"❌ Error fetching exchange rate: {e}")
        
        return self._simulate_exchange_rate()
    
    def _fetch_commodity_prices(self, usd_to_inr: float) -> Dict[str, float]:
        """Fetch Aluminum and Copper using Alpha Vantage"""
        cache_key = 'commodity_prices'
        
        if SETTINGS['enable_caching'] and self._check_cache(cache_key):
            self.cache_hits += 1
            print("📦 Using cached commodity prices")
            return self.cache[cache_key]['data']
            
        if not validate_api_key(self.api_keys.get('alpha_vantage'), 'Alpha Vantage'):
            return self._simulate_commodity_prices()

        try:
            print(f"🔄 Connecting to Alpha Vantage (Aluminum & Copper)...")
            base_url = API_ENDPOINTS.get('alpha_vantage_base', "https://www.alphavantage.co") + "/query"
            
            def get_price(function_name):
                params = {
                    "function": function_name,
                    "interval": "monthly", 
                    "apikey": self.api_keys['alpha_vantage']
                }
                res = requests.get(base_url, params=params, timeout=SETTINGS['request_timeout'])
                data = res.json()
                if "data" in data and len(data["data"]) > 0:
                    return float(data["data"][0]["value"]) # Returns USD price
                return None

            alu_usd = get_price("ALUMINUM")
            cop_usd = get_price("COPPER") 
            
            if alu_usd and cop_usd:
                alu_inr = alu_usd * usd_to_inr
                cop_inr = cop_usd * usd_to_inr

                result = {
                    'aluminum': round(alu_inr, 2),
                    'copper': round(cop_inr, 2)
                }
                
                self.api_calls_made += 2
                self._update_cache(cache_key, result)
                print(f"✅ Live Global Prices Fetched: Aluminum=${alu_usd:.2f}, Copper=${cop_usd:.2f}")
                return result

        except Exception as e:
            print(f"❌ Error fetching commodities: {e}")
            
        return self._simulate_commodity_prices()

    def _fetch_fuel_price(self) -> float:
        """Fetch WTI Crude Oil as a proxy for Fuel Trends"""
        if not validate_api_key(self.api_keys.get('alpha_vantage'), 'Alpha Vantage'):
            return self._simulate_fuel_price()
            
        try:
            url = API_ENDPOINTS.get('alpha_vantage_base', "https://www.alphavantage.co") + "/query"
            params = {
                "function": "WTI", 
                "interval": "monthly", 
                "apikey": self.api_keys['alpha_vantage']
            }
            response = requests.get(url, params=params, timeout=SETTINGS['request_timeout'])
            data = response.json()
            if "data" in data and len(data["data"]) > 0:
                oil_usd = float(data["data"][0]["value"])
                # Factor: When Oil is $70, Diesel in India is approx ₹90. Factor ~1.28
                estimated_diesel_inr = oil_usd * 1.28
                print(f"✅ Fetched WTI Crude: ${oil_usd:.2f} (Est. Diesel: ₹{estimated_diesel_inr:.2f})")
                return round(estimated_diesel_inr, 2)
        except Exception:
            pass
        return self._simulate_fuel_price()
    
    def _fetch_steel_price(self, usd_to_inr: float) -> float:
        print("⚠️ Using simulated steel prices (World Bank API has monthly delays)")
        return self._simulate_steel_price()
    
    # ==================== SIMULATION METHODS ====================
    
    def _simulate_exchange_rate(self) -> float:
        return simulate_price_fluctuation(self.base_prices['usd_to_inr'], PRICE_VOLATILITY['usd_to_inr'])
    
    def _simulate_commodity_prices(self) -> Dict[str, float]:
        aluminum = simulate_price_fluctuation(self.base_prices['aluminum_price_per_tonne'], PRICE_VOLATILITY['aluminum_price_per_tonne'], trend=0.03)
        return {'aluminum': round(aluminum, 2), 'copper': 850000} # Fallback Copper
    
    def _simulate_steel_price(self) -> float:
        return simulate_price_fluctuation(self.base_prices['steel_price_per_tonne'], PRICE_VOLATILITY['steel_price_per_tonne'], trend=0.05)
    
    def _simulate_fuel_price(self) -> float:
        return simulate_price_fluctuation(self.base_prices['fuel_price_per_liter'], PRICE_VOLATILITY['fuel_price_per_liter'], trend=-0.02)
    
    # ==================== CACHE MANAGEMENT ====================
    
    def _check_cache(self, key: str) -> bool:
        if key not in self.cache: return False
        return is_cache_valid(self.cache[key]['timestamp'], self.cache_duration)
    
    def _update_cache(self, key: str, data: any):
        self.cache[key] = {'data': data, 'timestamp': datetime.now()}
    
    def clear_cache(self):
        self.cache = {}
        print("🗑️ Cache cleared")