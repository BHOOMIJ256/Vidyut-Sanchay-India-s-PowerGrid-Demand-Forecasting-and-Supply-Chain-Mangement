"""
Utility functions for Price Agent
Contains helper functions for calculations, formatting, and data processing
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

def calculate_percentage_change(base_value: float, current_value: float) -> float:
    """Calculate percentage change between two values"""
    if base_value == 0:
        return 0.0
    return ((current_value - base_value) / base_value) * 100


def format_currency(amount: float, currency: str = 'INR', decimals: int = 2) -> str:
    """
    Format amount as currency. 
    Smartly switches to Cr/Lakhs for INR if value is large.
    """
    if currency == 'INR':
        # Smart Indian Formatting
        if amount >= 10000000: # 1 Crore
            return f"₹{amount/10000000:.2f} Cr"
        elif amount >= 100000:   # 1 Lakh
            return f"₹{amount/100000:.2f} L"
        else:
            return f"₹{amount:,.{decimals}f}"
            
    elif currency == 'USD':
        return f"${amount:,.{decimals}f}"
    else:
        return f"{amount:,.{decimals}f} {currency}"


def format_large_number(number: float) -> str:
    """Format large numbers with appropriate suffixes (K, L, Cr)"""
    if number >= 10000000:  # 1 Crore
        return f"{number/10000000:.2f} Cr"
    elif number >= 100000:  # 1 Lakh
        return f"{number/100000:.2f} L"
    elif number >= 1000:    # 1 Thousand
        return f"{number/1000:.2f} K"
    else:
        return f"{number:.2f}"


def get_severity_level(change_pct: float, thresholds: Dict[str, float] = None) -> Dict[str, Any]:
    """Determine severity level based on price change percentage"""
    if thresholds is None:
        thresholds = {'critical': 10, 'high': 5, 'medium': 2}
    
    abs_change = abs(change_pct)
    
    if abs_change >= thresholds['critical']:
        return {'level': 'CRITICAL', 'icon': '🚨', 'color': 'red'}
    elif abs_change >= thresholds['high']:
        return {'level': 'HIGH', 'icon': '⚠️', 'color': 'orange'}
    elif abs_change >= thresholds['medium']:
        return {'level': 'MEDIUM', 'icon': '💡', 'color': 'yellow'}
    else:
        return {'level': 'LOW', 'icon': '✅', 'color': 'green'}


def generate_recommendation(material: str, change_pct: float, 
                            current_price: float, base_price: float) -> str:
    """Generate actionable procurement recommendation"""
    material_display = material.replace('_', ' ').title()
    
    if change_pct > 10:
        return (f"🚨 URGENT: {material_display} surged {change_pct:.1f}%. "
                f"Immediate procurement recommended before further hikes.")
    elif change_pct > 5:
        return (f"⚠️ WARNING: {material_display} up {change_pct:.1f}%. "
                f"Consider advance procurement.")
    elif change_pct < -5:
        return (f"💰 SAVINGS: {material_display} down {abs(change_pct):.1f}%. "
                f"Good time to bulk buy.")
    else:
        return f"✅ STABLE: Normal procurement."


def is_cache_valid(cache_timestamp: Optional[datetime], duration_seconds: int) -> bool:
    """Check if cached data is still valid"""
    if cache_timestamp is None:
        return False
    time_elapsed = (datetime.now() - cache_timestamp).total_seconds()
    return time_elapsed < duration_seconds


def simulate_price_fluctuation(base_price: float, volatility: float, 
                               days: int = 30, trend: float = 0.0) -> float:
    """Simulate realistic price fluctuation using random walk with trend"""
    # Daily volatility
    daily_volatility = volatility / np.sqrt(365)
    
    # Random walk
    daily_changes = np.random.normal(0, daily_volatility * base_price, days)
    cumulative_change = np.sum(daily_changes)
    
    # Trend
    trend_change = base_price * trend * (days / 365)
    
    # Ensure price doesn't go negative
    simulated_price = max(base_price * 0.5, base_price + cumulative_change + trend_change)
    return simulated_price


def validate_api_key(api_key: str, service_name: str) -> bool:
    """Validate if API key is configured (not placeholder)"""
    # Add the specific placeholders we used in config.py
    invalid_keys = [
        'YOUR_ALPHA_VANTAGE_KEY', 
        'YOUR_EXCHANGERATE_KEY', 
        'YOUR_NEWS_API_KEY',
        'YOUR_API_KEY_HERE', 
        '', 
        None, 
        'demo'
    ]
    
    if api_key in invalid_keys:
        print(f"⚠️ {service_name} API key not configured. Using simulation mode.")
        return False
    return True