"""
Price Agent Package
Multi-factor intelligent price monitoring and supplier evaluation
"""

from .price_agent import PriceAgent
from .config import API_KEYS, BASE_PRICES

__version__ = "1.0.0"
__all__ = ['PriceAgent', 'API_KEYS', 'BASE_PRICES']