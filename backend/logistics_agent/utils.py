# utils.py
import math
from typing import Tuple

# Mean Earth radius in kilometers
EARTH_RADIUS_KM = 6371.0


def calculate_haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """Calculate great-circle distance between two coordinates using the haversine formula."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    delta_lon = lon2_rad - lon1_rad
    delta_lat = lat2_rad - lat1_rad

    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return float(EARTH_RADIUS_KM * c)


def apply_road_curvature(distance_km: float, factor: float) -> float:
    """Adjust air distance by a curvature/route factor to approximate road distance."""
    return float(distance_km * factor)


def calculate_driving_time_hours(distance_km: float, speed_kmph: float) -> float:
    """Compute raw driving time (hours)."""
    if speed_kmph <= 0:
        raise ValueError("Speed must be positive.")
    return float(distance_km / speed_kmph)


def calculate_rest_hours(total_driving_hours: float, shift_hours: float, rest_hours: float) -> float:
    """Calculate mandatory rest hours based on driving shifts."""
    if shift_hours <= 0:
        raise ValueError("Shift hours must be positive.")
    shifts = math.floor(total_driving_hours / shift_hours)
    return float(shifts * rest_hours)


def calculate_cost(distance_km: float, tonnes: float, rate: float) -> float:
    """Calculate transport cost for a single material."""
    if tonnes < 0:
        raise ValueError("Tonnes cannot be negative.")
    if rate < 0:
        raise ValueError("Rate cannot be negative.")
    # NOTE: The extra comma was removed here
    return float(distance_km * tonnes * rate)