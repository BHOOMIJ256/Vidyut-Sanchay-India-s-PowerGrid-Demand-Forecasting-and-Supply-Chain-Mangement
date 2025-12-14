# test_logistic.py (Final version with corrected keys)

import math
import sys
import os

# --- Setup to import LogisticsAgent correctly ---
try:
    # Running inside the directory: use direct import
    from logistics_agent import LogisticsAgent 
except ImportError:
    # If run from outside, use the full package import structure (fallback)
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from logistics_agent import LogisticsAgent
    
# Global list to track test results
TEST_RESULTS = []
TEST_COUNT = 0

# --- Helper Function for Testing ---
def run_test(test_function, test_name):
    """Executes a test function and prints the result."""
    global TEST_COUNT
    TEST_COUNT += 1
    
    try:
        test_function(LogisticsAgent())
        print(f"  ✅ PASS: {test_name}")
        TEST_RESULTS.append(True)
    except AssertionError as e:
        print(f"  ❌ FAIL: {test_name}")
        print(f"    - Assertion Error: {e}")
        TEST_RESULTS.append(False)
    except ValueError as e:
        # Check if the error raised was the expected one for invalid inputs
        if "expected_error" in test_function.__annotations__ and test_function.__annotations__["expected_error"]:
            print(f"  ✅ PASS: {test_name} (Caught expected error: {e})")
            TEST_RESULTS.append(True)
        else:
            print(f"  ❌ FAIL: {test_name}")
            print(f"    - Unexpected Error: {e}")
            TEST_RESULTS.append(False)
    except Exception as e:
        print(f"  ❌ FAIL: {test_name}")
        print(f"    - Unexpected Exception: {e}")
        TEST_RESULTS.append(False)


# ==============================================================================
# --- TEST DEFINITIONS (USING CORRECTED KEYS) ---
# ==============================================================================

def test_delivery_metrics_are_positive(agent):
    """Verifies that distance, time, and cost are all positive numbers."""
    quantity = 100.0
    report = agent.calculate_delivery(
        supplier_name="TATA_JAMSHEDPUR",  # CORRECTED KEY
        project_site_key="MUMBAI_SITE",
        quantity_tonnes=quantity,
    )
    
    assert report["distance_km"] > 0, f"Distance was not positive: {report['distance_km']}"
    assert report["transit_time_days"] > 0, f"Time was not positive: {report['transit_time_days']}"
    assert report["transport_cost_inr"] > 0, f"Cost was not positive: {report['transport_cost_inr']}"


def test_cost_scales_linearly_with_quantity(agent):
    """Verifies that doubling the quantity doubles the transport cost."""
    route_supplier = "SAIL_BHILAI"  # CORRECTED KEY
    route_site = "MUMBAI_SITE"     # Using Mumbai since Nagpur might be missing

    report_base = agent.calculate_delivery(
        supplier_name=route_supplier,
        project_site_key=route_site,
        quantity_tonnes=100.0,
    )
    cost_base = report_base["transport_cost_inr"]

    report_double = agent.calculate_delivery(
        supplier_name=route_supplier,
        project_site_key=route_site,
        quantity_tonnes=200.0,
    )
    cost_double = report_double["transport_cost_inr"]
    
    assert math.isclose(cost_double, cost_base * 2, rel_tol=1e-6), \
        f"Cost not linear. Base: {cost_base}, Double: {cost_double}"
    
    # Check that time remains unchanged
    assert math.isclose(report_base["transit_time_days"], report_double["transit_time_days"], rel_tol=1e-6), \
        "Transit time changed when only quantity changed."


def test_invalid_location_raises_value_error(agent) -> "expected_error":
    """Verifies that an unknown location key raises a ValueError."""
    try:
        agent.calculate_delivery(
            supplier_name="UNKNOWN_PLANT",
            project_site_key="MUMBAI_SITE",
            quantity_tonnes=10.0,
        )
        assert False, "ValueError was NOT raised for UNKNOWN_PLANT."
    except ValueError as e:
        assert "not found in the LOCATION_DB" in str(e), f"Wrong ValueError message: {e}"


def test_negative_quantity_raises_value_error(agent) -> "expected_error":
    """Verifies that passing a negative quantity raises a ValueError."""
    try:
        # We must use a VALID location key first so the test hits the quantity check
        agent.calculate_delivery(
            supplier_name="TATA_JAMSHEDPUR", # CORRECTED KEY
            project_site_key="MUMBAI_SITE",
            quantity_tonnes=-50.0,
        )
        assert False, "ValueError was NOT raised for negative quantity."
    except ValueError as e:
        assert "Tonnes cannot be negative" in str(e), f"Wrong ValueError message: {e}"


def test_zero_distance_route(agent):
    """Verifies self-delivery results in zero distance/cost, and fixed buffer time."""
    report = agent.calculate_delivery(
        supplier_name="MUMBAI_SITE", 
        project_site_key="MUMBAI_SITE",
        quantity_tonnes=1000.0,
    )
    
    assert math.isclose(report["distance_km"], 0.0, abs_tol=1e-6), "Distance should be zero for same location."
    assert math.isclose(report["transport_cost_inr"], 0.0, abs_tol=1e-6), "Cost should be zero for zero distance."
    # ETA should only be the fixed 2-day (48-hour) loading buffer
    assert math.isclose(report["transit_time_days"], 2.0, rel_tol=1e-6), "ETA not equal to fixed buffer time."


# ==============================================================================
# --- MAIN EXECUTION BLOCK ---
# ==============================================================================

if __name__ == "__main__":
    print("--- Running Standalone Logistics Agent Unit Tests ---")
    
    tests_to_run = [
        test_delivery_metrics_are_positive,
        test_cost_scales_linearly_with_quantity,
        test_invalid_location_raises_value_error,
        test_negative_quantity_raises_value_error,
        test_zero_distance_route,
    ]

    for test_func in tests_to_run:
        run_test(test_func, test_func.__name__)

    total_passed = sum(TEST_RESULTS)
    total_failed = TEST_COUNT - total_passed
    
    print("\n" + "="*40)
    print(f"SUMMARY: {total_passed} Passed / {total_failed} Failed / {TEST_COUNT} Total")
    print("="*40)