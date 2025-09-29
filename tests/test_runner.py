#!/usr/bin/env python3
"""
Test runner for AI Cybersecurity Tool
Runs all tests with proper configuration and reporting
"""

import unittest
import sys
import os
import time
import requests
from io import StringIO

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_api_availability():
    """Check if API server is running"""
    try:
        response = requests.get("http://localhost:5001/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def run_unit_tests():
    """Run unit tests"""
    print("🧪 Running Unit Tests...")
    print("=" * 50)

    # Discover and run unit tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern="unit_test.py")

    runner = unittest.TextTestRunner(verbosity=2, stream=StringIO())
    result = runner.run(suite)

    return (
        result.wasSuccessful(),
        len(result.testsRun),
        len(result.failures),
        len(result.errors),
    )


def run_integration_tests():
    """Run integration tests"""
    print("\n🔗 Running Integration Tests...")
    print("=" * 50)

    if not check_api_availability():
        print("⚠️  API server not running. Skipping integration tests.")
        print("   Start API with: python api/app.py")
        return False, 0, 0, 0

    # Discover and run integration tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern="integration_test.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return (
        result.wasSuccessful(),
        len(result.testsRun),
        len(result.failures),
        len(result.errors),
    )


def run_performance_tests():
    """Run performance tests"""
    print("\n⚡ Running Performance Tests...")
    print("=" * 50)

    if not check_api_availability():
        print("⚠️  API server not running. Skipping performance tests.")
        return False, 0, 0, 0

    try:
        import test_comprehensive_threats

        print("Running comprehensive threat detection tests...")
        test_comprehensive_threats.test_threat_detection()
        return True, 1, 0, 0
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False, 1, 0, 1


def generate_test_report(
    unit_success,
    unit_total,
    unit_failures,
    unit_errors,
    integration_success,
    integration_total,
    integration_failures,
    integration_errors,
    performance_success,
    performance_total,
    performance_failures,
    performance_errors,
):
    """Generate test report"""
    print("\n📊 Test Report")
    print("=" * 50)

    total_tests = unit_total + integration_total + performance_total
    total_failures = unit_failures + integration_failures + performance_failures
    total_errors = unit_errors + integration_errors + performance_errors
    total_success = total_tests - total_failures - total_errors

    print(
        f"Unit Tests:        {unit_total:3d} tests, {unit_failures:2d} failures, {unit_errors:2d} errors"
    )
    print(
        f"Integration Tests: {integration_total:3d} tests, {integration_failures:2d} failures, {integration_errors:2d} errors"
    )
    print(
        f"Performance Tests: {performance_total:3d} tests, {performance_failures:2d} failures, {performance_errors:2d} errors"
    )
    print("-" * 50)
    print(
        f"Total:             {total_tests:3d} tests, {total_failures:2d} failures, {total_errors:2d} errors"
    )

    success_rate = (total_success / total_tests * 100) if total_tests > 0 else 0
    print(f"Success Rate:      {success_rate:.1f}%")

    if total_failures == 0 and total_errors == 0:
        print("\n✅ All tests passed!")
        return True
    else:
        print(f"\n❌ {total_failures + total_errors} test(s) failed")
        return False


def main():
    """Main test runner"""
    print("🚀 AI Cybersecurity Tool - Test Suite")
    print("=" * 50)
    print(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Run unit tests
    unit_success, unit_total, unit_failures, unit_errors = run_unit_tests()

    # Run integration tests
    integration_success, integration_total, integration_failures, integration_errors = (
        run_integration_tests()
    )

    # Run performance tests
    performance_success, performance_total, performance_failures, performance_errors = (
        run_performance_tests()
    )

    # Generate report
    overall_success = generate_test_report(
        unit_success,
        unit_total,
        unit_failures,
        unit_errors,
        integration_success,
        integration_total,
        integration_failures,
        integration_errors,
        performance_success,
        performance_total,
        performance_failures,
        performance_errors,
    )

    print(f"\nTest completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Exit with appropriate code
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()
