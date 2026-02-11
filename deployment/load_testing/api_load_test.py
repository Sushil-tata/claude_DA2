"""
Load Testing for Decision Agent Inference API

Tests the inference API under various load conditions to identify:
- Maximum throughput
- Latency at different loads
- Error rates
- Bottlenecks

Uses Locust for distributed load generation.

Usage:
    # Run locally
    locust -f api_load_test.py --host=http://localhost:8000

    # Run distributed (master)
    locust -f api_load_test.py --host=http://localhost:8000 --master

    # Run distributed (worker)
    locust -f api_load_test.py --master-host=localhost --worker

    # Headless mode
    locust -f api_load_test.py --host=http://localhost:8000 \
        --users 1000 --spawn-rate 50 --run-time 5m --headless
"""

from locust import HttpUser, task, between, events
import json
import random
import time
from datetime import datetime

# Test configuration
CUSTOMER_IDS = [f"CUST{i:06d}" for i in range(10000)]  # 10K test customers
LATENCY_TARGETS = {
    "p50": 50,   # ms
    "p95": 100,  # ms
    "p99": 200   # ms
}


class InferenceAPIUser(HttpUser):
    """
    Simulates a user making inference requests to the API.
    """
    wait_time = between(0.1, 0.5)  # Wait 0.1-0.5 seconds between requests

    def on_start(self):
        """Called when a user starts"""
        self.customer_id = random.choice(CUSTOMER_IDS)

    @task(10)  # Weight: 10 (most common)
    def predict_customer(self):
        """
        Make prediction request for a customer.
        """
        payload = {
            "customer_id": random.choice(CUSTOMER_IDS)
        }

        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()

                # Validate response schema
                required_fields = ["customer_id", "prediction", "confidence",
                                 "model_version", "latency_ms"]
                if all(field in data for field in required_fields):
                    # Check latency
                    if data["latency_ms"] > LATENCY_TARGETS["p99"]:
                        response.failure(f"High latency: {data['latency_ms']}ms")
                    else:
                        response.success()
                else:
                    response.failure("Missing required fields in response")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(3)  # Weight: 3
    def predict_with_features(self):
        """
        Make prediction with pre-provided features.
        """
        # Generate random feature values
        features = {
            f"feature_{i}": random.gauss(0, 1)
            for i in range(20)
        }

        payload = {
            "customer_id": random.choice(CUSTOMER_IDS),
            "features": features
        }

        with self.client.post("/predict", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)  # Weight: 1 (least common)
    def health_check(self):
        """Check API health"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure(f"Unhealthy status: {data.get('status')}")
            else:
                response.failure(f"HTTP {response.status_code}")


class BurstyUser(HttpUser):
    """
    Simulates bursty traffic patterns (sudden spikes).
    """
    wait_time = between(1, 5)  # Longer wait between bursts

    @task
    def burst_requests(self):
        """Make a burst of requests"""
        num_requests = random.randint(5, 20)

        for _ in range(num_requests):
            payload = {"customer_id": random.choice(CUSTOMER_IDS)}

            self.client.post("/predict", json=payload)

            # Small delay within burst
            time.sleep(0.01)


# Event listeners for custom metrics
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts"""
    print("=" * 80)
    print("LOAD TEST STARTED")
    print("=" * 80)
    print(f"Target host: {environment.host}")
    print(f"Users: {environment.runner.target_user_count}")
    print(f"Spawn rate: {environment.runner.spawn_rate}")
    print()
    print("Latency targets:")
    for percentile, target in LATENCY_TARGETS.items():
        print(f"  {percentile}: <{target}ms")
    print("=" * 80)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops"""
    stats = environment.stats

    print()
    print("=" * 80)
    print("LOAD TEST COMPLETED")
    print("=" * 80)

    # Print summary
    print("\nSummary:")
    print(f"  Total requests: {stats.total.num_requests:,}")
    print(f"  Total failures: {stats.total.num_failures:,}")
    print(f"  Failure rate: {stats.total.fail_ratio:.2%}")
    print(f"  Average response time: {stats.total.avg_response_time:.1f}ms")
    print(f"  RPS: {stats.total.total_rps:.1f}")

    # Print percentiles
    print("\nLatency percentiles:")
    print(f"  p50: {stats.total.get_response_time_percentile(0.50):.1f}ms")
    print(f"  p95: {stats.total.get_response_time_percentile(0.95):.1f}ms")
    print(f"  p99: {stats.total.get_response_time_percentile(0.99):.1f}ms")
    print(f"  Max: {stats.total.max_response_time:.1f}ms")

    # Check if targets met
    print("\nTarget validation:")
    p50 = stats.total.get_response_time_percentile(0.50)
    p95 = stats.total.get_response_time_percentile(0.95)
    p99 = stats.total.get_response_time_percentile(0.99)

    targets_met = True
    if p50 > LATENCY_TARGETS["p50"]:
        print(f"  ✗ p50 exceeded target: {p50:.1f}ms > {LATENCY_TARGETS['p50']}ms")
        targets_met = False
    else:
        print(f"  ✓ p50 within target: {p50:.1f}ms <= {LATENCY_TARGETS['p50']}ms")

    if p95 > LATENCY_TARGETS["p95"]:
        print(f"  ✗ p95 exceeded target: {p95:.1f}ms > {LATENCY_TARGETS['p95']}ms")
        targets_met = False
    else:
        print(f"  ✓ p95 within target: {p95:.1f}ms <= {LATENCY_TARGETS['p95']}ms")

    if p99 > LATENCY_TARGETS["p99"]:
        print(f"  ✗ p99 exceeded target: {p99:.1f}ms > {LATENCY_TARGETS['p99']}ms")
        targets_met = False
    else:
        print(f"  ✓ p99 within target: {p99:.1f}ms <= {LATENCY_TARGETS['p99']}ms")

    if stats.total.fail_ratio > 0.01:  # > 1% failure rate
        print(f"  ✗ High failure rate: {stats.total.fail_ratio:.2%}")
        targets_met = False
    else:
        print(f"  ✓ Acceptable failure rate: {stats.total.fail_ratio:.2%}")

    print()
    if targets_met:
        print("✓ ALL TARGETS MET")
    else:
        print("✗ SOME TARGETS NOT MET")

    print("=" * 80)


if __name__ == "__main__":
    import os
    import subprocess

    # Run Locust with default settings
    cmd = [
        "locust",
        "-f", __file__,
        "--host", os.getenv("API_HOST", "http://localhost:8000"),
        "--users", os.getenv("USERS", "100"),
        "--spawn-rate", os.getenv("SPAWN_RATE", "10"),
        "--run-time", os.getenv("RUN_TIME", "2m"),
        "--headless"
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
