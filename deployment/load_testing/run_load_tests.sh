#!/bin/bash
################################################################################
# Comprehensive Load Testing Suite
#
# Runs load tests for all Decision Agent components:
# 1. Inference API load test
# 2. Streaming pipeline throughput test
# 3. Feature cache performance test
# 4. Batch inference benchmark
#
# Generates detailed reports with performance metrics.
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
API_HOST="${API_HOST:-http://localhost:8000}"
USERS="${USERS:-1000}"
SPAWN_RATE="${SPAWN_RATE:-50}"
RUN_TIME="${RUN_TIME:-5m}"
REPORT_DIR="./load_test_reports/$(date +%Y%m%d_%H%M%S)"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Decision Agent Load Testing Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Configuration:"
echo "  API Host: ${API_HOST}"
echo "  Users: ${USERS}"
echo "  Spawn Rate: ${SPAWN_RATE}"
echo "  Duration: ${RUN_TIME}"
echo "  Report Dir: ${REPORT_DIR}"
echo ""

# Create report directory
mkdir -p "${REPORT_DIR}"

# Test 1: Inference API Load Test
echo -e "${BLUE}[1/4] Running Inference API Load Test...${NC}"
locust -f api_load_test.py \
  --host="${API_HOST}" \
  --users="${USERS}" \
  --spawn-rate="${SPAWN_RATE}" \
  --run-time="${RUN_TIME}" \
  --headless \
  --csv="${REPORT_DIR}/api_load_test" \
  --html="${REPORT_DIR}/api_load_test.html"

echo -e "${GREEN}✓ Inference API load test complete${NC}"
echo ""

# Test 2: Streaming Pipeline Throughput
echo -e "${BLUE}[2/4] Running Streaming Pipeline Throughput Test...${NC}"
python streaming_throughput_test.py \
  --events-per-second=10000 \
  --duration=300 \
  --output="${REPORT_DIR}/streaming_throughput.json"

echo -e "${GREEN}✓ Streaming throughput test complete${NC}"
echo ""

# Test 3: Feature Cache Performance
echo -e "${BLUE}[3/4] Running Feature Cache Performance Test...${NC}"
python cache_performance_test.py \
  --requests=100000 \
  --cache-size=50000 \
  --output="${REPORT_DIR}/cache_performance.json"

echo -e "${GREEN}✓ Feature cache test complete${NC}"
echo ""

# Test 4: Batch Inference Benchmark
echo -e "${BLUE}[4/4] Running Batch Inference Benchmark...${NC}"
python batch_inference_benchmark.py \
  --batch-sizes="100,500,1000,5000,10000" \
  --num-samples=100000 \
  --output="${REPORT_DIR}/batch_inference.json"

echo -e "${GREEN}✓ Batch inference benchmark complete${NC}"
echo ""

# Generate summary report
echo -e "${BLUE}Generating Summary Report...${NC}"
python generate_load_test_report.py \
  --report-dir="${REPORT_DIR}" \
  --output="${REPORT_DIR}/summary.html"

echo -e "${GREEN}✓ Summary report generated${NC}"
echo ""

# Print results
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Load Test Results${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

echo "Reports generated in: ${REPORT_DIR}"
echo ""

echo "Quick Summary:"
echo ""

# Parse API load test results
if [ -f "${REPORT_DIR}/api_load_test_stats.csv" ]; then
  echo "API Load Test:"
  # Extract key metrics from CSV
  python -c "
import pandas as pd
df = pd.read_csv('${REPORT_DIR}/api_load_test_stats.csv')
total = df[df['Name'] == 'Aggregated']
if not total.empty:
    print(f'  Requests: {total[\"Request Count\"].iloc[0]:,.0f}')
    print(f'  Failures: {total[\"Failure Count\"].iloc[0]:,.0f}')
    print(f'  Median: {total[\"Median Response Time\"].iloc[0]:.1f}ms')
    print(f'  95th: {total[\"95%\"].iloc[0]:.1f}ms')
    print(f'  RPS: {total[\"Requests/s\"].iloc[0]:.1f}')
"
  echo ""
fi

# Show streaming results
if [ -f "${REPORT_DIR}/streaming_throughput.json" ]; then
  echo "Streaming Pipeline:"
  python -c "
import json
with open('${REPORT_DIR}/streaming_throughput.json') as f:
    data = json.load(f)
    print(f'  Throughput: {data[\"avg_events_per_sec\"]:,.0f} events/sec')
    print(f'  Latency: {data[\"avg_latency_ms\"]:.1f}ms')
    print(f'  Total events: {data[\"total_events\"]:,.0f}')
"
  echo ""
fi

# Show cache results
if [ -f "${REPORT_DIR}/cache_performance.json" ]; then
  echo "Feature Cache:"
  python -c "
import json
with open('${REPORT_DIR}/cache_performance.json') as f:
    data = json.load(f)
    print(f'  Hit rate: {data[\"hit_rate\"]:.1%}')
    print(f'  Avg latency (hit): {data[\"avg_hit_latency_ms\"]:.2f}ms')
    print(f'  Avg latency (miss): {data[\"avg_miss_latency_ms\"]:.2f}ms')
"
  echo ""
fi

# Show batch results
if [ -f "${REPORT_DIR}/batch_inference.json" ]; then
  echo "Batch Inference:"
  python -c "
import json
with open('${REPORT_DIR}/batch_inference.json') as f:
    data = json.load(f)
    best = min(data['results'], key=lambda x: x['latency_per_sample_ms'])
    print(f'  Best batch size: {best[\"batch_size\"]}')
    print(f'  Throughput: {best[\"throughput_per_sec\"]:,.0f} samples/sec')
    print(f'  Latency: {best[\"latency_per_sample_ms\"]:.3f}ms/sample')
"
  echo ""
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}All load tests complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "View detailed report: ${REPORT_DIR}/summary.html"
echo "View API report: ${REPORT_DIR}/api_load_test.html"
