#!/usr/bin/env python3
"""
Experiment 5: Performance benchmarking for tool calling
Measures latency, throughput, and reliability with concurrency support
"""

from openai import OpenAI
import httpx
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from statistics import mean, median, stdev
from typing import List, Dict, Optional
from threading import Lock
from dataclasses import dataclass

TOOLS = [{
    "type": "function",
    "function": {
        "name": "process_data",
        "description": "Process a data string",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "Data to process"
                },
                "operation": {
                    "type": "string",
                    "enum": ["analyze", "transform", "validate"],
                    "description": "Operation to perform"
                }
            },
            "required": ["data", "operation"]
        }
    }
}]

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs"""
    client: OpenAI
    model: str
    num_requests: int
    warmup: int
    concurrency: int


# Thread-safe progress tracking
progress_lock = Lock()
completed_requests = 0


def run_single_request(client: OpenAI, model: str, request_id: int, worker_id: int = 0, 
                       request_timeout: Optional[float] = None) -> Dict:
    """Run a single request and measure performance"""
    start = time.time()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": f"Please analyze this data: sample_{request_id}"}
            ],
            tools=TOOLS,
            tool_choice="required",
            timeout=request_timeout
        )

        end = time.time()
        latency = (end - start) * 1000  # Convert to ms

        has_tool_call = bool(response.choices[0].message.tool_calls)

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        return {
            "request_id": request_id,
            "worker_id": worker_id,
            "latency_ms": latency,
            "success": True,
            "has_tool_call": has_tool_call,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "start_time": start,
            "end_time": end
        }

    except Exception as e:
        end = time.time()
        latency = (end - start) * 1000

        return {
            "request_id": request_id,
            "worker_id": worker_id,
            "latency_ms": latency,
            "success": False,
            "error": str(e),
            "has_tool_call": False,
            "start_time": start,
            "end_time": end
        }

def run_sequential_benchmark(client: OpenAI, model: str, num_requests: int, warmup: int,
                             request_timeout: Optional[float] = None):
    """Run benchmark with sequential requests"""
    all_results = []

    print(f"\nWarming up ({warmup} requests)...")
    for i in range(warmup):
        run_single_request(client, model, i, request_timeout=request_timeout)
        print(f"  Warmup {i+1}/{warmup} complete")

    print(f"\nRunning sequential benchmark ({num_requests} requests)...")
    benchmark_start = time.time()
    
    for i in range(num_requests):
        result = run_single_request(client, model, i, request_timeout=request_timeout)
        all_results.append(result)

        status = "✓" if result["success"] else "✗"
        tool_status = "TOOL" if result.get("has_tool_call") else "TEXT"
        error_info = f" [{result.get('error', '')[:30]}]" if not result["success"] else ""
        print(f"  {status} Request {i+1:3d}: {result['latency_ms']:7.2f}ms [{tool_status}]{error_info}")

    benchmark_end = time.time()
    wall_clock_time = benchmark_end - benchmark_start
    
    return all_results, wall_clock_time


def run_concurrent_benchmark(client: OpenAI, model: str, num_requests: int, warmup: int, 
                             concurrency: int, request_timeout: Optional[float] = None,
                             future_timeout: Optional[float] = None):
    """Run benchmark with concurrent requests using thread pool"""
    global completed_requests
    completed_requests = 0
    all_results = []
    timed_out_futures = 0

    print(f"\nWarming up ({warmup} sequential requests)...")
    for i in range(warmup):
        run_single_request(client, model, i, request_timeout=request_timeout)
        print(f"  Warmup {i+1}/{warmup} complete")

    print(f"\nRunning concurrent benchmark ({num_requests} requests, {concurrency} workers)...")
    benchmark_start = time.time()

    def worker_task(request_id: int, worker_id: int) -> Dict:
        global completed_requests
        result = run_single_request(client, model, request_id, worker_id, request_timeout=request_timeout)
        
        with progress_lock:
            completed_requests += 1
            current = completed_requests
        
        status = "✓" if result["success"] else "✗"
        tool_status = "TOOL" if result.get("has_tool_call") else "TEXT"
        error_info = f" [{result.get('error', '')[:30]}]" if not result["success"] else ""
        print(f"  {status} Request {current:3d}/{num_requests} (W{worker_id}): {result['latency_ms']:7.2f}ms [{tool_status}]{error_info}")
        
        return result

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_request = {}
        for i in range(num_requests):
            worker_id = i % concurrency
            future = executor.submit(worker_task, i, worker_id)
            future_to_request[future] = i

        for future in as_completed(future_to_request, timeout=future_timeout):
            try:
                result = future.result(timeout=1.0)  # Short timeout since task should be done
                all_results.append(result)
            except FuturesTimeoutError:
                request_id = future_to_request[future]
                timed_out_futures += 1
                print(f"  ✗ Request {request_id} future timed out")
                all_results.append({
                    "request_id": request_id,
                    "worker_id": request_id % concurrency,
                    "latency_ms": 0,
                    "success": False,
                    "error": "Future timeout",
                    "has_tool_call": False,
                    "start_time": benchmark_start,
                    "end_time": time.time()
                })
            except Exception as e:
                request_id = future_to_request[future]
                print(f"  ✗ Request {request_id} raised exception: {e}")
                all_results.append({
                    "request_id": request_id,
                    "worker_id": request_id % concurrency,
                    "latency_ms": 0,
                    "success": False,
                    "error": str(e),
                    "has_tool_call": False,
                    "start_time": benchmark_start,
                    "end_time": time.time()
                })

    benchmark_end = time.time()
    wall_clock_time = benchmark_end - benchmark_start

    if timed_out_futures > 0:
        print(f"\n⚠ {timed_out_futures} request(s) timed out")

    # Sort results by request_id for consistent ordering
    all_results.sort(key=lambda x: x["request_id"])
    
    return all_results, wall_clock_time


def run_benchmark(client: OpenAI, model: str, num_requests: int = 20, warmup: int = 3, 
                  concurrency: int = 1, request_timeout: float = 60.0):
    """Run benchmark with optional concurrency"""
    print("="*70)
    print(f"Performance Benchmark: {num_requests} requests (+{warmup} warmup)")
    print(f"Model: {model}")
    print(f"Concurrency: {concurrency} worker(s)")
    print(f"Request Timeout: {request_timeout}s")
    print("="*70)

    # Set future timeout to be a bit longer than request timeout * expected requests per worker
    future_timeout = request_timeout * (num_requests // concurrency + 1) * 2

    if concurrency == 1:
        all_results, wall_clock_time = run_sequential_benchmark(
            client, model, num_requests, warmup, request_timeout=request_timeout
        )
    else:
        all_results, wall_clock_time = run_concurrent_benchmark(
            client, model, num_requests, warmup, concurrency,
            request_timeout=request_timeout, future_timeout=future_timeout
        )

    analyze_results(all_results, wall_clock_time, concurrency)

def analyze_results(results: List[Dict], wall_clock_time: float, concurrency: int):
    """Analyze and display benchmark results"""
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    if not successful:
        print("✗ All requests failed!")
        return

    latencies = [r["latency_ms"] for r in successful]
    tool_calls = [r for r in successful if r.get("has_tool_call")]

    # Overall statistics
    print(f"\n{'Metric':<35} {'Value':>15}")
    print("-"*70)
    print(f"{'Total Requests':<35} {len(results):>15}")
    print(f"{'Successful Requests':<35} {len(successful):>15}")
    print(f"{'Failed Requests':<35} {len(failed):>15}")
    print(f"{'Success Rate':<35} {len(successful)/len(results)*100:>14.1f}%")
    print(f"{'Tool Call Rate':<35} {len(tool_calls)/len(successful)*100:>14.1f}%")
    print(f"{'Concurrency Level':<35} {concurrency:>15}")

    # Latency statistics
    print(f"\n{'Latency Metric':<35} {'Value (ms)':>15}")
    print("-"*70)
    print(f"{'Mean':<35} {mean(latencies):>15.2f}")
    print(f"{'Median':<35} {median(latencies):>15.2f}")
    print(f"{'Min':<35} {min(latencies):>15.2f}")
    print(f"{'Max':<35} {max(latencies):>15.2f}")

    if len(latencies) > 1:
        print(f"{'Std Deviation':<35} {stdev(latencies):>15.2f}")

    # Percentiles
    sorted_latencies = sorted(latencies)
    p50_idx = max(0, len(sorted_latencies)//2 - 1)
    p95_idx = max(0, int(len(sorted_latencies)*0.95) - 1)
    p99_idx = max(0, int(len(sorted_latencies)*0.99) - 1)
    
    p50 = sorted_latencies[p50_idx]
    p95 = sorted_latencies[p95_idx]
    p99 = sorted_latencies[p99_idx]

    print(f"{'P50 (Median)':<35} {p50:>15.2f}")
    print(f"{'P95':<35} {p95:>15.2f}")
    print(f"{'P99':<35} {p99:>15.2f}")

    # Throughput metrics
    sum_latencies = sum(latencies) / 1000  # Total processing time in seconds
    actual_throughput = len(successful) / wall_clock_time if wall_clock_time > 0 else 0

    print(f"\n{'Throughput Metric':<35} {'Value':>15}")
    print("-"*70)
    print(f"{'Wall Clock Time':<35} {wall_clock_time:>14.2f}s")
    print(f"{'Sum of Latencies':<35} {sum_latencies:>14.2f}s")
    print(f"{'Actual Throughput (req/s)':<35} {actual_throughput:>15.2f}")
    
    if concurrency > 1:
        theoretical_sequential = sum_latencies
        speedup = theoretical_sequential / wall_clock_time if wall_clock_time > 0 else 0
        efficiency = (speedup / concurrency) * 100
        
        print(f"{'Speedup vs Sequential':<35} {speedup:>14.2f}x")
        print(f"{'Parallel Efficiency':<35} {efficiency:>14.1f}%")
        print(f"{'Avg Concurrent Requests':<35} {sum_latencies / wall_clock_time:>15.2f}")

    # Token statistics (if available)
    if successful[0].get("total_tokens", 0) > 0:
        total_tokens = sum(r.get("total_tokens", 0) for r in successful)
        avg_tokens = total_tokens / len(successful)

        print(f"\n{'Token Metric':<35} {'Value':>15}")
        print("-"*70)
        print(f"{'Total Tokens':<35} {total_tokens:>15,}")
        print(f"{'Avg Tokens/Request':<35} {avg_tokens:>15.1f}")
        print(f"{'Tokens/Second (wall clock)':<35} {total_tokens/wall_clock_time:>15.1f}")

    # Per-worker statistics (for concurrent runs)
    if concurrency > 1:
        print(f"\n{'Per-Worker Statistics':<35}")
        print("-"*70)
        worker_results = {}
        for r in successful:
            wid = r.get("worker_id", 0)
            if wid not in worker_results:
                worker_results[wid] = []
            worker_results[wid].append(r["latency_ms"])
        
        for wid in sorted(worker_results.keys()):
            worker_latencies = worker_results[wid]
            print(f"  Worker {wid}: {len(worker_latencies)} reqs, "
                  f"avg {mean(worker_latencies):.2f}ms, "
                  f"min {min(worker_latencies):.2f}ms, "
                  f"max {max(worker_latencies):.2f}ms")

    # Error analysis
    if failed:
        print(f"\n{'Error Analysis':<35}")
        print("-"*70)
        error_types = {}
        for r in failed:
            error = r.get("error", "Unknown")
            error_types[error] = error_types.get(error, 0) + 1

        for error, count in error_types.items():
            print(f"  {error}: {count} occurrences")

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM tool calling performance with concurrency support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sequential benchmark (default)
  python experiment_5_benchmark.py -m gpt-oss-120b -n 20

  # Concurrent benchmark with 4 workers
  python experiment_5_benchmark.py -m gpt-oss-120b -n 100 -c 4

  # High concurrency stress test
  python experiment_5_benchmark.py -m gpt-oss-120b -n 200 -c 10 -w 5

  # With custom timeout (useful for slow models)
  python experiment_5_benchmark.py -m gpt-oss-120b -n 20 -c 4 -t 120
        """
    )
    parser.add_argument("-m", "--model", type=str, required=True,
                       help="Model name to use for inference")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1",
                       help="Base URL for the API endpoint (default: http://localhost:8000/v1)")
    parser.add_argument("--api-key", type=str, default="test",
                       help="API key (default: test)")
    parser.add_argument("-n", "--num-requests", type=int, default=20,
                       help="Total number of requests to send (default: 20)")
    parser.add_argument("-c", "--concurrency", type=int, default=1,
                       help="Number of concurrent workers/users (default: 1 = sequential)")
    parser.add_argument("-w", "--warmup", type=int, default=3,
                       help="Number of warmup requests (default: 3)")
    parser.add_argument("-t", "--timeout", type=float, default=60.0,
                       help="Timeout per request in seconds (default: 60)")

    args = parser.parse_args()

    # Create client with timeout configured
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=httpx.Timeout(args.timeout, connect=10.0)
    )

    run_benchmark(
        client,
        args.model,
        num_requests=args.num_requests,
        warmup=args.warmup,
        concurrency=args.concurrency,
        request_timeout=args.timeout
    )

if __name__ == "__main__":
    main()
