#!/usr/bin/env python3
"""
Test the concurrent data collection system
"""

import asyncio
import time

from data_collection.batch_data_collection import BatchDataCollector


async def test_concurrent_vs_sequential():
    """
    Test concurrent vs sequential execution to measure performance improvement
    """
    print("🧪 Testing Concurrent vs Sequential Data Collection")
    print("=" * 60)

    collector = BatchDataCollector("test_output", max_concurrent=3)

    # Test parameters
    num_sessions = 3
    turns_per_session = 2
    delay = 1  # Use integer for delay

    print("📊 Test Configuration:")
    print(f"   Sessions: {num_sessions}")
    print(f"   Turns per session: {turns_per_session}")
    print(f"   Base delay: {delay}s")
    print()

    # Test Sequential
    print("🐌 Testing Sequential Execution...")
    start_time = time.time()

    try:
        _ = await collector.run_batch_collection(
            num_sessions=num_sessions,
            turns_per_session=turns_per_session,
            delay_between_sessions=delay,
            concurrent=False,
        )
        sequential_time = time.time() - start_time
        print(f"⏱️  Sequential time: {sequential_time:.1f}s")

    except Exception as e:
        print(f"❌ Sequential test failed: {e}")
        sequential_time = None

    print("\n" + "=" * 40 + "\n")

    # Test Concurrent
    print("🚀 Testing Concurrent Execution...")
    start_time = time.time()

    try:
        _ = await collector.run_batch_collection(
            num_sessions=num_sessions,
            turns_per_session=turns_per_session,
            delay_between_sessions=delay,
            concurrent=True,
        )
        concurrent_time = time.time() - start_time
        print(f"⏱️  Concurrent time: {concurrent_time:.1f}s")

    except Exception as e:
        print(f"❌ Concurrent test failed: {e}")
        concurrent_time = None

    # Compare results
    print("\n" + "=" * 60)
    print("📊 Performance Comparison:")

    if sequential_time and concurrent_time:
        speedup = sequential_time / concurrent_time
        print(f"   Sequential: {sequential_time:.1f}s")
        print(f"   Concurrent: {concurrent_time:.1f}s")
        print(f"   Speedup: {speedup:.1f}x faster")

        if speedup > 1.5:
            print("✅ Significant performance improvement achieved!")
        else:
            print("⚠️  Limited speedup - check server capacity or network")
    else:
        print("❌ Unable to compare due to test failures")


if __name__ == "__main__":
    asyncio.run(test_concurrent_vs_sequential())
