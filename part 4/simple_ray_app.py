import ray
import time
import random

@ray.remote
def cpu_intensive_task(item_id: int) -> int:
    """A function that simulates a simple CPU-intensive task."""
    print(f"Starting processing for item {item_id} on a worker...")
    time.sleep(random.uniform(1, 3))
    result = item_id * item_id
    print(f"Finished processing for item {item_id}. Result: {result}")
    return result

if __name__ == "__main__":
    print("Connecting to the Ray cluster...")
    # Make sure the address and port match your port-forwarding command
    ray.init(address='ray://127.0.0.1:6189')
    print("Successfully connected.")

    start_time = time.time()

    # Execute the function in parallel for 5 items
    results_refs = [cpu_intensive_task.remote(i) for i in range(5)]
    print("All 5 tasks have been submitted to the cluster.")

    # Wait for the results
    results = ray.get(results_refs)

    end_time = time.time()

    print("\n--- Final Results ---")
    print(results)
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    ray.shutdown()