import os
import time
import concurrent.futures
import requests
from requests.auth import HTTPBasicAuth
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fl-driver")

# Configuration from environment variables
OPENFAAS_GATEWAY_URL = os.getenv(
    "OPENFAAS_GATEWAY_URL",
    "http://gateway.openfaas.svc.cluster.local:8080"
)
OPENFAAS_USERNAME = os.getenv("OPENFAAS_USERNAME", "admin")
OPENFAAS_PASSWORD = os.getenv("OPENFAAS_PASSWORD", "")

NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "3"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))
TIMEOUT = (10, 120)  # (connect, read)
HEADERS_BIN = {'Content-Type': 'application/octet-stream'}
HEADERS_JSON = {'Content-Type': 'application/json'}
AUTH = HTTPBasicAuth(OPENFAAS_USERNAME, OPENFAAS_PASSWORD)

def get_global_model():
    """Get global model from aggregator via OpenFaaS Gateway"""
    logger.info("--> Getting global model from aggregator...")
    url = f"{OPENFAAS_GATEWAY_URL}/function/fl-aggregator"
    params = {"action": "get_global_model"}
    
    resp = requests.get(url, params=params, auth=AUTH, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.content

def invoke_client(model_bytes, idx):
    """Invoke client training via OpenFaaS Gateway"""
    logger.info(f"    -> Invoking client {idx} for training...")
    url = f"{OPENFAAS_GATEWAY_URL}/function/fl-client"
    
    resp = requests.post(
        url,
        data=model_bytes,
        headers=HEADERS_BIN,
        auth=AUTH,
        timeout=TIMEOUT
    )
    resp.raise_for_status()
    return resp.content

def submit_update(update_bytes):
    """Submit client update to aggregator via OpenFaaS Gateway"""
    logger.info("    <- Submitting client update to aggregator...")
    url = f"{OPENFAAS_GATEWAY_URL}/function/fl-aggregator"
    params = {"action": "submit_update"}
    
    resp = requests.post(
        url,
        params=params,
        data=update_bytes,
        headers=HEADERS_BIN,
        auth=AUTH,
        timeout=TIMEOUT
    )
    resp.raise_for_status()
    
    try:
        return resp.json()
    except:
        return {"status": "received"}

def main():
    """Main federated learning execution"""
    logger.info("=== Starting Federated Learning Driver ===")
    
    successful_rounds = 0
    
    for rnd in range(1, NUM_ROUNDS + 1):
        logger.info(f"\n--- FL Round {rnd}/{NUM_ROUNDS} ---")
        
        try:
            # Get global model
            global_model = get_global_model()
            logger.info(f"Global model received: {len(global_model)} bytes")
        except Exception as e:
            logger.error(f"Failed to fetch global model: {e}")
            continue

        # Parallel client training
        updates = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CLIENTS) as pool:
            futures = {
                pool.submit(invoke_client, global_model, i): i
                for i in range(NUM_CLIENTS)
            }
            
            for fut in concurrent.futures.as_completed(futures):
                i = futures[fut]
                try:
                    update = fut.result()
                    updates.append(update)
                    logger.info(f"    <- Received update from client {i}: {len(update)} bytes")
                except Exception as e:
                    logger.error(f"Client {i} training failed: {e}")

        if len(updates) == 0:
            logger.warning(f"No updates received, skipping round {rnd}")
            continue
        
        logger.info(f"Received {len(updates)}/{NUM_CLIENTS} updates")

        # Submit updates to aggregator
        aggregation_success = False
        for j, upd in enumerate(updates):
            try:
                result = submit_update(upd)
                logger.info(f"Update {j+1} submitted: {result}")
                aggregation_success = True
            except Exception as e:
                logger.error(f"Failed to submit update {j+1}: {e}")

        if aggregation_success:
            successful_rounds += 1
            logger.info(f"--- Round {rnd} completed successfully ---")
        else:
            logger.warning(f"--- Round {rnd} failed ---")
        
        if rnd < NUM_ROUNDS:
            time.sleep(5)

    logger.info(f"=== Federated Learning Finished: {successful_rounds}/{NUM_ROUNDS} successful rounds ===")

if __name__ == "__main__":
    main()
