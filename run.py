import subprocess
import time
import requests

# List of microservices with their check URLs
microservices = [
    {"name": "backend", "url": "http://localhost:8000/check"},
    {"name": "intent", "url": "http://localhost:8001/check"},
    {"name": "rag", "url": "http://localhost:8002/check"},
    {"name": "function_doctorname", "url": "http://localhost:8003/check"},
    {"name": "function_doctordisease", "url": "http://localhost:8004/check"},
    {"name": "function_doctorspecialization", "url": "http://localhost:8005/check"},
    {"name": "function_generalquery", "url": "http://localhost:8006/check"}
]

# Function to check if a microservice is up
def is_microservice_up(url):
    try:
        response = requests.get(url)
        return response.status_code == 200
    except requests.RequestException:
        return False

# Function to start microservices
def start_microservices():
    processes = []
    for microservice in microservices:
        print(f"Starting {microservice['name']}...")
        process = subprocess.Popen(["uvicorn", microservice["name"] + ":app", "--host", "0.0.0.0", "--port", str(microservice["url"].split(":")[-1].split("/")[0])])
        processes.append(process)
    return processes

# Function to wait until all microservices are up
def wait_for_microservices():
    all_up = False
    while not all_up:
        print("Checking if all microservices are up...")
        all_up = all(is_microservice_up(microservice["url"]) for microservice in microservices)
        if not all_up:
            time.sleep(5)
    print("All microservices are up!")

# Function to stop all microservices
def stop_microservices(processes):
    print("Stopping all microservices...")
    for process in processes:
        process.terminate()  # Gracefully terminate the process
        try:
            process.wait(timeout=5)  # Wait for the process to terminate
        except subprocess.TimeoutExpired:
            print(f"Process {process.pid} did not terminate in time, killing it.")
            process.kill()  # Force kill if it didn't terminate within the timeout

# Start microservices and wait until they are fully up
if __name__ == "__main__":
    processes = start_microservices()

    try:
        # Wait for the services to be fully up
        wait_for_microservices()

        # After all microservices are up, start the Streamlit frontend
        print("Starting Streamlit frontend...")
        subprocess.run(["streamlit", "run", "frontend.py"])

    except KeyboardInterrupt:
        print("\nManual stop triggered. Stopping microservices...")

    # Stop all microservices
    stop_microservices(processes)
