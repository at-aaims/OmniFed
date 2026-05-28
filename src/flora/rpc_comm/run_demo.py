#!/usr/bin/env python3
"""
Demo script to run the federated learning system with multiple clients
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. parameter_server.proto
"""

import subprocess
import time
import sys
import os
import signal
import multiprocessing as mp


def run_server():
    """Run the parameter server"""
    print("Starting parameter server...")
    try:
        # process = subprocess.Popen([sys.executable, "parameter_server_memleak.py"])
        # process = subprocess.Popen([sys.executable, "parameter_server.py"])
        process = subprocess.Popen([sys.executable, "tempfix_paramserver.py"])
        return process
    except Exception as e:
        print(f"Failed to start server: {e}")
        return None


def run_client(client_id, delay=0):
    """Run a client with specified ID"""
    if delay > 0:
        time.sleep(delay)

    print(f"Starting client {client_id}...")
    try:
        process = subprocess.Popen([sys.executable, "client.py", client_id])
        process.wait()  # Wait for client to finish
        print(f"Client {client_id} completed")
    except Exception as e:
        print(f"Failed to start client {client_id}: {e}")


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\nShutting down demo...")
    sys.exit(0)


def main():
    # Check if required files exist
    required_files = [
        # "parameter_server.py",
        "tempfix_paramserver.py",
        "client.py",
        "parameter_server_pb2.py",
        "parameter_server_pb2_grpc.py",
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("Error: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease run: python setup.py")
        sys.exit(1)

    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)

    print("🚀 Starting Federated Learning Demo")
    print("=" * 50)

    # Start parameter server
    server_process = run_server()
    if server_process is None:
        print("Failed to start server")
        sys.exit(1)

    # Give server time to start
    time.sleep(3)

    # Start clients in separate processes
    num_clients = 3
    client_processes = []

    for i in range(num_clients):
        client_id = f"client_{i + 1}"
        # Start each client with a small delay
        p = mp.Process(target=run_client, args=(client_id, i * 2))
        p.start()
        client_processes.append(p)

    try:
        # Wait for all clients to complete
        for p in client_processes:
            p.join()

        print("\n✅ All clients completed training")

    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")

        # Terminate client processes
        for p in client_processes:
            if p.is_alive():
                p.terminate()
                p.join()

    finally:
        # Terminate server
        if server_process and server_process.poll() is None:
            print("Shutting down parameter server...")
            server_process.terminate()
            server_process.wait()

        print("Demo completed")


if __name__ == "__main__":
    main()
