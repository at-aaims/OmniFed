#!/usr/bin/env python3
"""
Setup script to generate gRPC code from protobuf definitions
"""

import subprocess
import sys
import os


def generate_grpc_code():
    """Generate Python gRPC code from .proto file"""
    proto_file = "/Users/ssq/Documents/github/ornl_projects/FLORA_beta/src/flora/rpc_comm/parameter_server.proto"

    if not os.path.exists(proto_file):
        print(f"Error: {proto_file} not found!")
        return False

    try:
        # Generate gRPC code
        cmd = [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            "--proto_path=.",
            "--python_out=.",
            "--grpc_python_out=.",
            proto_file,
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        print("Successfully generated gRPC code:")
        print("  - parameter_server_pb2.py")
        print("  - parameter_server_pb2_grpc.py")

        return True

    except subprocess.CalledProcessError as e:
        print(f"Error generating gRPC code: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: grpcio-tools not installed. Please run: pip install grpcio-tools")
        return False


def install_requirements():
    """Install required packages"""
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
        )
        print("Successfully installed requirements")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False


def main():
    print("Setting up Federated Learning Parameter Server...")

    # Install requirements
    # print("\n1. Installing requirements...")
    # if not install_requirements():
    #     sys.exit(1)

    # Generate gRPC code
    print("\n2. Generating gRPC code...")
    if not generate_grpc_code():
        sys.exit(1)

    print("\n✅ Setup completed successfully!")
    print("\nTo run the system:")
    print("1. Start the server: python parameter_server_memleak.py")
    print("2. Start clients in separate terminals:")
    print("   python scalable_client.py client1")
    print("   python scalable_client.py client2")
    print("   python scalable_client.py client3")


if __name__ == "__main__":
    main()
