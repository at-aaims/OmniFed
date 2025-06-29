# Federated Learning Parameter Server

A synchronous parameter server implementation using gRPC and PyTorch for federated learning. This system allows multiple clients to train a shared model collaboratively by averaging their parameter updates.

## Architecture

The system consists of:

- **Parameter Server**: Collects model updates from clients, averages them, and distributes the averaged model back to clients
- **Clients**: Train local models and communicate with the parameter server
- **gRPC Communication**: Efficient, type-safe communication using Protocol Buffers

## Components

### 1. Protocol Buffer Definition (`parameter_server.proto`)
Defines the communication interface between server and clients:
- `SendUpdate`: Clients send their model updates
- `GetAveragedModel`: Clients request the averaged model
- `RegisterClient`: Client registration

### 2. Parameter Server (`parameter_server.py`)
- Maintains a simple PyTorch neural network model
- Collects updates from all registered clients
- Computes averaged parameters when all updates are received
- Serves averaged model to clients

### 3. Client (`client.py`)
- Implements local training with dummy data
- Sends parameter updates to server
- Retrieves and applies averaged model parameters
- Supports federated training rounds

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate gRPC code**:
   ```bash
   python setup.py
   ```

## Usage

### Manual Usage

1. **Start the parameter server**:
   ```bash
   python parameter_server_memleak.py
   ```

2. **Start clients** (in separate terminals):
   ```bash
   python scalable_client.py client1
   python scalable_client.py client2
   python scalable_client.py client3
   ```

### Automated Demo

Run the complete demo with all components:
```bash
python run_scalable_demo.py
```

## How It Works

1. **Initialization**:
   - Server starts and waits for client registrations
   - Clients register with the server and initialize their local models

2. **Training Round**:
   - Each client trains its local model with local data
   - Clients send their updated parameters to the server
   - Server waits for all clients to send updates
   - Server computes averaged parameters across all clients
   - Server sends averaged model back to all clients

3. **Synchronous Communication**:
   - Clients wait for the averaged model before proceeding to the next round
   - Ensures all clients stay synchronized

## Model Architecture

The example uses a simple 2-layer neural network:
- Input layer: 784 features (simulating flattened 28x28 images)
- Hidden layer: 128 neurons with ReLU activation
- Output layer: 10 classes (simulating classification task)

## Key Features

- **Synchronous Parameter Averaging**: Ensures all clients participate in each round
- **Thread-Safe Server**: Handles concurrent client requests safely
- **Robust Error Handling**: Includes retry logic and graceful error handling
- **Flexible Client Management**: Supports dynamic number of clients
- **Protocol Buffer Serialization**: Efficient parameter transmission

## Configuration

You can modify various parameters:

- **Number of clients**: Change `num_clients` in `parameter_server.py`
- **Model architecture**: Modify the `SimpleModel` class
- **Training parameters**: Adjust learning rate, batch size, etc. in `client.py`
- **Communication settings**: Change server address/port in client initialization

## Extensions

This implementation can be extended to support:

- **Real datasets**: Replace dummy data generation with actual datasets
- **Different models**: Swap out the simple neural network for more complex architectures  
- **Asynchronous updates**: Modify server to handle updates as they arrive
- **Client sampling**: Randomly select subset of clients per round
- **Secure aggregation**: Add encryption for parameter transmission
- **Fault tolerance**: Handle client failures and dropouts

## Files

- `parameter_server.proto`: Protocol buffer definitions
- `parameter_server.py`: Server implementation
- `client.py`: Client implementation  
- `setup.py`: Setup script to generate gRPC code
- `run_demo.py`: Automated demo script
- `requirements.txt`: Python dependencies