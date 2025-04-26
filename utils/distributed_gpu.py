import os
import torch
import torch.distributed as dist
import socket
import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def find_free_port():
    """Find a free port on localhost"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # Bind to a free port provided by the OS
        return s.getsockname()[1]

def print_gpu_info():
    """Print information about available GPUs"""
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
    try:
        if torch.cuda.is_available():
            print(f"Available GPUs: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA is not available. Check your environment variables and GPU access.")
    except Exception as e:
        print(f"Error accessing CUDA: {e}")

def setup(rank, world_size):
    """
    Initialize the distributed environment with dynamic port selection
    
    Args:
        rank: Process rank
        world_size: Total number of processes
    """
    # Only rank 0 needs to find a port, then broadcast to others
    if rank == 0:
        port = find_free_port()
        # Store port in a file for other ranks to read
        with open('.dist_port', 'w') as f:
            f.write(str(port))
        print(f"Rank 0 selected port: {port}")
    else:
        # Give rank 0 time to write the port
        import time
        time.sleep(1)
        # Read port from file
        try:
            with open('.dist_port', 'r') as f:
                port = int(f.read().strip())
        except:
            # If file read fails, use a default port
            port = 12367
        print(f"Rank {rank} using port: {port}")
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    
    # Set timeout for initialization
    timeout = 300  # 5 minutes
    print(f"Initializing process group with gloo backend, rank={rank}, world_size={world_size}")
    
    # Initialize the process group
    dist.init_process_group(
        "gloo", 
        rank=rank, 
        world_size=world_size,
        timeout=datetime.timedelta(seconds=timeout)
    )
    print(f"Process {rank}/{world_size} distributed setup complete with gloo backend on port {port}")

def cleanup():
    """Clean up distributed environment"""
    # Remove the port file when done
    if os.path.exists('.dist_port'):
        try:
            os.remove('.dist_port')
        except:
            pass
    
    dist.destroy_process_group()

def get_device(rank):
    """
    Get the appropriate device for a process
    
    Args:
        rank: Process rank
        
    Returns:
        device: Torch device
    """
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        print(f"WARNING: Process {rank} using CPU instead of GPU!")
    
    return device

def wrap_ddp_model(model, device_id):
    """
    Wrap a model with DistributedDataParallel
    
    Args:
        model: PyTorch model
        device_id: Device ID
        
    Returns:
        ddp_model: DDP wrapped model
    """
    if torch.cuda.is_available():
        return DDP(model, device_ids=[device_id])
    else:
        # Fall back to CPU if GPU is not available
        return DDP(model)

def create_distributed_sampler(dataset, rank, world_size):
    """
    Create a distributed sampler for a dataset
    
    Args:
        dataset: PyTorch dataset
        rank: Process rank
        world_size: Total number of processes
        
    Returns:
        sampler: Distributed sampler
    """
    return DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank
    )