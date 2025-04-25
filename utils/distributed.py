import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Define MIG UUIDs - these should be configured based on your system
MIG_UUIDS = [
    "MIG-848a657a-efa3-57e0-9cb0-f78995e946f1",
    "MIG-b43985b9-b966-5ee1-b596-5e4440ba80a5",
    "MIG-adfbd773-a827-5d98-a461-f5483bb634af",
    "MIG-37e22da3-01d1-5528-ad44-005d3158089a"
]

def print_gpu_info():
    """Print information about available GPUs"""
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

def setup_mig_environment(rank, world_size):
    """
    Set CUDA environment variables for MIG configuration
    
    Args:
        rank: Process rank
        world_size: Total number of processes
    """
    if rank < len(MIG_UUIDS):
        # Set CUDA visible devices to the MIG UUID
        os.environ["CUDA_VISIBLE_DEVICES"] = MIG_UUIDS[rank]
        
        # Set CUDA device order to PCI bus ID
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        
        # Set unique socket interface for NCCL
        os.environ["NCCL_SOCKET_IFNAME"] = f"lo,eth{rank}"
        
        # Set NCCL debug to WARN to get more information during initial setup
        os.environ["NCCL_DEBUG"] = "WARN"
        
        # Set a unique node rank for each MIG instance (important!)
        os.environ["NCCL_UNIQUE_ID_RANK"] = str(rank)
        
        print(f"Process {rank} using MIG: {MIG_UUIDS[rank]}")
    else:
        print(f"Warning: Process {rank} doesn't have a corresponding MIG UUID")

def setup(rank, world_size):
    """
    Initialize the distributed environment
    
    Args:
        rank: Process rank
        world_size: Total number of processes
    """
    # Use a unique port to avoid conflicts
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group with TCP first to validate connectivity
    # Once basic connectivity is established, you can switch back to NCCL
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(f"Process {rank}/{world_size} distributed setup complete with gloo backend")
    
    # After initial setup succeeds, you can try switching to NCCL
    # dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # print(f"Process {rank}/{world_size} distributed setup complete with NCCL backend")

def cleanup():
    """Clean up the distributed environment"""
    dist.destroy_process_group()

def get_device(rank):
    """
    Get the appropriate device for a process
    
    Args:
        rank: Process rank
        
    Returns:
        device: Torch device
    """
    # After setting CUDA_VISIBLE_DEVICES, this process should see only one GPU (index 0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return device

def wrap_ddp_model(model, device_id=0):
    """
    Wrap a model with DistributedDataParallel
    
    Args:
        model: PyTorch model
        device_id: Device ID
        
    Returns:
        ddp_model: DDP wrapped model
    """
    return DDP(model, device_ids=[device_id])

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