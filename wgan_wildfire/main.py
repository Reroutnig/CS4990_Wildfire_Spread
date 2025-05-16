import torch
import os
import argparse
import time
from functools import lru_cache
from torch.utils.data import DataLoader

from models import Generator, Critic
from dataset import SequentialWildfireDataset
from training import train_wgan_ndvi_model

# ------------------------------
# Main Function
# ------------------------------
if __name__ == "__main__":
    import argparse
    import time
    from functools import lru_cache
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train wildfire prediction model with improved loss functions and spatial constraints')
    parser.add_argument('--ndvi_dir', type=str, default='../data/ndvi_west_coast_tensors', help='Directory with NDVI tensors')
    parser.add_argument('--burn_dir', type=str, default='../data/burn_wc_tensors', help='Directory with burn tensors')
    parser.add_argument('--save_path', type=str, default='new_model/wgan_improved_run2', help='Directory to save models')
    parser.add_argument('--history_days', type=int, default=3, help='Number of previous days to use for prediction')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--lambda_L1', type=float, default=100, help='Weight for L1 loss')
    parser.add_argument('--lambda_fire', type=float, default=50.0, help='Weight for fire loss')
    parser.add_argument('--fire_weight', type=float, default=250.0, help='Weight for fire pixels')
    parser.add_argument('--lambda_confidence', type=float, default=10.0, help='Weight for confidence loss')
    parser.add_argument('--alpha', type=float, default=0.25, help='Focal loss alpha parameter (class weight)')
    parser.add_argument('--gamma', type=float, default=2.0, help='Focal loss gamma parameter (focusing parameter)')
    parser.add_argument('--n_critic', type=int, default=5, help='Number of critic updates per generator update')
    parser.add_argument('--lambda_gp', type=float, default=10, help='Weight for gradient penalty')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--lr_decay_epochs', type=int, default=30, help='Decay learning rate every N epochs')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='Learning rate decay factor')
    parser.add_argument('--vis_dir', type=str, default='visualizations', help='Directory to save visualizations')
    parser.add_argument('--post_process_threshold', type=float, default=0.7, help='Threshold for post-processing')
    parser.add_argument('--post_process_min_area', type=int, default=0, help='Minimum area for post-processing')
    parser.add_argument('--post_process_max_gap', type=int, default=0, help='Maximum gap for post-processing')
    parser.add_argument('--cache_size', type=int, default=256, help='Number of tensors to cache in memory')
    parser.add_argument('--save_every', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--vis_every', type=int, default=1000, help='Create visualization every N batches')
    parser.add_argument('--profile', action='store_true', help='Enable performance profiling')
    parser.add_argument('--optimize_memory', action='store_true', help='Enable memory optimizations')
    parser.add_argument('--workers', type=int, default=4, help='Number of dataloader workers (0-8)')
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'fp16'], help='Training precision')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Ratio of data for training vs validation (default: 0.8)')
    parser.add_argument('--balance_fire', action='store_true', help='Balance fire content between train and validation sets')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--validate_every', type=int, default=5, help='Run validation every N epochs')
    parser.add_argument('--loss_plot_every', type=int, default=20, help='Generate loss plots every N epochs')
    parser.add_argument('--curriculum_learning', action='store_true', help='Use curriculum learning to gradually introduce NDVI')
    parser.add_argument('--curriculum_start', type=int, default=20, help='Epoch to start introducing NDVI')
    parser.add_argument('--curriculum_end', type=int, default=60, help='Epoch where NDVI reaches full influence')
    parser.add_argument('--curriculum_max_ndvi', type=float, default=1.0, help='Maximum NDVI influence (1.0 = full influence)')
    parser.add_argument('--curriculum_type', type=str, default='linear', choices=['linear', 'cosine'], 
                    help='Type of curriculum schedule')
    
    args = parser.parse_args()
    
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA")
    else:
        device = torch.device("cpu")
        print(f"Using CPU")
    
    if args.profile:
        # Enable performance profiling
        print("Performance profiling enabled")
        start_time = time.time()
        
    # Add LRU cache for tensor loading
    @lru_cache(maxsize=args.cache_size)
    def cached_tensor_load(path):
        return torch.load(path)
    
    # Modify SequentialWildfireDataset to use the cache
    original_getitem = SequentialWildfireDataset.__getitem__
    def optimized_getitem(self, idx):
        ndvi_path, burn_history_paths, curr_burn_path = self.valid_sequences[idx]
        
        try:
            if args.optimize_memory:
                # Load tensors using cache
                ndvi_tensor = cached_tensor_load(ndvi_path)
                burn_history_tensors = [cached_tensor_load(path) for path in burn_history_paths]
                curr_burn_tensor = cached_tensor_load(curr_burn_path)
            else:
                # Use original loading
                ndvi_tensor = torch.load(ndvi_path)
                burn_history_tensors = [torch.load(path) for path in burn_history_paths]
                curr_burn_tensor = torch.load(curr_burn_path)
            
            # Ensure 3D tensors
            if ndvi_tensor.dim() == 2:
                ndvi_tensor = ndvi_tensor.unsqueeze(0)
            burn_history_tensors = [b.unsqueeze(0) if b.dim() == 2 else b for b in burn_history_tensors]
            if curr_burn_tensor.dim() == 2:
                curr_burn_tensor = curr_burn_tensor.unsqueeze(0)
            
            # Normalize NDVI to [-1, 1] range
            ndvi_tensor = torch.clamp(ndvi_tensor, -1.0, 1.0)
            
            # Convert burn tensors to binary and then to [-1, 1] range for WGAN
            burn_history_tensors = [(b > 0).float() * 2 - 1 for b in burn_history_tensors]
            curr_burn_tensor = (curr_burn_tensor > 0).float() * 2 - 1
            
            # Stack burn history along channel dimension
            burn_history_tensor = torch.cat(burn_history_tensors, dim=0)
            
            return ndvi_tensor, burn_history_tensor, curr_burn_tensor
            
        except Exception as e:
            print(f"Error loading tensors at index {idx}: {e}")
            # Return placeholder tensors in case of error
            return (torch.zeros(1, 64, 64), torch.zeros(self.history_days, 64, 64), torch.zeros(1, 64, 64))
    
    # Apply optimized method if memory optimization is enabled
    if args.optimize_memory:
        print("Memory optimizations enabled - using tensor caching")
        SequentialWildfireDataset.__getitem__ = optimized_getitem
    
    # Create dataset and dataloader
    print(f"Creating Sequential Wildfire Dataset with NDVI and {args.history_days}-day history...")

    # Training dataset
    train_dataset = SequentialWildfireDataset(
        args.ndvi_dir, 
        args.burn_dir, 
        history_days=args.history_days,
        train=True,
        split_ratio=args.split_ratio,
        balance_fire=args.balance_fire,
        seed=args.seed
    )
    # Validation dataset
    val_dataset = SequentialWildfireDataset(
        args.ndvi_dir, 
        args.burn_dir, 
        history_days=args.history_days,
        train=False,
        split_ratio=args.split_ratio,
        balance_fire=args.balance_fire,
        seed=args.seed
    )

    # Start with 0 workers to avoid potential issues
    safe_workers = 0 if device.type == 'mps' else args.workers
    
    # Training dataloader
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=safe_workers,
        pin_memory=True if device.type != "cpu" else False,
        persistent_workers=True if safe_workers > 0 else False,
        prefetch_factor=2 if safe_workers > 0 else None
    )

    # Validation dataloader
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=safe_workers,
        pin_memory=True if device.type != "cpu" else False,
        persistent_workers=True if safe_workers > 0 else False,
        prefetch_factor=2 if safe_workers > 0 else None
)

    print(f"Created dataloaders with {len(train_dataset)} training samples and {len(val_dataset)} validation samples")  
    
    # Initialize models
    generator = Generator(in_channels=1 + args.history_days + 1, out_channels=1).to(device)
    critic = Critic(ndvi_channels=1, history_channels=args.history_days, output_channels=1).to(device)
    
    # Set post-processing parameters for evaluation
    generator.set_post_processing(
        enabled=False,
        threshold=args.post_process_threshold,
        min_area=args.post_process_min_area,
        max_gap=args.post_process_max_gap
    )
    
    # Load checkpoint if provided
    checkpoint = None
    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"Resuming from epoch {checkpoint['epoch']}")
    
    # Print model summary
    print(f"Generator Parameters: {sum(p.numel() for p in generator.parameters())}")
    print(f"Critic Parameters: {sum(p.numel() for p in critic.parameters())}")
    
    # Configure training parameters
    train_params = {
        'lr': args.lr,
        'lambda_L1': args.lambda_L1,
        'lambda_fire': args.lambda_fire,
        'fire_weight': args.fire_weight,
        'alpha': args.alpha,
        'gamma': args.gamma,
        'lambda_confidence': args.lambda_confidence,
        'save_path': args.save_path,
        'checkpoint': checkpoint,
        'n_critic': args.n_critic,
        'lambda_gp': args.lambda_gp,
        'lr_decay_epochs': args.lr_decay_epochs,
        'lr_decay_rate': args.lr_decay_rate,
        'visualization_dir': args.vis_dir,
        'save_every': args.save_every,
        'vis_every': args.vis_every,
        'precision': args.precision
    }

    if args.curriculum_learning:
        print("Enabling curriculum learning to gradually introduce NDVI influence")
        train_params.update({
            'curriculum_start': args.curriculum_start,
            'curriculum_end': args.curriculum_end,
            'curriculum_max_ndvi': args.curriculum_max_ndvi,
            'curriculum_type': args.curriculum_type
        })
        
        # Rename the save path to indicate curriculum learning
        train_params['save_path'] = os.path.join(args.save_path, f"curriculum_{args.curriculum_type}")
        os.makedirs(train_params['save_path'], exist_ok=True)
        
        # Print curriculum learning configuration
        print(f"  - NDVI starts at: Epoch {args.curriculum_start}")
        print(f"  - NDVI fully introduced by: Epoch {args.curriculum_end}")
        print(f"  - Maximum NDVI influence: {args.curriculum_max_ndvi:.2f}")
        print(f"  - Schedule type: {args.curriculum_type}")
    else:
        # Default behavior - full NDVI influence from start
        train_params.update({
            'curriculum_start': 0,
            'curriculum_end': 0,
            'curriculum_max_ndvi': 1.0,
            'curriculum_type': 'linear'
        })
    
    print(f"Starting training with precision: {args.precision}")
    train_start = time.time()

    # Train the model with improved loss functions
    generator, critic, history = train_wgan_ndvi_model(
        generator, critic, 
        train_dataloader, val_dataloader,
        args.epochs, device,
        validate_every=args.validate_every,
        loss_plot_every=args.loss_plot_every,
        **train_params
    )
    
    train_time = time.time() - train_start
    
    if args.profile:
        total_time = time.time() - start_time
        print(f"\nPerformance Summary:")
        print(f"  Total runtime: {total_time:.2f} seconds")
        print(f"  Training time: {train_time:.2f} seconds")
        print(f"  Setup time: {total_time - train_time:.2f} seconds")
        print(f"  Avg. time per epoch: {train_time / args.epochs:.2f} seconds")
    
    print("Training completed!")