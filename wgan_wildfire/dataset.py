import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class SequentialWildfireDataset(Dataset):
    def __init__(self, ndvi_dir, burn_dir, history_days=3, train=True, split_ratio=0.8, balance_fire=True, seed=42):
        """
        Dataset for wildfire prediction with sequential burn history and NDVI data
        
        Args:
            ndvi_dir (str): Directory containing NDVI tensor files
            burn_dir (str): Directory containing burn tensor files
            history_days (int): Number of previous days to use for prediction
            train (bool): Whether this is a training dataset (True) or validation dataset (False)
            split_ratio (float): Ratio of data to use for training vs. validation (e.g., 0.8 = 80% training)
            balance_fire (bool): Whether to ensure balanced fire distribution between train/val sets
            seed (int): Random seed for reproducibility
        """
        self.ndvi_dir = ndvi_dir
        self.burn_dir = burn_dir
        self.history_days = history_days
        self.train = train
        self.split_ratio = split_ratio
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Get all burn files and sort them by day number
        burn_files = [f for f in os.listdir(burn_dir) if f.endswith('.pt')]
        
        # Extract day number from each filename and create (filename, day) pairs
        burn_day_pairs = []
        for filename in burn_files:
            try:
                # Extract day from filename
                day = int(filename.split('_')[2].split('.')[0])
                burn_day_pairs.append((filename, day))
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not parse day from filename {filename}: {e}")
                continue
        
        # Sort the pairs by day number
        burn_day_pairs.sort(key=lambda x: x[1])  # Sort by day number (second element in tuple)
        
        # Extract just the filenames in the correct order
        self.burn_files = [pair[0] for pair in burn_day_pairs]
        
        print(f"Found {len(self.burn_files)} burn files, sorted by day number")
        
        # Build valid sequences
        all_valid_sequences = []
        
        # Start from day that has enough history days before it
        for i in range(self.history_days, len(self.burn_files)):
            curr_burn_file = self.burn_files[i]
            curr_day = int(curr_burn_file.split('_')[2].split('.')[0])
            
            # Get history files - the days immediately preceding the current day
            burn_history = []
            history_idx = i - 1  # Start with the day just before current
            
            # Check if we have the right consecutive days
            expected_days = [curr_day - j - 1 for j in range(self.history_days)]
            valid_history = True
            
            for expected_day in expected_days:
                # Check if we've reached the beginning of the file list
                if history_idx < 0:
                    valid_history = False
                    break
                
                history_file = self.burn_files[history_idx]
                history_day = int(history_file.split('_')[2].split('.')[0])
                
                # If this history day matches what we expect, add it
                if history_day == expected_day:
                    burn_history.append(history_file)
                    history_idx -= 1
                else:
                    # We have a gap in the data - missing days
                    valid_history = False
                    break
            
            # Only create sequence if we have valid consecutive history
            if not valid_history or len(burn_history) != self.history_days:
                continue
                
            # Reverse burn history to have oldest first
            burn_history.reverse()
            
            # Find appropriate NDVI file
            ndvi_day = self._find_nearest_ndvi_day(curr_day)
            ndvi_file = f"NDVI_2024_{ndvi_day}.pt" # Currently only training with 2024 data
            
            ndvi_path = os.path.join(self.ndvi_dir, ndvi_file)
            burn_history_paths = [os.path.join(self.burn_dir, f) for f in burn_history]
            curr_burn_path = os.path.join(self.burn_dir, curr_burn_file)
            
            # Check if all files exist
            if os.path.exists(ndvi_path) and all(os.path.exists(p) for p in burn_history_paths) and os.path.exists(curr_burn_path):
                all_valid_sequences.append((ndvi_path, burn_history_paths, curr_burn_path, curr_day))
        
        print(f"Found {len(all_valid_sequences)} valid sequences")
        
        # Calculate fire content for each sequence
        sequence_fire_metrics = []
        for seq_idx, (ndvi_path, burn_history_paths, curr_burn_path, day) in enumerate(all_valid_sequences):
            try:
                # Load current burn (target) tensor
                curr_burn_tensor = torch.load(curr_burn_path)
                
                # Calculate fire ratio (percentage of fire pixels)
                if curr_burn_tensor.dim() == 2:
                    curr_burn_tensor = curr_burn_tensor.unsqueeze(0)
                
                binary_burn = (curr_burn_tensor > 0).float()
                fire_ratio = binary_burn.mean().item()
                
                # Store sequence with metrics and temporal info
                sequence_fire_metrics.append((seq_idx, fire_ratio, day))
            except Exception as e:
                print(f"Error processing sequence {seq_idx}: {e}")
                continue
        
        if balance_fire:
            # Group sequences by fire content
            fire_threshold = 0.00005

            has_fire = [seq for seq in sequence_fire_metrics if seq[1] > fire_threshold]  # >1% pixels are fire
            no_fire = [seq for seq in sequence_fire_metrics if seq[1] <= fire_threshold]
            
            print(f"Found {len(has_fire)} sequences with fire (>{fire_threshold*100:.4f}% pixels)")
            print(f"Found {len(no_fire)} sequences without significant fire")
            
            # Sort each group by day for temporal coherence
            has_fire.sort(key=lambda x: x[2])
            no_fire.sort(key=lambda x: x[2])
            
            # Calculate split indices for each group
            has_fire_split = int(len(has_fire) * split_ratio)
            no_fire_split = int(len(no_fire) * split_ratio)
            
            # Split each group
            if train:
                fire_sequences = has_fire[:has_fire_split]
                no_fire_sequences = no_fire[:no_fire_split]
            else:
                fire_sequences = has_fire[has_fire_split:]
                no_fire_sequences = no_fire[no_fire_split:]
            
            # Combine and get original indices
            selected_indices = [seq[0] for seq in fire_sequences + no_fire_sequences]
        else:
            # Simple temporal split
            # Sort by day
            sequence_fire_metrics.sort(key=lambda x: x[2])
            
            # Calculate split index
            split_idx = int(len(sequence_fire_metrics) * split_ratio)
            
            if train:
                selected_indices = [seq[0] for seq in sequence_fire_metrics[:split_idx]]
            else:
                selected_indices = [seq[0] for seq in sequence_fire_metrics[split_idx:]]
        
        # Get the selected sequences
        self.valid_sequences = [(all_valid_sequences[idx][0], all_valid_sequences[idx][1], all_valid_sequences[idx][2]) 
                               for idx in selected_indices]
        
        # Print dataset stats
        dataset_type = "Training" if train else "Validation"
        print(f"Created {dataset_type} dataset with {len(self.valid_sequences)} sequences " 
              f"({len(self.valid_sequences) / len(all_valid_sequences) * 100:.1f}% of total)")
        
        # Print fire statistics
        if len(self.valid_sequences) > 0:
            # Calculate fire content in this split
            fire_ratios = []
            for ndvi_path, burn_history_paths, curr_burn_path in self.valid_sequences:
                try:
                    curr_burn_tensor = torch.load(curr_burn_path)
                    if curr_burn_tensor.dim() == 2:
                        curr_burn_tensor = curr_burn_tensor.unsqueeze(0)
                    binary_burn = (curr_burn_tensor > 0).float()
                    fire_ratios.append(binary_burn.mean().item())
                except Exception as e:
                    print(f"Error calculating fire ratio: {e}")
                    continue
            
            if fire_ratios:
                avg_fire_ratio = sum(fire_ratios) / len(fire_ratios)
                fire_sequences = sum(1 for r in fire_ratios if r > 0.01)
                print(f"{dataset_type} set stats: {fire_sequences} sequences with fire " 
                      f"({fire_sequences / len(self.valid_sequences) * 100:.1f}% of set)")
                print(f"Average fire pixels in {dataset_type} set: {avg_fire_ratio * 100:.4f}%")
            
            # Print example sequences
            print("\nExample sequences:")
            for seq in self.valid_sequences[:3]:
                print(f"  NDVI: {os.path.basename(seq[0])}, History: {[os.path.basename(p) for p in seq[1]]}, Curr Burn: {os.path.basename(seq[2])}")
                # Print the day numbers
                history_days = [int(os.path.basename(p).split('_')[2].split('.')[0]) for p in seq[1]]
                curr_day = int(os.path.basename(seq[2]).split('_')[2].split('.')[0])
                print(f"  Day sequence: History days {history_days} → Current day {curr_day}")
    
    def _find_nearest_ndvi_day(self, day):
        """
        Finds the nearest day with available NDVI data
        
        Args:
            day (int): Target day number (1-366)
            
        Returns:
            int: Nearest day number with NDVI data
        """
        # Reference days where NDVI values are available (start of each month)
        ndvi_days = [1, 32, 60, 61, 91, 92, 121, 122, 152, 153, 182, 183, 213, 214, 244, 245, 274, 275, 305, 306, 335, 336]
        return min(ndvi_days, key=lambda x: abs(x - day))
    
    def __len__(self):
        """
        Returns the total number of valid sequences in the dataset
        """
        return len(self.valid_sequences)
    
    def __getitem__(self, idx):
        """
        Fetches a specific sequence by index
        
        Args:
            idx: Index of the sequence to retrieve
            
        Returns:
            Tuple of (ndvi_tensor, burn_history_tensor, curr_burn_tensor)
        """
        ndvi_path, burn_history_paths, curr_burn_path = self.valid_sequences[idx]
        
        try:
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
            burn_history_tensors = [(b > 0).float() for b in burn_history_tensors]
            curr_burn_tensor = (curr_burn_tensor > 0).float()
            
            # Stack burn history along channel dimension
            burn_history_tensor = torch.cat(burn_history_tensors, dim=0)
            
            return ndvi_tensor, burn_history_tensor, curr_burn_tensor
            
        except Exception as e:
            print(f"Error loading tensors at index {idx}: {e}")
            # Return placeholder tensors in case of error
            return (torch.zeros(1, 64, 64), torch.zeros(self.history_days, 64, 64), torch.zeros(1, 64, 64))