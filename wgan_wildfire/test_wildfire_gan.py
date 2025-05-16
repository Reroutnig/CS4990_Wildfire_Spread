import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import glob
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Import your model and utility functions
from models import Generator, Critic
from utils import post_process_predictions, visualize_validation

def load_model(checkpoint_path, device):
    """
    Load trained generator model from checkpoint
    """
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model instances
    generator = Generator(in_channels=5, out_channels=1).to(device)
    critic = Critic(ndvi_channels=1, history_channels=3, output_channels=1).to(device)
    
    # Load weights
    generator.load_state_dict(checkpoint['generator_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    
    # Set to evaluation mode
    generator.eval()
    critic.eval()
    
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    return generator, critic, checkpoint

def get_day_from_filename(filename):
    """Extract day number from filename"""
    base = os.path.basename(filename)
    parts = base.split('_')
    return int(parts[2].split('.')[0])

def load_tensor_files(directory, start_day, end_day):
    """
    Load tensor files for days in the specified range
    """
    files = glob.glob(os.path.join(directory, "*.pt"))
    day_files = {}
    
    for file in files:
        day = get_day_from_filename(file)
        if start_day <= day <= end_day:
            day_files[day] = file
    
    return day_files

def find_consecutive_days(burn_day_files, history_days=3):
    """
    Find sequences of days where we have current day and required history days
    """
    valid_days = []
    all_days = sorted(burn_day_files.keys())
    
    for day in all_days:
        # Check if we have all required history days
        history_needed = [day - i - 1 for i in range(history_days)]
        if all(d in burn_day_files for d in history_needed):
            valid_days.append(day)
    
    return valid_days

def find_nearest_ndvi_day(day):
    """
    Finds the nearest day with available NDVI data
    """
    # Reference days where NDVI values are available (start of each month)
    ndvi_days = [1, 32, 60,61, 91,92, 121,122, 152,153, 182,183, 213,214, 244,245, 274,275, 305,306, 335,336]
    return min(ndvi_days, key=lambda x: abs(x - day))

def create_test_sequence(ndvi_dir, burn_dir, day, history_days=3, device='cpu'):
    """
    Create test sequence for a specific day with history
    """
    # Get nearest NDVI day
    ndvi_day = find_nearest_ndvi_day(day)
    ndvi_file = os.path.join(ndvi_dir, f"NDVI_2020_{ndvi_day}.pt")
    
    # Get burn history files (from oldest to newest)
    history_days_list = [day - i - 1 for i in range(history_days)]
    history_days_list.reverse()  # Ensure oldest day is first
    burn_history_files = [os.path.join(burn_dir, f"burn_2020_{d}.pt") for d in history_days_list]
    
    # Get current burn file (ground truth)
    curr_burn_file = os.path.join(burn_dir, f"burn_2020_{day}.pt")
    
    # Check if all files exist
    if not os.path.exists(ndvi_file):
        raise FileNotFoundError(f"NDVI file not found: {ndvi_file}")
    for file in burn_history_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Burn history file not found: {file}")
    if not os.path.exists(curr_burn_file):
        raise FileNotFoundError(f"Current burn file not found: {curr_burn_file}")
    
    # Load tensors
    ndvi_tensor = torch.load(ndvi_file)
    burn_history_tensors = [torch.load(file) for file in burn_history_files]
    curr_burn_tensor = torch.load(curr_burn_file)
    
    # Ensure 3D tensors
    if ndvi_tensor.dim() == 2:
        ndvi_tensor = ndvi_tensor.unsqueeze(0)
    if curr_burn_tensor.dim() == 2:
        curr_burn_tensor = curr_burn_tensor.unsqueeze(0)
    
    # Process burn history tensors
    for i in range(len(burn_history_tensors)):
        if burn_history_tensors[i].dim() == 2:
            burn_history_tensors[i] = burn_history_tensors[i].unsqueeze(0)
    
    # Normalize NDVI to [-1, 1] range
    ndvi_tensor = torch.clamp(ndvi_tensor, -1.0, 1.0)
    
    # Convert burn tensors to binary in [0,1] range
    burn_history_tensors = [(b > 0).float() for b in burn_history_tensors]
    curr_burn_tensor = (curr_burn_tensor > 0).float()
    
    # Stack burn history along channel dimension (dim=0)
    burn_history_tensor = torch.cat(burn_history_tensors, dim=0)
    
    # Add batch dimension if needed
    if ndvi_tensor.dim() == 3:  # [C,H,W]
        ndvi_tensor = ndvi_tensor.unsqueeze(0)  # [B,C,H,W]
    if curr_burn_tensor.dim() == 3:
        curr_burn_tensor = curr_burn_tensor.unsqueeze(0)
        
    # The burn_history_tensor should have shape [history_days, H, W]
    # Add batch dimension
    burn_history_tensor = burn_history_tensor.unsqueeze(0)  # [1, history_days, H, W]
    
    # Move to device
    ndvi_tensor = ndvi_tensor.to(device)
    burn_history_tensor = burn_history_tensor.to(device)
    curr_burn_tensor = curr_burn_tensor.to(device)
    
    # Debug info
    print(f"NDVI shape: {ndvi_tensor.shape}")
    print(f"Burn history shape: {burn_history_tensor.shape}")
    print(f"Current burn shape: {curr_burn_tensor.shape}")
    
    return ndvi_tensor, burn_history_tensor, curr_burn_tensor

def calculate_metrics(pred, target, threshold=0.5):
    """
    Calculate performance metrics
    """
    # Convert to binary using threshold
    pred_binary = (pred > threshold).float()
    target_binary = (target > 0.5).float()
    
    # Calculate IoU
    intersection = (pred_binary * target_binary).sum().item()
    union = (pred_binary + target_binary).clamp(0, 1).sum().item()
    iou = intersection / (union + 1e-8)
    
    # Convert to numpy for sklearn metrics
    pred_np = pred_binary.cpu().numpy().flatten()
    target_np = target_binary.cpu().numpy().flatten()
    
    # Calculate precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        target_np, pred_np, average='binary', zero_division=0
    )
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(target_np, pred_np, labels=[0, 1]).ravel()
    
    # Calculate metrics for fire pixels (positives)
    total_fire_pixels = (target_binary > 0.5).sum().item()
    total_pred_fire = (pred_binary > 0.5).sum().item()
    
    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_fire_pixels': total_fire_pixels,
        'total_pred_fire': total_pred_fire,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }

def generate_prediction(generator, ndvi, burn_history, device):
    """
    Generate prediction using the model
    """
    with torch.no_grad():
        # Create fire probability channel (global context)
        # Estimate based on history (we don't know ground truth for real prediction)
        fire_probability = torch.sum(burn_history > 0.5).float() / torch.numel(burn_history)
        fire_prob_channel = torch.ones_like(ndvi) * fire_probability
        
        # Create generator input
        generator_input = torch.cat([ndvi, burn_history, fire_prob_channel], dim=1)
        
        # Print shape for debugging
        print(f"Generator input shape: {generator_input.shape}")
        
        # Generate prediction
        prediction = generator(generator_input)
    
    return prediction

def test_model(generator, ndvi_dir, burn_dir, output_dir, start_day, end_day, 
               history_days=3, device='cpu', threshold=0.5, post_process=True,
               min_area=0, max_gap=0, checkpoint_path=None):
    """
    Test model on specified days
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load burn files for the specified range
    burn_day_files = load_tensor_files(burn_dir, start_day - history_days, end_day)
    
    # Find consecutive day sequences
    valid_days = find_consecutive_days(burn_day_files, history_days)
    print(f"Found {len(valid_days)} valid days to test in range {start_day}-{end_day}")
    
    if len(valid_days) == 0:
        print("No valid test days found. Check your day range and data.")
        return
    
    # Set post-processing if needed
    if post_process:
        generator.set_post_processing(True, threshold=threshold, min_area=min_area, max_gap=max_gap)
    else:
        generator.set_post_processing(False)
    
    # Track metrics
    all_metrics = []
    
    # Test on each valid day
    for day in tqdm(valid_days):
        try:
            print(f"\nProcessing day {day}:")
            # Create test sequence
            ndvi, burn_history, curr_burn = create_test_sequence(
                ndvi_dir, burn_dir, day, history_days, device
            )
            
            # Generate prediction
            prediction = generate_prediction(generator, ndvi, burn_history, device)
            
            # Ensure prediction has correct shape
            print(f"Prediction shape: {prediction.shape}")
            print(f"Current burn shape: {curr_burn.shape}")
            
            # Calculate metrics
            metrics = calculate_metrics(prediction[0], curr_burn[0], threshold)
            metrics['day'] = day
            all_metrics.append(metrics)
            
            # Save visualization
            visualize_validation(
                prediction[0].cpu(), 
                curr_burn[0].cpu(),
                save_path=os.path.join(output_dir, f"day_{day}_prediction.png"),
                title=f"Day {day} - IoU: {metrics['iou']:.4f}, F1: {metrics['f1']:.4f}"
            )
            
        except Exception as e:
            print(f"Error processing day {day}: {e}")
            import traceback
            traceback.print_exc()
    
    # Check if we have any valid metrics
    if not all_metrics:
        print("No valid metrics collected. Check your data and model compatibility.")
        return None
    
    # Save results
    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    
    # Calculate and save summary
    summary = {
        'mean_iou': results_df['iou'].mean(),
        'mean_precision': results_df['precision'].mean(),
        'mean_recall': results_df['recall'].mean(),
        'mean_f1': results_df['f1'].mean(),
        'median_iou': results_df['iou'].median(),
        'median_f1': results_df['f1'].median(),
        'total_days': len(results_df),
        'total_fire_pixels': results_df['total_fire_pixels'].sum(),
        'total_predicted_fire': results_df['total_pred_fire'].sum(),
        'total_true_positives': results_df['true_positives'].sum(),
        'total_false_positives': results_df['false_positives'].sum(),
        'total_false_negatives': results_df['false_negatives'].sum(),
    }
    
    # Calculate global metrics
    if summary['total_fire_pixels'] > 0:
        summary['global_precision'] = summary['total_true_positives'] / (summary['total_true_positives'] + summary['total_false_positives'] + 1e-8)
        summary['global_recall'] = summary['total_true_positives'] / (summary['total_true_positives'] + summary['total_false_negatives'] + 1e-8)
        summary['global_f1'] = 2 * summary['global_precision'] * summary['global_recall'] / (summary['global_precision'] + summary['global_recall'] + 1e-8)
    
    # Save summary
    with open(os.path.join(output_dir, "summary.txt"), 'w') as f:
        f.write("=== Wildfire Prediction Model Test Results ===\n\n")
        f.write(f"Model Checkpoint: {checkpoint_path}\n\n")
        f.write(f"Days tested: {summary['total_days']} (range: {start_day}-{end_day})\n")
        f.write(f"Threshold: {threshold}, Post-process: {post_process}, Min area: {min_area}, Max gap: {max_gap}\n\n")
        
        f.write("=== Mean Metrics ===\n")
        f.write(f"IoU: {summary['mean_iou']:.4f}\n")
        f.write(f"Precision: {summary['mean_precision']:.4f}\n")
        f.write(f"Recall: {summary['mean_recall']:.4f}\n")
        f.write(f"F1: {summary['mean_f1']:.4f}\n\n")
        
        f.write("=== Median Metrics ===\n")
        f.write(f"IoU: {summary['median_iou']:.4f}\n")
        f.write(f"F1: {summary['median_f1']:.4f}\n\n")
        
        f.write("=== Global Metrics ===\n")
        f.write(f"Total fire pixels (ground truth): {summary['total_fire_pixels']}\n")
        f.write(f"Total predicted fire pixels: {summary['total_predicted_fire']}\n")
        f.write(f"True positives: {summary['total_true_positives']}\n")
        f.write(f"False positives: {summary['total_false_positives']}\n")
        f.write(f"False negatives: {summary['total_false_negatives']}\n\n")
        
        if 'global_precision' in summary:
            f.write(f"Global precision: {summary['global_precision']:.4f}\n")
            f.write(f"Global recall: {summary['global_recall']:.4f}\n")
            f.write(f"Global F1: {summary['global_f1']:.4f}\n")
        
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Mean IoU: {summary['mean_iou']:.4f}")
    print(f"Mean F1: {summary['mean_f1']:.4f}")
    print(f"Global metrics calculated over {summary['total_fire_pixels']} fire pixels")
    if 'global_f1' in summary:
        print(f"Global F1: {summary['global_f1']:.4f}")
    
    # Generate plots
    try:
        plot_metrics(results_df, output_dir)
    except Exception as e:
        print(f"Error generating plots: {e}")
    
    return results_df

def plot_metrics(results_df, output_dir):
    """
    Generate plots of metrics over time
    """
    # Sort by day
    results_df = results_df.sort_values('day')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot IoU over time
    axes[0, 0].plot(results_df['day'], results_df['iou'], 'b-', marker='o')
    axes[0, 0].set_title('IoU Score by Day')
    axes[0, 0].set_xlabel('Day of Year')
    axes[0, 0].set_ylabel('IoU Score')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot F1 score over time
    axes[0, 1].plot(results_df['day'], results_df['f1'], 'g-', marker='o')
    axes[0, 1].set_title('F1 Score by Day')
    axes[0, 1].set_xlabel('Day of Year')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot Precision and Recall
    axes[1, 0].plot(results_df['day'], results_df['precision'], 'r-', marker='o', label='Precision')
    axes[1, 0].plot(results_df['day'], results_df['recall'], 'b-', marker='s', label='Recall')
    axes[1, 0].set_title('Precision and Recall by Day')
    axes[1, 0].set_xlabel('Day of Year')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot ground truth vs predicted fire pixels
    axes[1, 1].plot(results_df['day'], results_df['total_fire_pixels'], 'r-', marker='o', label='Ground Truth')
    axes[1, 1].plot(results_df['day'], results_df['total_pred_fire'], 'b-', marker='s', label='Predicted')
    axes[1, 1].set_title('Fire Pixels Count by Day')
    axes[1, 1].set_xlabel('Day of Year')
    axes[1, 1].set_ylabel('Number of Fire Pixels')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_over_time.png'), dpi=150)
    plt.close()
    
    # Plot histogram of metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(results_df['iou'], bins=20, alpha=0.7)
    axes[0, 0].set_title('IoU Score Distribution')
    axes[0, 0].set_xlabel('IoU Score')
    axes[0, 0].set_ylabel('Frequency')
    
    axes[0, 1].hist(results_df['f1'], bins=20, alpha=0.7)
    axes[0, 1].set_title('F1 Score Distribution')
    axes[0, 1].set_xlabel('F1 Score')
    axes[0, 1].set_ylabel('Frequency')
    
    axes[1, 0].hist(results_df['precision'], bins=20, alpha=0.7)
    axes[1, 0].set_title('Precision Distribution')
    axes[1, 0].set_xlabel('Precision')
    axes[1, 0].set_ylabel('Frequency')
    
    axes[1, 1].hist(results_df['recall'], bins=20, alpha=0.7)
    axes[1, 1].set_title('Recall Distribution')
    axes[1, 1].set_xlabel('Recall')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_distribution.png'), dpi=150)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Wildfire Prediction GAN')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--ndvi_dir', type=str, default='../data/ndvi_west_coast_tensors', help='Directory with NDVI tensors')
    parser.add_argument('--burn_dir', type=str, default='../data/burn_wc_tensors', help='Directory with burn tensors')
    parser.add_argument('--output_dir', type=str, default='test_results', help='Directory to save test results')
    parser.add_argument('--start_day', type=int, required=True, help='First day to test (day of year)')
    parser.add_argument('--end_day', type=int, required=True, help='Last day to test (day of year)')
    parser.add_argument('--history_days', type=int, default=3, help='Number of previous days to use for prediction')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary prediction')
    parser.add_argument('--post_process', action='store_true', help='Apply post-processing to predictions')
    parser.add_argument('--min_area', type=int, default=0, help='Minimum area for post-processing')
    parser.add_argument('--max_gap', type=int, default=0, help='Maximum gap for post-processing')
    parser.add_argument('--debug', action='store_true', help='Enable extra debug output')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    generator, critic, checkpoint = load_model(args.checkpoint, device)
    
    # Print model details
    print(f"Generator in_channels: {generator.ndvi_channels + generator.history_channels + 1}")
    
    # Create output directory with informative name
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    output_name = f"test_{args.start_day}_{args.end_day}_{timestamp}"
    output_dir = os.path.join(args.output_dir, output_name)
    
    # Test model
    try:
        test_results = test_model(
            generator=generator,
            ndvi_dir=args.ndvi_dir,
            burn_dir=args.burn_dir,
            output_dir=output_dir,
            start_day=args.start_day,
            end_day=args.end_day,
            history_days=args.history_days,
            device=device,
            threshold=args.threshold,
            post_process=args.post_process,
            min_area=args.min_area,
            max_gap=args.max_gap,
            checkpoint_path=args.checkpoint
        )
        
        print(f"Testing complete. Results saved to {output_dir}")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()