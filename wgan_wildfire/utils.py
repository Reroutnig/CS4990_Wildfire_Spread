import torch
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show

# ------------------------------
# Post-Processing Function for Spatial Constraints
# ------------------------------
def post_process_predictions(predictions, threshold=0.8, min_area=0, max_gap=0):
    """
    Post-process predictions to enforce spatial constraints
    
    Args:
        predictions: Tensor of prediction probabilities (between 0 and 1)
        threshold: Value to use for thresholding predictions
        min_area: Minimum area of a fire region to keep (removes noise)
        max_gap: Maximum gap between fire pixels to fill (connects nearby fires)
        
    Returns:
        Processed predictions tensor with same shape and value range as input
    """
    # Remember original device
    device = predictions.device
    
    # Ensure predictions are in [0,1] range
    if torch.min(predictions) < 0 or torch.max(predictions) > 1:
        raise ValueError("Predictions must be in [0,1] range for post-processing")
    
    # Convert to numpy for processing (process each batch item separately)
    processed_batch = []
    
    for i in range(predictions.shape[0]):
        # Get this batch item and move to CPU
        pred = predictions[i].cpu().detach().numpy()
        if len(pred.shape) > 2:  # If [C,H,W] format
            pred = pred.squeeze()  # Remove channel dim if present
        
        # Threshold to get binary prediction
        binary_pred = (pred > threshold).astype(np.float32)
        
        # Label connected components (fires)
        labeled, num_features = ndimage.label(binary_pred)
        
        # Remove small components (noise reduction)
        if min_area > 0:
            for component_id in range(1, num_features+1):
                component_mask = (labeled == component_id)
                component_size = component_mask.sum()
                if component_size < min_area:
                    binary_pred[component_mask] = 0
        
        # Optional: Fill small gaps in fire regions (connect nearby fires)
        if max_gap > 0:
            # Dilate
            dilated = ndimage.binary_dilation(binary_pred, iterations=max_gap)
            # Fill holes - areas completely surrounded by fire
            filled = ndimage.binary_fill_holes(dilated)
            # Erode back to approximate original size
            eroded = ndimage.binary_erosion(filled, iterations=max_gap)
            # Use as our new prediction
            binary_pred = eroded.astype(np.float32)
        
        # Convert back to original shape
        if len(predictions.shape) == 4:  # [B,C,H,W]
            processed = binary_pred.reshape(1, binary_pred.shape[0], binary_pred.shape[1])
        else:
            processed = binary_pred
            
        processed_batch.append(torch.from_numpy(processed))
    
    # Stack batch and move back to original device
    processed_tensor = torch.stack(processed_batch).to(device)
    
    # Reshape to match original
    if len(predictions.shape) == 4:  # [B,C,H,W]
        processed_tensor = processed_tensor.unsqueeze(1)
    
    return processed_tensor

# ------------------------------
# Visualization function for debugging post-processing
# ------------------------------
def visualize_post_processing(original, processed, save_path=None, title=None):
    """
    Visualize the effects of post-processing on a prediction
    
    Args:
        original: Original prediction tensor (in [0,1] range)
        processed: Post-processed prediction tensor (in [0,1] range)
        save_path: Path to save the visualization, if None will show plot
        title: Optional title for the visualization
    """
    # Convert tensors to numpy if needed
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(processed, torch.Tensor):
        processed = processed.detach().cpu().numpy()
    
    # Squeeze dimensions if needed
    if len(original.shape) > 2:
        original = original.squeeze()
    if len(processed.shape) > 2:
        processed = processed.squeeze()
    
    # Create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Add title if provided
    if title:
        fig.suptitle(title, fontsize=16)
    
    # Show original prediction
    im0 = axes[0].imshow(original, cmap='Reds', vmin=0, vmax=1)
    axes[0].set_title('Original Prediction')
    fig.colorbar(im0, ax=axes[0])
    
    # Show processed prediction
    im1 = axes[1].imshow(processed, cmap='Reds', vmin=0, vmax=1)
    axes[1].set_title('Processed Prediction')
    fig.colorbar(im1, ax=axes[1])
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def visualize_validation(prediction, ground_truth, save_path=None, title=None):
    """
    Visualize validation predictions compared to ground truth
    
    Args:
        prediction: Model prediction tensor (in [0,1] range)
        ground_truth: Ground truth tensor (in [0,1] range)
        save_path: Path to save the visualization, if None will show plot
        title: Optional title for the visualization
    """
    # Convert tensors to numpy if needed
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()
    
    # Squeeze dimensions if needed
    if len(prediction.shape) > 2:
        prediction = prediction.squeeze()
    if len(ground_truth.shape) > 2:
        ground_truth = ground_truth.squeeze()
    
    # Create figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Add title if provided
    if title:
        fig.suptitle(title, fontsize=16)
    
    # Show prediction
    im0 = axes[0].imshow(prediction, cmap='Reds', vmin=0, vmax=1)
    axes[0].set_title('Model Prediction')
    fig.colorbar(im0, ax=axes[0])
    
    # Show ground truth
    im1 = axes[1].imshow(ground_truth, cmap='Reds', vmin=0, vmax=1)
    axes[1].set_title('Ground Truth')
    fig.colorbar(im1, ax=axes[1])
    
    # Show difference (error)
    diff = np.abs(prediction - ground_truth)
    im2 = axes[2].imshow(diff, cmap='viridis', vmin=0, vmax=1)
    axes[2].set_title('Absolute Error')
    fig.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def visualize_training_losses(history, save_path=None, figure_size=(12, 10)):
    """
    Visualize training losses over epochs
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save visualization image
        figure_size: Size of the figure (width, height)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    # Helper function to convert potential torch tensors to numpy
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        elif isinstance(x, list) and len(x) > 0 and isinstance(x[0], torch.Tensor):
            return [item.detach().cpu().numpy() for item in x]
        return x
    
    # Create a copy of history with all tensors converted to numpy
    history_np = {}
    for key, value in history.items():
        history_np[key] = to_numpy(value)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figure_size)
    
    # Plot generator and critic losses
    epochs = np.arange(1, len(history_np['loss_G']) + 1)
    
    # Plot main losses (Generator and Critic)
    axes[0, 0].plot(epochs, history_np['loss_G'], 'b-', label='Generator Loss')
    axes[0, 0].plot(epochs, history_np['loss_C'], 'r-', label='Critic Loss')
    axes[0, 0].set_title('Main Losses')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot generator component losses
    axes[0, 1].plot(epochs, history_np['loss_G_adv'], 'g-', label='Adversarial Loss')
    axes[0, 1].plot(epochs, history_np['loss_L1'], 'm-', label='L1 Loss')
    axes[0, 1].plot(epochs, history_np['loss_fire'], 'c-', label='Fire Loss')
    axes[0, 1].plot(epochs, history_np['loss_confidence'], 'y-', label='Confidence Loss')
    axes[0, 1].set_title('Generator Component Losses')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot gradient penalty
    axes[1, 0].plot(epochs, history_np['gradient_penalty'], 'k-', label='Gradient Penalty')
    axes[1, 0].set_title('Gradient Penalty')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot learning rates and NDVI weight
    if 'lr_G' in history_np and 'lr_C' in history_np:
        axes[1, 1].plot(epochs, history_np['lr_G'], 'orange', label='Generator LR')
        axes[1, 1].plot(epochs, history_np['lr_C'], 'r--', label='Critic LR')
        axes[1, 1].set_title('Learning Rates')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
    elif 'lr' in history_np:
        axes[1, 1].plot(epochs, history_np['lr'], 'orange', label='Learning Rate')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
    
    # Plot NDVI weight on secondary axis if available
    if 'ndvi_weight' in history_np:
        ax2 = axes[1, 1].twinx()
        ax2.plot(epochs, history_np['ndvi_weight'], 'g-', label='NDVI Weight')
        ax2.set_ylabel('NDVI Weight', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
    
    axes[1, 1].legend(loc='upper left')
    if 'ndvi_weight' in history_np:
        ax2.legend(loc='upper right')
    
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add validation losses if available
    if 'val_loss_G' in history_np and len(history_np['val_loss_G']) > 0:
        # Find valid epochs where validation was run
        validation_frequency = max(1, len(history_np['loss_G']) // max(1, len(history_np['val_loss_G'])))
        val_epochs = np.arange(validation_frequency, len(history_np['loss_G']) + 1, validation_frequency)
        val_epochs = val_epochs[:len(history_np['val_loss_G'])]
        
        # Add validation losses to main losses plot
        axes[0, 0].plot(val_epochs, history_np['val_loss_G'], 'b--', label='Val Generator Loss')
        
        # Add validation component losses
        if 'val_loss_G_adv' in history_np:
            axes[0, 1].plot(val_epochs, history_np['val_loss_G_adv'], 'g--', label='Val Adversarial Loss')
        if 'val_loss_L1' in history_np:
            axes[0, 1].plot(val_epochs, history_np['val_loss_L1'], 'm--', label='Val L1 Loss')
        if 'val_loss_fire' in history_np:
            axes[0, 1].plot(val_epochs, history_np['val_loss_fire'], 'c--', label='Val Fire Loss')
        if 'val_loss_confidence' in history_np:
            axes[0, 1].plot(val_epochs, history_np['val_loss_confidence'], 'y--', label='Val Confidence Loss')
        
        # Update legends
        axes[0, 0].legend()
        axes[0, 1].legend()
    
    # Add overall title
    plt.suptitle('Training Progress', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for title
    
    # Add final loss values as text
    if len(history_np['loss_G']) > 0:
        fig.text(0.01, 0.01, 
                f"Final losses - G: {history_np['loss_G'][-1]:.4f}, C: {history_np['loss_C'][-1]:.4f}, "
                f"GP: {history_np['gradient_penalty'][-1]:.4f}", fontsize=9)
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

# Create a nullcontext class for systems without PyTorch's native nullcontext
class nullcontext:
    def __enter__(self):
        return None
    def __exit__(self, *excinfo):
        pass

# For consistent visualization across training runs
def visualize_fixed_sample_progress(generator, sample_tuple, epoch, ndvi_weight, save_path=None):
    """
    Visualize progress on a fixed sample throughout training
    
    Args:
        generator: The trained generator model
        sample_tuple: Tuple of (ndvi, burn_history, curr_burn) tensors
        epoch: Current epoch number
        ndvi_weight: Current NDVI weight from curriculum learning
        save_path: Path to save the visualization
    """
    # Extract data
    ndvi, burn_history, curr_burn = sample_tuple
    
    # Set evaluation mode
    generator.eval()
    
    with torch.no_grad():
        # Create generator input
        fire_probability = torch.sum(curr_burn > 0.5).float() / torch.numel(curr_burn)
        fire_prob_channel = torch.ones_like(ndvi) * fire_probability
        input_G = torch.cat((ndvi, burn_history, fire_prob_channel), dim=1)
        
        # Set NDVI weight in the generator
        original_weight = generator.ndvi_weight  # Save original
        generator.set_ndvi_weight(ndvi_weight)
        
        # Generate prediction
        prediction = generator(input_G)
        
        # Restore original weight
        generator.set_ndvi_weight(original_weight)
    
    # Visualize the prediction vs ground truth
    visualize_validation(
        prediction[0], curr_burn[0],
        save_path=save_path,
        title=f"Fixed Sample - Epoch {epoch} (NDVI Weight: {ndvi_weight:.3f})"
    )
    
    # Return to training mode
    generator.train()

# Visualization function to track mode collapse
def visualize_fire_distribution(predictions, ground_truth, save_path=None, epoch=None):
    """
    Visualize the distribution of fire probability values to detect mode collapse
    
    Args:
        predictions: Batch of predictions from the generator
        ground_truth: Corresponding ground truth
        save_path: Path to save the visualization
        epoch: Current epoch number for the title
    """
    # Convert to numpy if tensors
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy().flatten()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy().flatten()
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create histogram of prediction values
    plt.subplot(1, 2, 1)
    plt.hist(predictions, bins=50, alpha=0.7, color='blue')
    plt.title('Distribution of Prediction Values')
    plt.xlabel('Prediction Value')
    plt.ylabel('Frequency')
    plt.xlim(0, 1)
    
    # Create scatter plot of prediction vs ground truth
    plt.subplot(1, 2, 2)
    # Sample at most 5000 points to avoid overcrowding
    sample_size = min(5000, len(predictions))
    idx = np.random.choice(len(predictions), sample_size, replace=False)
    plt.scatter(ground_truth[idx], predictions[idx], alpha=0.1, s=1)
    plt.plot([0, 1], [0, 1], 'r--')  # Perfect prediction line
    plt.title('Prediction vs Ground Truth')
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Add overall title if epoch provided
    if epoch is not None:
        plt.suptitle(f'Fire Distribution Analysis - Epoch {epoch}', fontsize=16)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def visualize_prediction_with_original_ndvi(prediction, ground_truth, ndvi_tif_path, save_path=None, title=None):
    """
    Visualize fire prediction overlaid on original NDVI tif file
    
    Args:
        prediction: Model prediction tensor (in [0,1] range)
        ground_truth: Ground truth tensor (in [0,1] range)
        ndvi_tif_path: Path to the original NDVI tif file
        save_path: Path to save the visualization
        title: Optional title for the visualization
    """
    # Convert tensors to numpy if needed
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()
    
    # Squeeze dimensions if needed
    if len(prediction.shape) > 2:
        prediction = prediction.squeeze()
    if len(ground_truth.shape) > 2:
        ground_truth = ground_truth.squeeze()
    
    # Open the original NDVI tif file
    with rasterio.open(ndvi_tif_path) as src:
        ndvi_original = src.read(1)  # Read the first band
        
        # You might need to resize if dimensions don't match
        if ndvi_original.shape != prediction.shape:
            from skimage.transform import resize
            ndvi_original = resize(ndvi_original, prediction.shape, 
                                   preserve_range=True, anti_aliasing=True)
    
    # Create figure and axes
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Add title if provided
    if title:
        fig.suptitle(title, fontsize=16)
    
    # 1. Show original NDVI data
    with rasterio.open(ndvi_tif_path) as src:
        show(src, ax=axes[0, 0], title="Original NDVI")
    
    # 2. Show fire prediction
    im1 = axes[0, 1].imshow(prediction, cmap='Reds', vmin=0, vmax=1)
    axes[0, 1].set_title('Fire Prediction')
    fig.colorbar(im1, ax=axes[0, 1])
    
    # 3. Overlay prediction on original NDVI
    with rasterio.open(ndvi_tif_path) as src:
        show(src, ax=axes[1, 0])
    
    # Create a mask for fire prediction
    fire_mask = prediction > 0.5  # Threshold for clearer visualization
    
    # Create overlay
    overlay = np.zeros((*fire_mask.shape, 4))  # RGBA
    overlay[fire_mask, 0] = 1.0  # Red for fire
    overlay[fire_mask, 1] = 1.0
    overlay[fire_mask, 2] = 0.0
    overlay[fire_mask, 3] = 0.85  # Alpha (transparency)
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('Prediction Overlay on NDVI')
    
    # 4. Overlay ground truth on original NDVI
    with rasterio.open(ndvi_tif_path) as src:
        show(src, ax=axes[1, 1])
    
    gt_mask = ground_truth > 0.5
    gt_overlay = np.zeros((*gt_mask.shape, 4))
    gt_overlay[gt_mask, 0] = 1.0  # Red for fire
    gt_overlay[gt_mask, 1] = 1.0
    gt_overlay[gt_mask, 2] = 0.0
    gt_overlay[gt_mask, 3] = 0.85  # Alpha
    axes[1, 1].imshow(gt_overlay)
    axes[1, 1].set_title('Ground Truth Overlay on NDVI')
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()