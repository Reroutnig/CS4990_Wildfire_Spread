import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import post_process_predictions, visualize_post_processing, visualize_validation, visualize_training_losses
from losses import FocalLoss
from curriculum_learning_scheduler import CurriculumLearningScheduler

# ------------------------------
# Gradient Penalty Function for WGAN-GP
# ------------------------------
def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """
    Computes the gradient penalty for WGAN-GP
    """
    # Random weight for interpolation
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    
    # Interpolated samples
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    
    # Critic scores for interpolated samples
    d_interpolates = critic(interpolates)
    
    # Get gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Compute gradient penalty
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty


# ------------------------------
# Training Functions for Multi-Day WGAN with NDVI and Improved Loss
# ------------------------------
def train_wgan_generator_ndvi(generator, critic, ndvi, burn_history, curr_burn, optimizer_G, 
                             lambda_L1=100, lambda_fire=50.0, fire_weight=250.0, alpha=0.25, gamma=2.0,
                             lambda_confidence=10.0):
    """
    Train generator with Focal Loss, improved class weighting, and confidence loss
    """
    optimizer_G.zero_grad()

    try:
        # Build input tensor: NDVI + burn history + global fire probability
        fire_probability = torch.sum(curr_burn > 0.5).float() / torch.numel(curr_burn)
        fire_prob_channel = torch.ones_like(ndvi) * fire_probability
        
        # Input is: [NDVI, burn_history days, fire_probability]
        input_G = torch.cat((ndvi, burn_history, fire_prob_channel), dim=1)
        
        # Generate burn prediction - now in [0,1] range with sigmoid activation
        gen_burn = generator(input_G)
        
        # Ensure dimensions match
        if gen_burn.shape != curr_burn.shape:
            gen_burn = F.interpolate(gen_burn, size=curr_burn.shape[2:], mode='bilinear', align_corners=False)
        
        # Input to critic: NDVI + burn history + generated burn OR current burn
        input_fake = torch.cat((ndvi, burn_history, gen_burn), dim=1)
        
        # Calculate critic score
        critic_fake = critic(input_fake)
        loss_G_adv = -torch.mean(critic_fake)
        
        # L1 loss between generated and actual burn
        # Note: Both are now in [0,1] range, no conversion needed
        weights = torch.ones_like(curr_burn)
        weights[curr_burn > 0.5] = fire_weight  # Significantly increased weight for fire pixels
        loss_L1 = torch.mean(weights * torch.abs(gen_burn - curr_burn))
        
        # Use Focal Loss for better handling of class imbalance
        focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean', from_logits=False)
        
        # Calculate class weights based on inverse class frequency
        num_fire_pixels = torch.sum(curr_burn > 0.5).float()
        num_nonfire_pixels = torch.numel(curr_burn) - num_fire_pixels
        if num_fire_pixels > 0:
            fire_to_nonfire_ratio = num_nonfire_pixels / (num_fire_pixels + 1e-8)
            # Create per-pixel weight tensor
            dynamic_weights = torch.ones_like(curr_burn)
            dynamic_weights[curr_burn > 0.5] = fire_to_nonfire_ratio
            # Apply additional emphasis based on fixed fire_weight
            dynamic_weights[curr_burn > 0.5] *= fire_weight / 100.0
        else:
            dynamic_weights = None
        
        # Apply Focal Loss with dynamic class weights
        if dynamic_weights is not None:
            # We need to handle reduction manually if we're using custom weights
            # First compute the per-pixel loss with 'none' reduction
            focal_custom = FocalLoss(alpha=alpha, gamma=gamma, reduction='none')
            fire_loss_raw = focal_custom(gen_burn, curr_burn, weights=dynamic_weights)
            # Then manually reduce it to a scalar
            loss_fire = fire_loss_raw.mean()
        else:
            # Use the standard focal loss with built-in 'mean' reduction
            loss_fire = focal_loss(gen_burn, curr_burn)
        
        # Confidence loss to encourage more definitive predictions
        # For [0,1] range, we want predictions to be close to 0 or 1
        confidence_loss = -torch.mean(torch.abs(gen_burn - 0.5)) + 0.5
        
        # Total generator loss
        loss_G = loss_G_adv + lambda_L1 * loss_L1 + lambda_fire * loss_fire + lambda_confidence * confidence_loss
        loss_G = torch.clamp(loss_G, min=-100.0, max=500.0)  # Prevent extreme values
        
        # Backpropagate and update
        loss_G.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        optimizer_G.step()

        return loss_G.item(), loss_G_adv.item(), loss_L1.item(), loss_fire.item(), confidence_loss.item(), gen_burn

    except Exception as e:
        print(f"Error in train_wgan_generator_ndvi: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0, 0.0, 0.0, 0.0, None


def train_wgan_critic_ndvi(critic, ndvi, burn_history, curr_burn, gen_burn, optimizer_C, device, 
                          lambda_gp=10, epoch=0, noise_factor=0.05, noise_decay=0.995):
    """
    Train critic with gradient penalty and noise for stability
    """
    # Zero the gradients
    optimizer_C.zero_grad()
    
    try:
        # If generator output is None, skip this batch
        if gen_burn is None:
            print("Skipping critic training due to generator error")
            return 0.0, 0.0
        
        # Calculate current noise level
        current_noise = max(noise_factor * (noise_decay ** epoch), 0.001)
        
        # Real samples: NDVI + burn_history + curr_burn
        input_real = torch.cat((ndvi, burn_history, curr_burn), dim=1)
        
        # Fake samples: NDVI + burn_history + generated burn
        input_fake = torch.cat((ndvi, burn_history, gen_burn.detach()), dim=1)
        
        # Apply noise to critic inputs during early training
        if epoch < 30:
            # Create noise tensors
            real_noise = torch.randn_like(input_real) * current_noise
            fake_noise = torch.randn_like(input_fake) * current_noise
            
            # Add noise to inputs
            input_real = input_real + real_noise
            input_fake = input_fake + fake_noise
        
        # Get critic scores
        critic_real = critic(input_real)
        critic_fake = critic(input_fake)
        
        # Wasserstein loss
        loss_C = torch.mean(critic_fake) - torch.mean(critic_real)
        
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(critic, input_real, input_fake, device)
        
        # Total critic loss
        loss_C_total = loss_C + lambda_gp * gradient_penalty
        
        # Backpropagate and update
        loss_C_total.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
        optimizer_C.step()
        
        return loss_C.item(), gradient_penalty.item()
        
    except Exception as e:
        print(f"Error in train_wgan_critic_ndvi: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0


# ------------------------------
# WGAN Training Loop with Multiple Day History and Improved Loss
# ------------------------------
def train_wgan_ndvi_model(generator, critic, train_dataloader, val_dataloader=None, epochs=100, device=None, 
                         generator_lr=None, critic_lr=None, lr=None, lambda_L1=100, save_path="new_model/wgan_improved", 
                         start_epoch=0, checkpoint=None, n_critic=5, lambda_gp=10,
                         lambda_fire=50.0, fire_weight=250.0, alpha=0.25, gamma=2.0,
                         lambda_confidence=10.0, lr_decay_epochs=30, lr_decay_rate=0.5,
                         visualization_dir=None, save_every=5, vis_every=100,
                         validate_every=1, precision='fp32', loss_plot_every=20,
                         curriculum_start=20, curriculum_end=60, 
                         curriculum_max_ndvi=1.0, curriculum_type='linear',
                         noise_factor=0.05, noise_decay=0.995):
    """
    Training loop for WGAN-GP with NDVI and multi-day burn history data
    with improved loss functions, validation, confidence loss, learning rate scheduling,
    and curriculum learning to gradually introduce NDVI dependency.
    """
    # Set up automatic mixed precision if requested
    use_amp = precision == 'fp16'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    if use_amp:
        print("⚡ Using mixed precision (FP16) training for faster performance")
    
    if lr is not None:
        if generator_lr is None:
            generator_lr = lr * 0.4  # Use 40% of single lr for generator
    if critic_lr is None:
        critic_lr = lr  # Use full lr for critic
    else:
        # Use defaults if nothing provided
        generator_lr = generator_lr or 0.00003
        critic_lr = critic_lr or 0.00008

    # Create optimizers (β1=0 for WGAN stability)
    optimizer_G = optim.Adam(generator.parameters(), lr=generator_lr, betas=(0.0, 0.9))
    optimizer_C = optim.Adam(critic.parameters(), lr=critic_lr, betas=(0.0, 0.9))
    
    # Learning rate scheduler - reduces learning rate every lr_decay_epochs
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=lr_decay_epochs, gamma=lr_decay_rate)
    scheduler_C = optim.lr_scheduler.StepLR(optimizer_C, step_size=lr_decay_epochs, gamma=lr_decay_rate)
    
    # Initialize curriculum learning scheduler
    curriculum = CurriculumLearningScheduler(
        start_epoch=curriculum_start,
        end_epoch=curriculum_end,
        ndvi_max_weight=curriculum_max_ndvi,
        schedule_type=curriculum_type
    )
    
    print(f"Using curriculum learning: {curriculum}")
    print(f"Using noise stabilization: initial={noise_factor}, decay={noise_decay}")
    
    # For context management with mixed precision
    class nullcontext:
        def __enter__(self): return None
        def __exit__(self, *excinfo): pass
    
    # Create visualization directory if needed
    if visualization_dir:
        os.makedirs(visualization_dir, exist_ok=True)
        os.makedirs(os.path.join(visualization_dir, "losses_run2"), exist_ok=True)
    
    # Load from checkpoint if provided
    if checkpoint is not None:
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_C.load_state_dict(checkpoint['optimizer_C_state_dict'])
        
        # Load scheduler states if available
        if 'scheduler_G_state_dict' in checkpoint:
            scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        if 'scheduler_C_state_dict' in checkpoint:
            scheduler_C.load_state_dict(checkpoint['scheduler_C_state_dict'])
            
        start_epoch = checkpoint['epoch']
        
        # Load history if available
        history = checkpoint.get('history', {
            'loss_G': [],
            'loss_C': [],
            'loss_G_adv': [],
            'loss_L1': [],
            'loss_fire': [],
            'loss_confidence': [],
            'gradient_penalty': [],
            'lr_G': [],
            'lr_C': [],
            'ndvi_weight': []  # Track NDVI influence
        })
    else:
        # Training history
        history = {
            'loss_G': [],
            'loss_C': [],
            'loss_G_adv': [],
            'loss_L1': [],
            'loss_fire': [],
            'loss_confidence': [],
            'gradient_penalty': [],
            'lr_G': [],
            'lr_C': [],
            'ndvi_weight': []
        }

    # Create directory for saving models
    os.makedirs(save_path, exist_ok=True)
    
    # Fixed validation sample for consistent visualization
    fixed_vis_sample = None
    fixed_vis_day = None
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        
        # Set models to training mode
        generator.train()
        critic.train()
        
        # Update NDVI weight based on curriculum schedule
        ndvi_weight = curriculum.get_ndvi_weight(epoch)
        print(f"Epoch {epoch+1}: NDVI weight = {ndvi_weight:.4f}")
        
        # Set NDVI weight in models
        generator.set_ndvi_weight(ndvi_weight)
        critic.set_ndvi_weight(ndvi_weight)
        
        # Progress bar for batches
        try:
            pbar = tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch+1}/{epochs}", total=len(train_dataloader))
            
            # Initialize epoch losses
            epoch_loss_G = 0
            epoch_loss_C = 0
            epoch_loss_G_adv = 0
            epoch_loss_L1 = 0
            epoch_loss_fire = 0
            epoch_loss_confidence = 0
            epoch_gp = 0
            valid_batches = 0
            
            for i, (ndvi, burn_history, curr_burn) in pbar:
                try:
                    # Move to device
                    ndvi = ndvi.to(device, non_blocking=True)
                    burn_history = burn_history.to(device, non_blocking=True)
                    curr_burn = curr_burn.to(device, non_blocking=True)
                    
                    # Print tensor shapes for the first batch of the first epoch
                    if epoch == start_epoch and i == 0:
                        print(f"NDVI tensor shape: {ndvi.shape}")
                        print(f"Burn history tensor shape: {burn_history.shape}")
                        print(f"Current burn tensor shape: {curr_burn.shape}")
                    
                    # Train critic multiple times per generator update
                    critic_loss = 0
                    gp_loss = 0
                    
                    # Generate initial prediction for critic training
                    with torch.no_grad():
                        # Create generator input
                        fire_probability = torch.sum(curr_burn > 0.5).float() / torch.numel(curr_burn)
                        fire_prob_channel = torch.ones_like(ndvi) * fire_probability
                        input_G = torch.cat((ndvi, burn_history, fire_prob_channel), dim=1)
                        
                        # Forward pass
                        with torch.cuda.amp.autocast() if use_amp else nullcontext():
                            gen_burn = generator(input_G)
                        
                        # Resize if needed
                        if gen_burn.shape != curr_burn.shape:
                            gen_burn = F.interpolate(gen_burn, size=curr_burn.shape[2:], mode='bilinear', align_corners=False)
                    
                    # Train critic multiple times
                    for _ in range(n_critic):
                        # Skip if generator output is None (due to error)
                        if gen_burn is None:
                            continue
                        
                        # Autocast for mixed precision
                        with torch.cuda.amp.autocast() if use_amp else nullcontext():
                            # Train critic with noise
                            c_loss, gp = train_wgan_critic_ndvi(
                                critic, ndvi, burn_history, curr_burn, gen_burn, 
                                optimizer_C, device, lambda_gp, 
                                epoch=epoch, noise_factor=noise_factor, noise_decay=noise_decay
                            )
                            
                        critic_loss += c_loss
                        gp_loss += gp
                    
                    # Average critic losses
                    critic_loss /= n_critic
                    gp_loss /= n_critic
                    
                    # Check for extreme gradient penalty values
                    if gp_loss > 100:
                        print(f"High gradient penalty: {gp_loss:.2f}. Reducing learning rate temporarily.")
                        for param_group in optimizer_C.param_groups:
                            param_group['lr'] *= 0.5
                    
                    # Train generator
                    # Skip if recent generator loss was too high (potential collapse)
                    if 'recent_g_loss' in locals() and recent_g_loss > 500:
                        print("Skipping generator update due to high recent loss.")
                    else:
                        # Autocast for mixed precision
                        with torch.cuda.amp.autocast() if use_amp else nullcontext():
                            g_loss, g_adv, l1, fire, conf, gen_burn = train_wgan_generator_ndvi(
                                generator, critic, ndvi, burn_history, curr_burn, optimizer_G,
                                lambda_L1, lambda_fire, fire_weight, alpha, gamma, lambda_confidence
                            )
                            
                        # Track recent generator loss for collapse detection
                        recent_g_loss = g_loss
                        
                        # Add batch losses to epoch totals if valid
                        if gen_burn is not None:
                            epoch_loss_G += g_loss
                            epoch_loss_G_adv += g_adv
                            epoch_loss_L1 += l1
                            epoch_loss_fire += fire
                            epoch_loss_confidence += conf
                            valid_batches += 1
                    
                    # Add critic losses to epoch totals
                    epoch_loss_C += critic_loss
                    epoch_gp += gp_loss
                    
                    # Update progress bar
                    current_lr_G = optimizer_G.param_groups[0]['lr']
                    current_lr_C = optimizer_C.param_groups[0]['lr']
                    pbar.set_postfix({
                        'loss_G': f"{g_loss:.4f}",
                        'loss_C': f"{critic_loss:.4f}",
                        'G_adv': f"{g_adv:.4f}",
                        'L1': f"{l1:.4f}",
                        'fire': f"{fire:.4f}",
                        'conf': f"{conf:.4f}",
                        'GP': f"{gp_loss:.4f}",
                        'lr_G': f"{current_lr_G:.6f}"
                    })
                    
                    # Visualize results at certain intervals
                    if visualization_dir and i % vis_every == 0 and gen_burn is not None:
                        with torch.no_grad():
                            # No need to rescale, already in [0,1] range
                            visualize_post_processing(
                                gen_burn[0], 
                                post_process_predictions(
                                    gen_burn[:1], threshold=0.8, min_area=0, max_gap=0)[0],
                                save_path=os.path.join(visualization_dir, f"post_process_epoch{epoch+1}_batch{i}.png")
                            )
                    
                    # Clean up to reduce memory usage
                    if i % 10 == 0:
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                except Exception as e:
                    print(f"Error processing batch {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Calculate average epoch losses
            if valid_batches > 0:
                epoch_loss_G /= valid_batches
                epoch_loss_G_adv /= valid_batches
                epoch_loss_L1 /= valid_batches
                epoch_loss_fire /= valid_batches
                epoch_loss_confidence /= valid_batches
            
            # Critic losses might have different count
            epoch_loss_C /= len(train_dataloader)
            epoch_gp /= len(train_dataloader)
            
            # Save losses to history
            history['loss_G'].append(epoch_loss_G)
            history['loss_C'].append(epoch_loss_C)
            history['loss_G_adv'].append(epoch_loss_G_adv)
            history['loss_L1'].append(epoch_loss_L1)
            history['loss_fire'].append(epoch_loss_fire)
            history['loss_confidence'].append(epoch_loss_confidence)
            history['gradient_penalty'].append(epoch_gp)
            history['lr_G'].append(optimizer_G.param_groups[0]['lr'])
            history['lr_C'].append(optimizer_C.param_groups[0]['lr'])
            history['ndvi_weight'].append(ndvi_weight)
            
            # Print epoch stats with timing
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch [{epoch+1}/{epochs}] Time: {epoch_time:.1f}s Loss_G: {epoch_loss_G:.4f} Loss_C: {epoch_loss_C:.4f} "
                  f"Loss_G_adv: {epoch_loss_G_adv:.4f} Loss_L1: {epoch_loss_L1:.4f} "
                  f"Loss_Fire: {epoch_loss_fire:.4f} Loss_Conf: {epoch_loss_confidence:.4f} GP: {epoch_gp:.4f} "
                  f"LR_G: {optimizer_G.param_groups[0]['lr']:.6f} LR_C: {optimizer_C.param_groups[0]['lr']:.6f}")
            
            # Step the learning rate schedulers
            scheduler_G.step()
            scheduler_C.step()
            
            # Run validation if requested
            if val_dataloader is not None and (epoch + 1) % validate_every == 0:
                print(f"Running validation after epoch {epoch+1}...")
                generator.eval()  # Set to evaluation mode
                critic.eval()
                
                with torch.no_grad():
                    val_loss_G = 0
                    val_loss_G_adv = 0
                    val_loss_L1 = 0
                    val_loss_fire = 0
                    val_loss_confidence = 0
                    val_batches = 0
                    
                    for val_idx, (val_ndvi, val_burn_history, val_curr_burn) in enumerate(val_dataloader):
                        try:
                            # Move to device
                            val_ndvi = val_ndvi.to(device, non_blocking=True)
                            val_burn_history = val_burn_history.to(device, non_blocking=True)
                            val_curr_burn = val_curr_burn.to(device, non_blocking=True)
                            
                            # Save fixed validation sample for consistency
                            if fixed_vis_sample is None and val_idx == 0:
                                fixed_vis_sample = (
                                    val_ndvi[0:1].clone(), 
                                    val_burn_history[0:1].clone(),
                                    val_curr_burn[0:1].clone()
                                )
                                # Try to get the day info if dataset has it
                                try:
                                    if hasattr(val_dataloader.dataset, 'valid_sequences'):
                                        fixed_vis_day = os.path.basename(val_dataloader.dataset.valid_sequences[0][2])
                                        print(f"Using fixed validation sample from day: {fixed_vis_day}")
                                except:
                                    fixed_vis_day = f"sample_0"
                            
                            # Create generator input
                            val_fire_probability = torch.sum(val_curr_burn > 0.5).float() / torch.numel(val_curr_burn)
                            val_fire_prob_channel = torch.ones_like(val_ndvi) * val_fire_probability
                            val_input_G = torch.cat((val_ndvi, val_burn_history, val_fire_prob_channel), dim=1)
                            
                            # Generate prediction
                            val_gen_burn = generator(val_input_G)
                            
                            # Ensure dimensions match
                            if val_gen_burn.shape != val_curr_burn.shape:
                                val_gen_burn = F.interpolate(val_gen_burn, size=val_curr_burn.shape[2:], 
                                                           mode='bilinear', align_corners=False)
                            
                            # Calculate losses
                            val_input_fake = torch.cat((val_ndvi, val_burn_history, val_gen_burn), dim=1)
                            val_critic_fake = critic(val_input_fake)
                            val_loss_G_adv = -torch.mean(val_critic_fake)
                            
                            # L1 loss with fire weighting
                            # Note: Both are in [0,1] range, no conversion needed
                            val_weights = torch.ones_like(val_curr_burn)
                            val_weights[val_curr_burn > 0.5] = fire_weight
                            val_loss_L1 = torch.mean(val_weights * torch.abs(val_gen_burn - val_curr_burn))
                            
                            # Focal Loss
                            val_focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')
                            
                            # Calculate class weights
                            val_num_fire_pixels = torch.sum(val_curr_burn > 0.5).float()
                            val_num_nonfire_pixels = torch.numel(val_curr_burn) - val_num_fire_pixels
                            
                            if val_num_fire_pixels > 0:
                                val_fire_to_nonfire_ratio = val_num_nonfire_pixels / (val_num_fire_pixels + 1e-8)
                                val_dynamic_weights = torch.ones_like(val_curr_burn)
                                val_dynamic_weights[val_curr_burn > 0.5] = val_fire_to_nonfire_ratio
                                val_dynamic_weights[val_curr_burn > 0.5] *= fire_weight / 100.0
                                
                                val_focal_custom = FocalLoss(alpha=alpha, gamma=gamma, reduction='none')
                                val_fire_loss_raw = val_focal_custom(val_gen_burn, val_curr_burn, 
                                                                   weights=val_dynamic_weights)
                                val_loss_fire = val_fire_loss_raw.mean()
                            else:
                                val_loss_fire = val_focal_loss(val_gen_burn, val_curr_burn)
                            
                            # Confidence loss
                            val_loss_confidence = -torch.mean(torch.abs(val_gen_burn - 0.5)) + 0.5
                            
                            # Total loss
                            val_loss_G = (val_loss_G_adv + lambda_L1 * val_loss_L1 + 
                                        lambda_fire * val_loss_fire + 
                                        lambda_confidence * val_loss_confidence)
                            
                            # Add to validation totals
                            val_loss_G += val_loss_G.item()
                            val_loss_G_adv += val_loss_G_adv.item()
                            val_loss_L1 += val_loss_L1.item()
                            val_loss_fire += val_loss_fire.item()
                            val_loss_confidence += val_loss_confidence.item()
                            val_batches += 1
                            
                            # Visualize validation results for the first few batches
                            if visualization_dir and val_idx < 3:
                                visualize_validation(
                                    val_gen_burn[0], 
                                    val_curr_burn[0],
                                    save_path=os.path.join(visualization_dir, f"val_epoch{epoch+1}_batch{val_idx}.png")
                                )
                            
                        except Exception as e:
                            print(f"Error in validation batch {val_idx}: {e}")
                            continue
                    
                    # Calculate average validation losses
                    if val_batches > 0:
                        val_loss_G /= val_batches
                        val_loss_G_adv /= val_batches
                        val_loss_L1 /= val_batches
                        val_loss_fire /= val_batches
                        val_loss_confidence /= val_batches
                        
                        # Log validation metrics
                        print(f"Validation: Loss_G: {val_loss_G:.4f} G_adv: {val_loss_G_adv:.4f} "
                              f"L1: {val_loss_L1:.4f} Fire: {val_loss_fire:.4f} Conf: {val_loss_confidence:.4f}")
                        
                        # Add to history
                        if 'val_loss_G' not in history:
                            history['val_loss_G'] = []
                            history['val_loss_G_adv'] = []
                            history['val_loss_L1'] = []
                            history['val_loss_fire'] = []
                            history['val_loss_confidence'] = []
                        
                        history['val_loss_G'].append(val_loss_G)
                        history['val_loss_G_adv'].append(val_loss_G_adv)
                        history['val_loss_L1'].append(val_loss_L1)
                        history['val_loss_fire'].append(val_loss_fire)
                        history['val_loss_confidence'].append(val_loss_confidence)
                    
                    # Create fixed sample visualization to track progress on same example
                    if fixed_vis_sample is not None and visualization_dir:
                        generator.eval()
                        val_ndvi, val_burn_history, val_curr_burn = fixed_vis_sample
                        
                        # Create generator input
                        val_fire_probability = torch.sum(val_curr_burn > 0.5).float() / torch.numel(val_curr_burn)
                        val_fire_prob_channel = torch.ones_like(val_ndvi) * val_fire_probability
                        val_input_G = torch.cat((val_ndvi, val_burn_history, val_fire_prob_channel), dim=1)
                        
                        # Generate prediction
                        val_gen_burn = generator(val_input_G)
                        
                        # Save fixed sample visualization
                        visualize_validation(
                            val_gen_burn[0],
                            val_curr_burn[0],
                            save_path=os.path.join(visualization_dir, f"fixed_sample_epoch{epoch+1}.png"),
                            title=f"Fixed Sample ({fixed_vis_day}) - Epoch {epoch+1}"
                        )
                
                # Set back to training mode
                generator.train()
                critic.train()
                
            # Visualize training losses and NDVI weight
            if visualization_dir and ((epoch + 1) % loss_plot_every == 0 or epoch == epochs - 1):
                print(f"Generating loss visualization at epoch {epoch+1}...")
                try:
                    visualize_training_losses(
                        history,
                        save_path=os.path.join(visualization_dir, "losses_run2", f"training_losses_epoch_{epoch+1}.png"),
                        figure_size=(14, 10)
                    )
                    
                    # Also visualize NDVI weight progression
                    plt.figure(figsize=(10, 5))
                    plt.plot(range(1, len(history['ndvi_weight']) + 1), history['ndvi_weight'], 'b-')
                    plt.xlabel('Epoch')
                    plt.ylabel('NDVI Weight')
                    plt.title('NDVI Influence Over Training (Curriculum Learning)')
                    plt.grid(True, alpha=0.3)
                    plt.savefig(os.path.join(visualization_dir, "losses", f"ndvi_weight_epoch_{epoch+1}.png"), dpi=150)
                    plt.close()
                except Exception as e:
                    print(f"Error generating visualizations: {e}")
            
            # Save models at specified intervals or at the end of training
            if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
                # Save all states in a single checkpoint file
                checkpoint = {
                    'epoch': epoch + 1,
                    'generator_state_dict': generator.state_dict(),
                    'critic_state_dict': critic.state_dict(),
                    'optimizer_G_state_dict': optimizer_G.state_dict(),
                    'optimizer_C_state_dict': optimizer_C.state_dict(),
                    'scheduler_G_state_dict': scheduler_G.state_dict(),
                    'scheduler_C_state_dict': scheduler_C.state_dict(),
                    'loss_G': epoch_loss_G,
                    'loss_C': epoch_loss_C,
                    'loss_G_adv': epoch_loss_G_adv,
                    'loss_L1': epoch_loss_L1,
                    'loss_fire': epoch_loss_fire,
                    'loss_confidence': epoch_loss_confidence,
                    'gradient_penalty': epoch_gp,
                    'ndvi_weight': ndvi_weight,
                    'curriculum_scheduler': {
                        'start_epoch': curriculum_start,
                        'end_epoch': curriculum_end,
                        'max_ndvi': curriculum_max_ndvi,
                        'schedule_type': curriculum_type
                    },
                    'history': history
                }
                torch.save(checkpoint, f"{save_path}/wgan_ndvi_improved_checkpoint_epoch_{epoch+1}.pt")
                print(f"Saved WGAN-NDVI improved checkpoint at epoch {epoch+1}")
            
        except Exception as e:
            print(f"Error during epoch {epoch+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save final models
    checkpoint = {
        'epoch': epochs,
        'generator_state_dict': generator.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_C_state_dict': optimizer_C.state_dict(),
        'scheduler_G_state_dict': scheduler_G.state_dict(),
        'scheduler_C_state_dict': scheduler_C.state_dict(),
        'history': history
    }
    torch.save(checkpoint, f"{save_path}/wgan_ndvi_improved_checkpoint_final.pt")
    
    # Also save individual model files
    torch.save(generator.state_dict(), f"{save_path}/wgan_ndvi_improved_generator_final.pth")
    torch.save(critic.state_dict(), f"{save_path}/wgan_ndvi_improved_critic_final.pth")
    
    return generator, critic, history