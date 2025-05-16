import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import post_process_predictions

# ------------------------------
# Generator (U-Net with skip connections)
# ------------------------------
class Generator(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, ndvi_channels=1):
        super(Generator, self).__init__()
        self.ndvi_channels = ndvi_channels
        self.history_channels = in_channels - ndvi_channels - 1  # -1 for fire prob channel
        
        # Create separate NDVI encoder branch
        self.ndvi_encoder = nn.Sequential(
            nn.Conv2d(ndvi_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # History encoder branch
        self.history_encoder = nn.Sequential(
            nn.Conv2d(self.history_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Fire probability encoder (global information)
        self.fire_prob_encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=1, stride=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Encoder (u-net style) - takes combined features
        self.enc1 = self.contract_block(32 + 32 + 8, 64)
        self.enc2 = self.contract_block(64, 128)
        self.enc3 = self.contract_block(128, 256)
        self.enc4 = self.contract_block(256, 512)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # Decoder with Dropout
        self.dec1 = self.expand_block(1024, 512, dropout=True)
        self.dec2 = self.expand_block(1024, 256, dropout=True)
        self.dec3 = self.expand_block(512, 128)
        self.dec4 = self.expand_block(256, 64)

        # Final layer with Sigmoid activation - directly output [0,1] range
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Added sigmoid activation for [0,1] output
        )
        
        # Flag for applying post-processing
        self.apply_post_processing = False
        self.post_process_params = {
            'threshold': 0.5,
            'min_area': 5,
            'max_gap': 3
        }
        
        # For curriculum learning
        self.ndvi_weight = 1.0

    def contract_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def expand_block(self, in_channels, out_channels, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)
        
    def set_post_processing(self, enabled=True, threshold=0.8, min_area=0, max_gap=0):
        self.apply_post_processing = enabled
        self.post_process_params = {
            'threshold': threshold,
            'min_area': min_area,
            'max_gap': max_gap
        }
    
    def set_ndvi_weight(self, weight):
        self.ndvi_weight = weight
        
    def forward(self, x):
        # Store original input dimensions
        input_shape = x.shape
        
        # Split the input tensor into components
        ndvi = x[:, :self.ndvi_channels]
        fire_prob = x[:, -1:] 
        history = x[:, self.ndvi_channels:self.ndvi_channels+self.history_channels]
        
        # Apply curriculum learning weight to NDVI
        if self.ndvi_weight < 1.0:
            zero_ndvi = torch.zeros_like(ndvi)
            ndvi = ndvi * self.ndvi_weight + zero_ndvi * (1.0 - self.ndvi_weight)
        
        # Process branches separately
        ndvi_features = self.ndvi_encoder(ndvi)
        history_features = self.history_encoder(history)
        fire_prob_features = self.fire_prob_encoder(fire_prob)
        
        # Combine features
        combined_features = torch.cat([ndvi_features, history_features, fire_prob_features], dim=1)
        
        # Encoder
        e1 = self.enc1(combined_features)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Bottleneck
        b = self.bottleneck(e4)
        
        # Decoder with skip connections
        d1 = self.dec1(b)
        if d1.shape[2:] != e4.shape[2:]:
            d1 = F.interpolate(d1, size=e4.shape[2:], mode='bilinear', align_corners=False)
        
        d2 = self.dec2(torch.cat([d1, e4], dim=1))
        if d2.shape[2:] != e3.shape[2:]:
            d2 = F.interpolate(d2, size=e3.shape[2:], mode='bilinear', align_corners=False)
        
        d3 = self.dec3(torch.cat([d2, e3], dim=1))
        if d3.shape[2:] != e2.shape[2:]:
            d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        
        d4 = self.dec4(torch.cat([d3, e2], dim=1))
        if d4.shape[2:] != e1.shape[2:]:
            d4 = F.interpolate(d4, size=e1.shape[2:], mode='bilinear', align_corners=False)
        
        # Final layer with sigmoid activation
        out = self.final(torch.cat([d4, e1], dim=1))
        
        # Ensure output has same spatial dimensions as input
        if out.shape[2:] != input_shape[2:]:
            out = F.interpolate(out, size=input_shape[2:], mode='bilinear', align_corners=False)
        
        # Apply post-processing during evaluation (not during training)
        # Note: No need to rescale since output is already in [0,1]
        if not self.training and self.apply_post_processing:
            processed = post_process_predictions(
                out, 
                threshold=self.post_process_params['threshold'],
                min_area=self.post_process_params['min_area'],
                max_gap=self.post_process_params['max_gap']
            )
            out = processed
            
        return out

# ------------------------------
# Critic (WGAN Discriminator)
# ------------------------------
class Critic(nn.Module):
    def __init__(self, ndvi_channels=1, history_channels=3, output_channels=1):
        super(Critic, self).__init__()
        self.ndvi_channels = ndvi_channels
        self.history_channels = history_channels
        self.output_channels = output_channels
        
        # Total input channels: NDVI + burn history + current/generated burn
        in_channels = ndvi_channels + history_channels + output_channels
        
        # Using Layer Normalization instead of BatchNorm
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(64, 256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(128, 512), 
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Global average pooling followed by a linear layer to output a single scalar
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.final = nn.Linear(512, 1)
        
        # For curriculum learning
        self.ndvi_weight = 1.0

    def set_ndvi_weight(self, weight):
        self.ndvi_weight = weight

    def forward(self, x):
        # Split input tensor into components
        ndvi = x[:, :self.ndvi_channels]
        burn_data = x[:, self.ndvi_channels:]
        
        # Apply curriculum weight to NDVI
        if self.ndvi_weight < 1.0:
            zero_ndvi = torch.zeros_like(ndvi)
            ndvi = ndvi * self.ndvi_weight + zero_ndvi * (1.0 - self.ndvi_weight)
            
        # Recombine components
        x = torch.cat([ndvi, burn_data], dim=1)
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Global pooling to get a single vector per sample
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Output a single scalar per sample
        return self.final(pooled)