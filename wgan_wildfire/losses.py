import torch
import torch.nn as nn

# ------------------------------
# Focal Loss Implementation
# ------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', from_logits=False):
        """
        Focal Loss for binary classification with extreme class imbalance
        
        Args:
            alpha: Weight factor for the positive class (usually the minority class)
            gamma: Focusing parameter to down-weight easy examples
            reduction: 'mean', 'sum' or 'none'
            from_logits: Whether inputs are in logit space (True) or probability space (False)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.from_logits = from_logits
        
        # Use BCE with logits only if inputs are in logit space
        if from_logits:
            self.bce = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.bce = nn.BCELoss(reduction='none')
        
    def forward(self, inputs, targets, weights=None):
        """
        Calculate focal loss with optional pixel-wise weights
        
        Args:
            inputs: Model predictions (either logits or probabilities)
            targets: Ground truth labels (0 or 1)
            weights: Optional pixel-wise weights 
            
        Returns:
            Calculated loss
        """
        # Get probabilities
        if self.from_logits:
            # Convert from logits to probabilities if necessary
            probs = torch.sigmoid(inputs)
        else:
            # Inputs are already probabilities
            probs = inputs
            
        # Binary cross entropy loss
        bce_loss = self.bce(inputs, targets)
        
        # For positive examples (targets=1), use pt = p
        # For negative examples (targets=0), use pt = 1-p
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # Apply alpha weighting
        # For positive examples, use alpha
        # For negative examples, use 1-alpha
        alpha_weight = torch.where(targets == 1, 
                                 self.alpha * torch.ones_like(targets),
                                 (1 - self.alpha) * torch.ones_like(targets))
        
        # Calculate focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Combine weighting factors
        loss = alpha_weight * focal_weight * bce_loss
        
        # Apply additional pixel-wise weights if provided
        if weights is not None:
            loss = loss * weights
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss