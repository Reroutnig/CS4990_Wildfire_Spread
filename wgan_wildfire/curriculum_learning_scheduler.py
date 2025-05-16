import numpy as np

class CurriculumLearningScheduler:
    """
    Scheduler for curriculum learning approach to gradually introduce NDVI data
    
    Args:
        start_epoch: Epoch to start introducing NDVI
        end_epoch: Epoch where NDVI reaches full influence
        ndvi_max_weight: Maximum weight for NDVI when fully introduced (1.0 = full influence)
        schedule_type: Type of schedule (linear, cosine)
    """
    def __init__(self, start_epoch=20, end_epoch=60, ndvi_max_weight=1.0, schedule_type='linear'):
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.ndvi_max_weight = ndvi_max_weight
        self.schedule_type = schedule_type
        
    def get_ndvi_weight(self, epoch):
        """
        Calculate NDVI weight based on current epoch
        
        Args:
            epoch: Current epoch
            
        Returns:
            float: NDVI weight between 0.0 and ndvi_max_weight
        """
        # Before start epoch, NDVI has no influence
        if epoch < self.start_epoch:
            return 0.0
        
        # After end epoch, NDVI has full influence (up to max weight)
        if epoch >= self.end_epoch:
            return self.ndvi_max_weight
        
        # Calculate progress between start and end epoch
        progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
        
        # # Apply different schedule types
        if self.schedule_type == 'linear':
            weight = progress * self.ndvi_max_weight
        
        elif self.schedule_type == 'cosine':
            # Cosine learning schedule (slow start, slow end)
            weight = (1 - np.cos(progress * np.pi)) / 2 * self.ndvi_max_weight
            
        else:
            # Default to linear
            weight = progress * self.ndvi_max_weight
            
        return weight
    
    def __str__(self):
        return (f"CurriculumLearningScheduler(start={self.start_epoch}, end={self.end_epoch}, "
                f"max_weight={self.ndvi_max_weight}, type={self.schedule_type})")