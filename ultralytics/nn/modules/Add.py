import torch
import torch.nn as nn

class Add(nn.Module):
    """Add layer for ControlNet structure that combines original and control features."""
    def __init__(self):
        """Initialize Add module."""
        super().__init__()
        # 为control分支创建可学习的缩放因子，初始化为0
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Add original features with scaled control features.
        
        Args:
            x (tuple): Tuple containing (original_features, control_features)
                original_features: Features from locked original model path
                control_features: Features from trainable control path
        
        Returns:
            (torch.Tensor): Combined features
        """
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("Add layer expects two inputs in a list/tuple")
        
        original, control = x
        if original.shape != control.shape:
            raise ValueError(f"Shape mismatch in Add layer: {original.shape} vs {control.shape}")
        
        # ControlNet style feature combination:
        # 1. 保持原始特征不变
        # 2. 控制特征通过可学习的scale因子进行缩放
        # 3. 将两者相加
        return original + self.scale * control

    def __repr__(self):
        """String representation of module."""
        return f'{self.__class__.__name__}(scale={self.scale.item():.3f})'
