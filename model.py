# model definition
import torch.nn as nn
import torchvision.models as models
import timm

# randon init / ImageNet pretrain
class ResNet18(nn.Module):
    def __init__(self, pretrained=True, out=37, dropout=0.5):
        super(ResNet18, self).__init__()

        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.model = models.resnet18(weights=weights)
        else:
            self.model = models.resnet18(weights=None)

        # fixed FC for 37 classes
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.model.fc.in_features, out)
        )

    def forward(self, x):
        return self.model(x)
    
    

class ViTTiny(nn.Module):
    def __init__(self, pretrained=True, out=37, dropout=0.5):
        super(ViTTiny, self).__init__()

        self.model = timm.create_model(
            'vit_tiny_patch16_224', 
            pretrained=pretrained,
            drop_rate=dropout
            )
        self.model.head = nn.Linear(self.model.head.in_features, out)

    def forward(self, x):
        return self.model(x)