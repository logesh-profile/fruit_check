import torch
from torch import nn
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s


class FruitRipenessModel(nn.Module):
    def __init__(self, num_ripeness_classes: int, pretrained: bool = True):
        super().__init__()
        if num_ripeness_classes < 1:
            raise ValueError("num_ripeness_classes must be positive")
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_v2_s(weights=weights)
        self.features = backbone.features
        self.pool = backbone.avgpool
        feature_dim = backbone.classifier[1].in_features
        self.fruit_head = nn.Linear(feature_dim, 2)
        self.ripeness_head = nn.Linear(feature_dim, num_ripeness_classes)
        self.sensor_branch = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.regression_head = nn.Sequential(nn.Linear(feature_dim + 32, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(image)).flatten(1)

    def forward(self, image: torch.Tensor, temperature: torch.Tensor, humidity: torch.Tensor) -> dict[str, torch.Tensor]:
        image_features = self.encode_image(image)
        sensor_features = self.sensor_branch(torch.stack((temperature, humidity), dim=1).float())
        fused = torch.cat((image_features, sensor_features), dim=1)
        return {"fruit_logits": self.fruit_head(image_features), "ripeness_logits": self.ripeness_head(image_features), "days_remaining": self.regression_head(fused).squeeze(1)}

    def freeze_backbone(self, frozen: bool = True) -> None:
        for parameter in self.features.parameters():
            parameter.requires_grad = not frozen