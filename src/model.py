import torch
from torch import nn
from torchvision.models import (
    EfficientNet_V2_S_Weights,
    efficientnet_v2_s,
)


class FruitRipenessModel(nn.Module):
    """
    EfficientNetV2-S based fruit recognition + ripeness model.

    Fruit classes:
        0 -> banana
        1 -> mango
        2 -> no_fruit

    Ripeness classes:
        0 -> unripe
        1 -> ripe
        2 -> overripe
    """

    def __init__(
        self,
        num_ripeness_classes: int,
        num_fruit_classes: int = 3,
        pretrained: bool = True,
    ):
        super().__init__()

        if num_ripeness_classes < 1:
            raise ValueError(
                "num_ripeness_classes must be positive"
            )

        if num_fruit_classes < 1:
            raise ValueError(
                "num_fruit_classes must be positive"
            )

        # --------------------------------------------------------
        # EfficientNetV2-S
        # --------------------------------------------------------

        weights = (
            EfficientNet_V2_S_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = efficientnet_v2_s(
            weights=weights
        )

        self.features = backbone.features
        self.pool = backbone.avgpool

        feature_dim = (
            backbone.classifier[1].in_features
        )

        # --------------------------------------------------------
        # FRUIT CLASSIFICATION HEAD
        #
        # banana
        # mango
        # no_fruit
        # --------------------------------------------------------

        self.fruit_head = nn.Linear(
            feature_dim,
            num_fruit_classes,
        )

        # --------------------------------------------------------
        # RIPENESS CLASSIFICATION HEAD
        #
        # unripe
        # ripe
        # overripe
        # --------------------------------------------------------

        self.ripeness_head = nn.Linear(
            feature_dim,
            num_ripeness_classes,
        )

        # --------------------------------------------------------
        # SENSOR BRANCH
        #
        # Temperature + Humidity
        #
        # This is kept for the later integrated model.
        # It is NOT required for the current image-only training.
        # --------------------------------------------------------

        self.sensor_branch = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # --------------------------------------------------------
        # REGRESSION HEAD
        #
        # Image features + sensor features
        #
        # Used later for Days Remaining prediction.
        # --------------------------------------------------------

        self.regression_head = nn.Sequential(
            nn.Linear(
                feature_dim + 32,
                128,
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                128,
                1,
            ),
        )

    # ============================================================
    # IMAGE FEATURE EXTRACTION
    # ============================================================

    def encode_image(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert an image into EfficientNet feature vectors.

        Input:
            [batch, 3, 384, 384]

        Output:
            [batch, feature_dim]
        """

        image_features = self.features(
            image
        )

        pooled_features = self.pool(
            image_features
        )

        return pooled_features.flatten(1)

    # ============================================================
    # FORWARD
    # ============================================================

    def forward(
        self,
        image: torch.Tensor,
        temperature: torch.Tensor,
        humidity: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        # --------------------------------------------------------
        # IMAGE FEATURES
        # --------------------------------------------------------

        image_features = self.encode_image(
            image
        )

        # --------------------------------------------------------
        # SENSOR FEATURES
        # --------------------------------------------------------

        sensor_input = torch.stack(
            (
                temperature,
                humidity,
            ),
            dim=1,
        ).float()

        sensor_features = self.sensor_branch(
            sensor_input
        )

        # --------------------------------------------------------
        # FUSION
        # --------------------------------------------------------

        fused_features = torch.cat(
            (
                image_features,
                sensor_features,
            ),
            dim=1,
        )

        # --------------------------------------------------------
        # OUTPUTS
        # --------------------------------------------------------

        fruit_logits = self.fruit_head(
            image_features
        )

        ripeness_logits = self.ripeness_head(
            image_features
        )

        days_remaining = self.regression_head(
            fused_features
        ).squeeze(1)

        return {
            "fruit_logits": fruit_logits,
            "ripeness_logits": ripeness_logits,
            "days_remaining": days_remaining,
        }

    # ============================================================
    # BACKBONE FREEZING
    # ============================================================

    def freeze_backbone(
        self,
        frozen: bool = True,
    ) -> None:
        """
        Freeze/unfreeze EfficientNet feature extractor.
        """

        for parameter in self.features.parameters():

            parameter.requires_grad = not frozen