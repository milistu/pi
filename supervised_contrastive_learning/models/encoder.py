import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    def __init__(
        self,
        projection_layer: int = 128,
        training: bool = False,
        weights: models.Weights = None,
    ) -> None:
        super(Encoder, self).__init__()

        self.training = training
        self.backbone = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*(list(self.backbone.children())[:-1]))
        #  multi-layer perceptron
        self.mlp = nn.Linear(2048, projection_layer)

    def forward(self, x) -> torch.Tensor:
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1)
        if self.training:
            x = self.mlp(x)
        return x


if __name__ == "__main__":
    encoder = Encoder()

    input_tensor = torch.randn(2, 3, 28, 28)

    print(encoder(input_tensor))
