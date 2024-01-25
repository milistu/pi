import torch
import torch.nn as nn
from torch.nn import functional as F
from encoder import Encoder


class Classifier(nn.Module):
    def __init__(
        self, encoder: Encoder, num_classes: int = 10, freeze_encoder: bool = True
    ) -> None:
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, num_classes)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x) -> torch.Tensor:
        x = self.encoder(x)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        return x


if __name__ == "__main__":
    encoder = Encoder()
    classifier = Classifier(encoder=encoder)

    input_tensor = torch.randn(10, 3, 28, 28)
    output = classifier(input_tensor)

    assert output.shape == torch.Size([10, 10]), "Error with output shape."
    print("We are all set!")
