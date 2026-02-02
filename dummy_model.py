"""
Toy MLP: input(2) -> hidden(10) -> output(5)
Exports to dummy_model.onnx
"""
import torch
import torch.nn as nn


class DummyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def main():
    model = DummyMLP()
    model.eval()

    # Example input shape: (batch_size, input_size)
    dummy_input = torch.randn(1, 2)

    torch.onnx.export(
        model,
        dummy_input,
        "dummy_model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print("Exported to dummy_model.onnx")


if __name__ == "__main__":
    main()

