import torch

import onnx
import torch.fx as fx

from onnxsim import simplify
import onnxoptimizer
from onnx.shape_inference import infer_shapes

class CustomModel(torch.nn.Module):
    """
    A pytorch implementation of Logistic Regression.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        return x[:,:12,:]


model = CustomModel()
sample = torch.zeros((1024,22,32),dtype=torch.int64)
torch.onnx.export(model,sample,f'/home/yssun/pytorch-fm/models/custom.onnx')