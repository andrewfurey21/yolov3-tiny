
"""

1. torch.fx
2. torch.compile - Dynamo (frontend) and Inductor (backend)
3. torch.jit, torchscript -> turn python models into a static, serializable and optimizable representation. can be run outside of cpython.

torch.jit.trace vs script vs symbolic trace, torchscript, 

torch.fx: make it easier to create, inspect, analyze, optimize, transform pytorch nn.Modules.
nn.Module + symbolic_trace -> GraphModule


torchdyanmo eval frame
1. check if frame should be skipped due to: filename exclusion, previous failures to compile, or cache limit exceeded.
(skipped files are like standard library calls that make no use of pytorch)
2. Check if previously commpiled. execute compiled function if so (obviously you don't want to recompile stuff thats already compiled!)

"""

import torch
from yolov3tiny.model import YOLOLayer

layer = YOLOLayer(85, [(81, 82)], 416)

input = torch.linspace(0, 20, 85 * 1 * 4 * 4).reshape(1, 1 * 85, 4, 4)
print(input)
output = layer(input)
print(output.shape)
print(output[0, 0])
