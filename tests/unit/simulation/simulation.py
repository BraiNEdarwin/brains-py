import torch
from torch import nn
import numpy as np


import matplotlib.pyplot as plt
from brainspy.utils.pytorch import TorchUtils

from brainspy.processors.dnpu import DNPU


configs = {}
configs["platform"] = "simulation"
configs[
    "torch_model_dict"
] = "/home/unai/Documents/3-programming/brainspy-processors/tmp/input/models/test.pt"
configs["input_indices"] = [0, 1]
configs["input_electrode_no"] = 7
configs["waveform"] = {}
configs["waveform"]["plateau_lengths"] = 80
configs["waveform"]["slope_lengths"] = 20
# configs['noise'] = {}
# configs['noise']['type'] = 'gaussian'
# configs['noise']['mse'] = 1.97


x = 0.5 * np.random.randn(10, 2)
x = TorchUtils.format(x)
target = TorchUtils.format([[5]] * 10)
node = DNPU(configs)
loss = nn.MSELoss()
optimizer = torch.optim.Adam([{"params": node.parameters()}], lr=0.01)

LOSS_LIST = []
CHANGE_PARAMS_NET = []
CHANGE_PARAMS0 = []

START_PARAMS = [p.clone().detach() for p in node.parameters()]

for eps in range(10000):

    optimizer.zero_grad()
    out = node(x)
    if np.isnan(out.data.cpu().numpy()[0]):
        break
    LOSS = loss(out, target) + node.regularizer()
    LOSS.backward()
    optimizer.step()
    LOSS_LIST.append(LOSS.data.cpu().numpy())
    CURRENT_PARAMS = [p.clone().detach() for p in node.parameters()]
    DELTA_PARAMS = [
        (current - start).sum() for current, start in zip(CURRENT_PARAMS, START_PARAMS)
    ]
    CHANGE_PARAMS0.append(DELTA_PARAMS[0])
    CHANGE_PARAMS_NET.append(sum(DELTA_PARAMS[1:]))

END_PARAMS = [p.clone().detach() for p in node.parameters()]
print("CV params at the beginning: \n ", START_PARAMS[0])
print("CV params at the end: \n", END_PARAMS[0])
print("Example params at the beginning: \n", START_PARAMS[-1][:8])
print("Example params at the end: \n", END_PARAMS[-1][:8])
print("Length of elements in node.parameters(): \n", [len(p) for p in END_PARAMS])
print("and their shape: \n", [p.shape for p in END_PARAMS])
print(f"OUTPUT: \n {out.data.cpu()}")

plt.figure()
plt.plot(LOSS_LIST)
plt.title("Loss per epoch")
plt.show()
plt.figure()
plt.plot(CHANGE_PARAMS0)
plt.plot(CHANGE_PARAMS_NET)
plt.title("Difference of parameter with initial params")
plt.legend(["CV params", "Net params"])
plt.show()
