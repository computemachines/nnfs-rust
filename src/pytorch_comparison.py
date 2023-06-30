import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# X = torch.from_numpy(np.linspace(0, 1, 100).reshape(-1, 1)).float().to(device)
# y = torch.from_numpy(np.sin(2 * np.pi * X.cpu().numpy())).float().to(device)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 5),
            nn.ReLU(),
            nn.Linear(5, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

    def save(self, path):
        weights = []
        biases = []
        for key, value in model.state_dict().items():
            if "weight" in key:
                weights.append(value.cpu().numpy().tolist())
            elif "bias" in key:
                biases.append(value.cpu().numpy().tolist())

        pickle.dump(weights, open(f"{path}/model-weights.pkl", "wb"))
        pickle.dump(biases, open(f"{path}/model-biases.pkl", "wb"))


class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def save(self, path):
        weights = []
        biases = []
        for key, value in model.state_dict().items():
            if "weight" in key:
                weights.append(value.cpu().numpy().tolist())
            elif "bias" in key:
                biases.append(value.cpu().numpy().tolist())

        pickle.dump(weights, open(f"{path}/model-weights.pkl", "wb"))
        pickle.dump(biases, open(f"{path}/model-biases.pkl", "wb"))

    def forward(self, x):
        output = self.linear(x)
        output.register_hook(lambda grad: print(grad))
        z = self.sigmoid(output)
        return z


binary_inputs = torch.tensor(
    [[0, 0], [0, 1], [1, 0], [1, 1]]).float().to(device)
or_outputs = torch.tensor([[0], [1], [1], [1]]).float().to(device)

# model = NeuralNetwork().to(device)
model = Perceptron().to(device)
model = torch.load("model.pth")

# loss_fn = nn.MSELoss()
loss_fn = nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=(1-1e-3))


def train_loop(X, y, model, loss_fn, optimizer, scheduler=None):
    outputs = model(X)
    loss = loss_fn(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # scheduler.step()

    return loss.item()


# for epoch in range(1):
#     loss = train_loop(binary_inputs, or_outputs, model, loss_fn, optimizer)
#     print(f"Epoch: {epoch}, Loss: {loss}")



input = torch.tensor([[0.5, 0.5]]).float().to(device)
test_out = torch.tensor([[0.]]).float().to(device)
original_output = model(input)
original_loss = loss_fn(model(input), test_out)
original_loss.retain_grad()
# model.sigmoid.retain_grad()
original_loss.backward()
dw11 = 1e-3
with torch.no_grad():
    model.linear.weight[0][0] += dw11
model.linear.weight.requires_grad = True
new_output = model(input)
new_loss = loss_fn(model(input), test_out)
dloss = new_loss - original_loss
# print(dloss / dw11)