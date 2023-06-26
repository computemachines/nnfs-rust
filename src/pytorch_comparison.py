import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = torch.from_numpy(np.linspace(0, 1, 100).reshape(-1, 1)).float().to(device)
y = torch.from_numpy(np.sin(2 * np.pi * X.cpu().numpy())).float().to(device)


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


model = NeuralNetwork().to(device)

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=(1-1e-3))

def train_loop(X, y, model, loss_fn, optimizer, scheduler):
    model.train()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    return loss.item()

def train(num_epochs=10000):
    for epoch in range(num_epochs):
        loss = train_loop(X, y, model, loss_fn, optimizer, scheduler)
        if epoch % 1000 == 0:
            print(f"Epoch: {epoch}, Loss: {loss}")

# train()
# # model.save(".")
# plt.plot(X.cpu().numpy(), model(X).cpu().detach().numpy(), "k--")
# plt.plot(X.cpu().numpy(), y.cpu().numpy())
# plt.show()

# if __name__ == "__main__":
#     train()
#     torch.save(model, "model.pth")

model = torch.load("model.pth")

X_infer = torch.from_numpy(np.array([0.1, 0.2]).reshape(-1, 1)).float().to(device)
with torch.no_grad():
    W1 = model.linear_relu_stack[0].weight.cpu().numpy()
    b1 = model.linear_relu_stack[0].bias.cpu().numpy()
    W2 = model.linear_relu_stack[2].weight.cpu().numpy()
    b2 = model.linear_relu_stack[2].bias.cpu().numpy()
    W3 = model.linear_relu_stack[4].weight.cpu().numpy()
    b3 = model.linear_relu_stack[4].bias.cpu().numpy()
    print("b1", b1)
def relu(x):
    return np.maximum(0, x)

def linear(x, W, b):
    return np.dot(x, W.T) + b

X_infer_np = X_infer.cpu().numpy()

# First layer (Linear -> ReLU)
Z1 = linear(X_infer_np, W1, b1)
print("Z1", Z1)
A1 = relu(Z1)

# Second layer (Linear -> ReLU)
Z2 = linear(A1, W2, b2)
A2 = relu(Z2)

    # Third layer (Linear)
Z3 = linear(A2, W3, b3)

# Z3 now contains the output of the network
print(Z3)

