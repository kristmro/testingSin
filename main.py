import numpy as np
import torch
from torch import nn, optim
from copy import deepcopy

# Generate synthetic dataset for the demonstration
def generate_synthetic_data():
    def sine_wave(x, amplitude, phase):
        return amplitude * np.sin(x + phase)

    x = np.random.uniform(-5, 5, size=(1000,))
    amplitude = np.random.uniform(0.1, 5)
    phase = np.random.uniform(0, np.pi)
    y = sine_wave(x, amplitude, phase)

    return x, y

# Define the model
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super(SimpleNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# MAML training step
def maml_train_step(model, optimizer, x, y, inner_lr=0.01, meta_lr=0.001):
    model.train()
    original_weights = deepcopy(model.state_dict())

    # Inner loop: task-specific adaptation
    for _ in range(5):
        predictions = model(x)
        loss = nn.MSELoss()(predictions, y)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        for param, grad in zip(model.parameters(), grads):
            param.data -= inner_lr * grad

    # Meta-update (outer loop)
    predictions = model(x)
    meta_loss = nn.MSELoss()(predictions, y)
    optimizer.zero_grad()
    meta_loss.backward()
    optimizer.step()

    # Restore original weights
    model.load_state_dict(original_weights)
    return meta_loss.item()

# Reptile training step
def reptile_train_step(model, optimizer, x, y, inner_lr=0.01, meta_lr=0.001):
    model.train()
    original_weights = deepcopy(model.state_dict())

    # Inner loop: task-specific adaptation
    for _ in range(5):
        predictions = model(x)
        loss = nn.MSELoss()(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Reptile meta-update
    for param, original_param in zip(model.parameters(), original_weights.values()):
        param.data = original_param + meta_lr * (param.data - original_param)

    return loss.item()

# Main training loop
def train_meta_learning(method="MAML", epochs=1000, inner_lr=0.01, meta_lr=0.001):
    model = SimpleNN(input_dim=1, hidden_layers=[40, 40], output_dim=1).cuda()
    optimizer = optim.Adam(model.parameters(), lr=meta_lr)

    for epoch in range(epochs):
        x, y = generate_synthetic_data()
        x = torch.tensor(x, dtype=torch.float32).view(-1, 1).cuda()
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1).cuda()

        if method == "MAML":
            loss = maml_train_step(model, optimizer, x, y, inner_lr=inner_lr, meta_lr=meta_lr)
        elif method == "Reptile":
            loss = reptile_train_step(model, optimizer, x, y, inner_lr=inner_lr, meta_lr=meta_lr)
        else:
            raise ValueError("Unsupported method. Choose 'MAML' or 'Reptile'.")

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss}")

if __name__ == "__main__":
    train_meta_learning(method="MAML")
