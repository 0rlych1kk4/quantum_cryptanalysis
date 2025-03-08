import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# Define a simple AI model for cryptanalysis
class CryptoAI(nn.Module):
    def __init__(self):
        super(CryptoAI, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(256, 128),  # Input Layer: 256 neurons -> Hidden Layer: 128 neurons
            nn.ReLU(),
            nn.Linear(128, 256)   # Hidden Layer: 128 neurons -> Output Layer: 256 neurons
        )

    def forward(self, x):
        return self.model(x)

# Generate synthetic training data (encrypted messages)
data = np.random.randint(0, 256, (1000, 256)).astype(np.float32)  # 1000 samples, 256 features
labels = data.copy()  # For simplicity, assume labels are the same as inputs

# Convert NumPy arrays to PyTorch tensors
train_data = torch.tensor(data)
train_labels = torch.tensor(labels)

# Initialize AI model
model = CryptoAI()
criterion = nn.MSELoss()  # Loss function: Mean Squared Error
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer: Adam

# Train AI model
print(" Training AI model for cryptanalysis...")
for epoch in range(100):
    optimizer.zero_grad()
    output = model(train_data)
    loss = criterion(output, train_labels)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f" Epoch {epoch}, Loss: {loss.item():.4f}")

# Ensure the models directory exists
if not os.path.exists("models"):
    os.makedirs("models")

# Save trained AI model
model_path = "models/ai_model.pth"
torch.save(model.state_dict(), model_path)
print(f" AI model saved to {model_path}")

