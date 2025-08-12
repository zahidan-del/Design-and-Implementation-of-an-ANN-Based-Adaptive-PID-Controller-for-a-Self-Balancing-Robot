import torch
import torch.nn as nn
import torch.onnx
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ======================
# 1. Define Neural Network Model
# ======================
class BalancingNN(nn.Module):
    def __init__(self):
        super(BalancingNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
            nn.Sigmoid()  # Outputs in range [0, 1] for Kp, Ki, Kd
        )

    def forward(self, x):
        return self.model(x)

# ======================
# 2. Load & Preprocess Dataset
# ======================
filename = "dataset_balancing.csv"
df = pd.read_csv(filename)

# Select input features and targets
X = df[['integral_error', 'angular_velocity', 'integral_angular_error', 'position', 'velocity']].values
y = df[['Kp', 'Ki', 'Kd']].values

# Normalize input & output to range [0, 1]
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Split into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# ======================
# 3. Train the Model
# ======================
model = BalancingNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 500
for epoch in range(epochs):
    model.train()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# ======================
# 4. Evaluate & Unscale Predictions
# ======================
model.eval()
with torch.no_grad():
    pred = model(X_test_tensor).numpy()  # predicted values in [0, 1]
    y_pred_unscaled = scaler_y.inverse_transform(pred)  # back to original scale
    y_test_unscaled = scaler_y.inverse_transform(y_test_tensor.numpy())  # original target values

    test_loss = criterion(torch.tensor(pred), y_test_tensor)
    print(f"Test Loss: {test_loss.item():.6f}")

    # Print sample predictions
    for i in range(5):
        print(f"Target (original): {y_test_unscaled[i]}, Prediction: {y_pred_unscaled[i]}")

# ======================
# 5. Export Model to ONNX
# ======================
dummy_input = torch.randn(1, 5)
torch.onnx.export(model, dummy_input, "balancing_nn.onnx",
                  input_names=['input'],
                  output_names=['Kp_Ki_Kd'],
                  dynamic_axes={'input': {0: 'batch_size'},
                                'Kp_Ki_Kd': {0: 'batch_size'}})

print("ONNX model saved as 'balancing_nn.onnx'")
