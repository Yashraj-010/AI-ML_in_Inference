import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import numpy as np
import os

class DataSet(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class ANN(nn.Module):

    def __init__(self, input_dim=3, hidden_layers=[128,128,64], output_dim=7):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def save_checkpoint(model, epoch, loss, filename):
    """Saves the model weights + optimizer state to resume training later."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    filename = filename + "/best_model.pth"
    torch.save(checkpoint, filename)
    print(f"--> Saved checkpoint: {filename}  at epoch: {epoch} with loss: {loss:.6f})")

def TrainANN(training_data, validation_data, hidden_layers=[128,128,64], batch_size=32, epochs=100, learning_rate=0.001, patience = 10, path = "ANN_model" ):
    """
    Input:
    - training_data: tuple of (x_train, y_train) where x_train is the input features and y_train is the target values for training.
    - validation_data: tuple of (x_val, y_val) where x_val is the input features and y_val is the target values for validation.
    - hidden_layers: list of integers specifying the number of neurons in each hidden layer of the ANN.
    - batch_size: integer specifying the number of samples per batch for training.
    - epochs: integer specifying the maximum number of epochs for training.
    - learning_rate: float specifying the learning rate for the optimizer.
    - patience: integer specifying the number of epochs to wait for improvement in validation loss before stopping training (early stopping).
    Output:
    - Trained ANN model with the best validation loss.

    """
    if not os.path.exists(path):
        os.makedirs(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    training_dataset = DataSet(training_data[0], training_data[1])
    validation_dataset = DataSet(validation_data[0], validation_data[1])

    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    best_loss = np.inf
    patience_count = 0

    train_losses = []
    val_losses = []

    model = ANN(hidden_layers=hidden_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting training...")

    for epoch in range(epochs):
            
        model.train()
        train_loss = 0.0   

        for x_batch, y_batch in train_loader:
            
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # print("size of x_batch: ", x_batch.size())

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:4d} | Train {train_loss:.6f} | Val {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_count = 0
            save_checkpoint(model, epoch, best_loss, filename=path)
            # torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_count += 1

        if patience_count > patience:
            print(f"Early stopping triggered at epoch {epoch:4d} with best validation loss: {best_loss:.6f}")
            break

    np.save(path + "/train_losses.npy", np.array(train_losses))
    np.save(path + "/val_losses.npy", np.array(val_losses))
    
    return train_losses, val_losses