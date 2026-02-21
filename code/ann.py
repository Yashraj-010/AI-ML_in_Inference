import torch
import torch.nn as nn
import torch.utils.data as DataLoader

import numpy as np

class DataLoader(DataLoader.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class ANN(nn.Module):
    def __init__(self, input_dim=3, hidden_layers=[128,128,64], output_dim=7):
        super().__init__()

        self.hidden_layers = hidden_layers
        self.fc1 = nn.Linear(input_dim, hidden_layers[0])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.fc4 = nn.Linear(hidden_layers[2], output_dim)

    def forward(self, x):
        model = nn.Sequential(
            self.fc1,
            self.relu,
            self.fc2,
            self.relu,
            self.fc3,
            self.relu,
            self.fc4
        )
        return model(x)

def save_checkpoint(model, epoch, loss, filename="best_model.pth"):
    """Saves the model weights + optimizer state to resume training later."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"--> Saved checkpoint: {filename}  at epoch: {epoch} with loss: {loss:.6f})")

def TrainEmulator(training_data, validation_data, hidden_layers=[128,128,64], batch_size=32, epochs=100, learning_rate=0.001, patientce = 10 ):

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    best_loss = np.inf
    patientce_count = 0

    model = ANN(hidden_layers=hidden_layers).to('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(100):
            
        model.train()
        train_loss = 0.0   

        for x_batch, y_batch in train_loader:
            
            x_batch, y_batch = x_batch.to('cuda' if torch.cuda.is_available() else 'cpu'), y_batch.to('cuda' if torch.cuda.is_available() else 'cpu')


            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred,)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to('cuda' if torch.cuda.is_available() else 'cpu'), y_batch.to('cuda' if torch.cuda.is_available() else 'cpu')
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch:4d} | Train {train_loss:.6f} | Val {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patientce_count = 0
            save_checkpoint(model, epoch, best_loss)
            # torch.save(model.state_dict(), "best_model.pt")
        else:
            patientce_count += 1

        if patience_counter > patience:
            print(f"Early stopping triggered at epoch {epoch:4d} with best validation loss: {best_loss:.6f}")
            break