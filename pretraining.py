import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from models.gcn import ASTNN
from utils import jar_processing as jar
from utils import library_ast_data as ast

class LibraryTrainer:
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device
        self.base_weights_path = 'models/base_gnn_weights.pt'
        # Freeze the linear layers (fully connected layers)
        for name, param in self.model.named_parameters():
            if "lin" in name:  # Matches lin1, lin2, lin3
                param.requires_grad = False
    
    def save_model(self):
        model_path = self.base_weights_path
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)

    def train_on_library(self, data_loader, epochs, learning_rate=0.001):
        """
        Train the model on a single library dataset.

        Args:
            data_loader: A DataLoader object containing the library dataset.
            epochs: Number of epochs to train for.
            learning_rate: Learning rate for the optimizer.
        """
        # Use only parameters that require gradients
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for data in data_loader:
                data = data.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, data.y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = output.max(1)
                correct += (predicted == data.y).sum().item()
                total += data.y.size(0)

            accuracy = correct / total
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

    def train_on_multiple_libraries(self, libraries, epochs_per_library, batch_size=32, learning_rate=0.001):
        """
        Train the model sequentially on multiple library datasets.

        Args:
            libraries: A list of library datasets (PyG datasets or Data objects).
            epochs_per_library: Number of epochs to train on each library.
            batch_size: Batch size for training.
            learning_rate: Learning rate for the optimizer.
        """
        for library_idx, library in enumerate(libraries):
            print(f"Training on library {library_idx + 1}/{len(libraries)}...")

            data_loader = DataLoader(library, batch_size=batch_size, shuffle=True)
            self.train_on_library(data_loader, epochs=epochs_per_library, learning_rate=learning_rate)
            self.save_model()

# Example Usage
if __name__ == "__main__":
    # Assuming `ASTNN` is defined elsewhere and libraries are loaded as PyG datasets
    model = ASTNN()
    base_weights_path = os.path.join('models', 'base_gnn_weights.pt')
    if os.path.exists(base_weights_path):
        weights_path = base_weights_path
        model.load_state_dict(torch.load(weights_path))

    dot_dirs = jar.process_lib_files('data/jars', detecting=False)
    
    libraries = []

    for out_dir, lib_name in dot_dirs:
        method_graphs = ast.create_dataset_from_dir(out_dir, 15, lib_name, eval=False)
        if not method_graphs:
            continue
        libraries.append(method_graphs)
    trainer = LibraryTrainer(model)
    trainer.train_on_multiple_libraries(libraries, epochs_per_library=50)
