import os
import torch
from models.gcn import ASTNN

class LibraryGNNTrainer:
    def __init__(self, library_name, loader, model=ASTNN(), batch_size=15, base_weights_path='models/base_gnn_weights.pt', pretraining=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loader = loader
        self.library_name = library_name
        self.base_weights_path = base_weights_path
        self.lib_weights_path = os.path.join('libraries/',self.library_name, self.library_name + "_gnn_weights.pt")
        self.batch_size = batch_size
        self.pretraining = pretraining

        if self.pretraining is False:
            if os.path.exists(self.lib_weights_path):
                self.model.load_state_dict(torch.load(self.lib_weights_path, map_location=self.device))
            elif os.path.exists(self.base_weights_path):
                self.model.load_state_dict(torch.load(self.base_weights_path, map_location=self.device))
        else:
            if os.path.exists(base_weights_path):
                self.model.load_state_dict(torch.load(self.base_weights_path, map_location=self.device))
        
        # Initialize the optimizer and loss criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.start_epoch = 1

    def train(self):
        self.model.train()
        for data in self.loader:
            data = data.to(self.device)
            out = self.model(data)
            loss = self.criterion(out, data.y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def test(self):
        self.model.eval()
        correct = 0
        for data in self.loader:
            data = data.to(self.device)
            out = self.model(data)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
        return correct / len(self.loader.dataset)

    def save_model(self):
        model_path = self.lib_weights_path if self.pretraining is False else self.base_weights_path
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)

    def train_and_evaluate(self, epochs, patience=10, accuracy_threshold=0.9, no_change_limit=50):
        best_train_acc = 0
        patience_counter = 0
        no_change_counter = 0 
        last_train_acc = None  

        for epoch in range(self.start_epoch, epochs + 1):
            self.train()
            train_acc = self.test()
            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}')

            if train_acc >= accuracy_threshold:
                if train_acc > best_train_acc:
                    best_train_acc = train_acc 
                else:
                    patience_counter += 1

                if last_train_acc is not None and train_acc == last_train_acc:
                    no_change_counter += 1
                else:
                    no_change_counter = 0

                if patience_counter >= patience:
                    print(f"Early stopping triggered. No improvement in train accuracy for {patience} epochs.")
                    break

                if no_change_counter >= no_change_limit:
                    print(f"Early stopping triggered. Train accuracy hasn't changed for {no_change_limit} epochs.")
                    break
            else:
                patience_counter = 0
                if last_train_acc is not None and train_acc == last_train_acc:
                    no_change_counter += 1
                else:
                    no_change_counter = 0
                if no_change_counter >= no_change_limit:
                    print(f"Early stopping triggered. Train accuracy hasn't changed for {no_change_limit} epochs.")
                    break
            last_train_acc = train_acc  

        self.save_model()
        return train_acc

