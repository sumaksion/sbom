import os
import torch
from torch_geometric.loader import DataLoader
from models.gcn import GCN
from utils import jar_processing as jar
from utils import library_ast_data as ast

class LibraryGNNTrainer:
    def __init__(self, library_name, loader, batch_size=15, base_weights_path='models/base_gnn_weights.pt', checkpoint_path='models/checkpoints/'):
        self.loader = loader
        self.library_name = library_name
        self.lib_weights_path = os.path.join('libraries/',self.library_name, self.library_name + "_gnn_weights.pt")
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path

        self.model = GCN(hidden_channels=16)
        if os.path.exists(self.lib_weights_path):
            self.model.load_state_dict(torch.load(self.lib_weights_path))
        elif os.path.exists(base_weights_path):
            self.model.load_state_dict(torch.load(base_weights_path))
        
        # Initialize the optimizer and loss criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.start_epoch = 1

    def train(self):
        self.model.train()
        for data in self.loader:
            out = self.model(data.x, data.edge_index, data.batch)
            loss = self.criterion(out, data.y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def test(self):
        self.model.eval()
        correct = 0
        for data in self.loader:
            out = self.model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
        return correct / len(self.loader.dataset)

    def save_model(self):
        model_path = f'libraries/{self.library_name}/{self.library_name}_gnn_weights.pt'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)

    def train_and_evaluate(self, epochs, patience=100, accuracy_threshold=0.9):
        best_train_acc = 0  
        patience_counter = 0  
        for epoch in range(self.start_epoch, epochs + 1):
            self.train()
            train_acc = self.test()
            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}')

            if train_acc >= accuracy_threshold:
                if train_acc > best_train_acc:
                    best_train_acc = train_acc  
                    patience_counter = 0  
                else:
                    patience_counter += 1  

                if patience_counter >= patience:
                    print(f"Early stopping triggered. No improvement in train accuracy for {patience} epochs.")
                    break
            else:

                patience_counter = 0


        self.save_model()



def find_out_directories(data_path):
    out_directories = []
    for root, dirs, files in os.walk(data_path):
        for dir_name in dirs:
            if dir_name == "out":
                inner_subdir = os.path.basename(root) 
                outer_subdir = os.path.basename(os.path.dirname(root)) 
                lib_name = f"{outer_subdir}-{inner_subdir}"
                out_directories.append((os.path.join(root, dir_name), lib_name))
    return out_directories


jar_path = "data/jars"
batch_size = 15

out_dirs_with_lib_names = jar.process_lib_files(jar_path, detecting=False)

#out_dirs_with_lib_names = find_out_directories(data_path)

all_datasets = {}

for out_dir, lib_name in out_dirs_with_lib_names:
    print(f"Creating dataset for library '{lib_name}' with data from '{out_dir}'...")
    method_graphs = ast.create_dataset_from_dot(out_dir, batch_size, lib_name, eval=False)
    all_datasets[lib_name] = method_graphs

for train_lib_name, train_dataset in all_datasets.items():
    print(f"Training model for library '{train_lib_name}'...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    trainer = LibraryGNNTrainer(
        library_name=train_lib_name,
        loader=train_loader,
        batch_size=batch_size
    )
    
    trainer.train_and_evaluate(epochs=170)
    
    all_certainties = [] 

    for eval_lib_name, eval_dataset in all_datasets.items():
        if eval_lib_name == train_lib_name:
            continue  # Skip evaluation on the training dataset

        print(f"Evaluating on dataset from library '{eval_lib_name}'...")

        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        trainer.model.eval()

        with torch.no_grad():
            for data in eval_loader:
                out = trainer.model(data.x, data.edge_index, data.batch)
                probabilities = torch.softmax(out, dim=1)
                certainty, pred = torch.max(probabilities, dim=1)
                
                for i, graph_certainty in enumerate(certainty):
                    all_certainties.append((graph_certainty.item(), data[i], eval_lib_name))

    all_certainties = sorted(all_certainties, key=lambda x: x[0], reverse=True)[:15]
    top_graphs = [graph for _, graph, _ in all_certainties]

    print(f"Adding top 15 graphs from other datasets to '{train_lib_name}' as class 0...")
    train_dataset.add_class_0(top_graphs)

    expanded_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    trainer.loader = expanded_loader

    print(f"Retraining model for library '{train_lib_name}' on the expanded dataset with class 0...")
    trainer.train_and_evaluate(epochs=700)  
    