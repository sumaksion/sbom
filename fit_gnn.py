import os
import torch
from library_gnn_trainer import LibraryGNNTrainer
from torch_geometric.loader import DataLoader
from utils import jar_processing as jar
from utils import library_ast_data as ast

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
    method_graphs = ast.create_dataset_from_dir(out_dir, batch_size, lib_name, eval=False)
    all_datasets[lib_name] = method_graphs

for train_lib_name, train_dataset in all_datasets.items():
    print(f"Training model for library '{train_lib_name}'...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    trainer = LibraryGNNTrainer(
        library_name=train_lib_name,
        loader=train_loader,
        batch_size=batch_size
    )
    
    trainer.train_and_evaluate(epochs=300)
    
    all_certainties = [] 

    for eval_lib_name, eval_dataset in all_datasets.items():
        if eval_lib_name == train_lib_name:
            continue  

        print(f"Evaluating on dataset from library '{eval_lib_name}'...")

        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        trainer.model.eval()

        with torch.no_grad():
            for data in eval_loader:
                out = trainer.model(data)
                probabilities = torch.softmax(out, dim=1)
                certainty, pred = torch.max(probabilities, dim=1)
                
                for i, graph_certainty in enumerate(certainty):
                    all_certainties.append((graph_certainty.item(), data[i], eval_lib_name))

    all_certainties = sorted(all_certainties, key=lambda x: x[0], reverse=True)[:15]
    top_graphs = [graph for _, graph, _ in all_certainties]

    print(f"Adding top 15 graphs from other datasets to '{train_lib_name}' as class 0...")
    train_dataset.add_class_0(top_graphs)

    expanded_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    trainer.loader = expanded_loader

    print(f"Retraining model for library '{train_lib_name}' on the expanded dataset with class 0...")
    trainer.train_and_evaluate(epochs=150)  
    