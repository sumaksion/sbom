import os
import random
import torch
import shutil
from library_gnn_trainer import LibraryGNNTrainer
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset
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

def move_file_after_processing(file, input_dir, processed_dir):
    source_file = os.path.join(input_dir, file)
    destination_file = os.path.join(processed_dir, file)
    if os.path.exists(source_file) and not os.path.isdir(source_file):
        shutil.move(source_file, destination_file)


def select_for_eval(all_datasets, num_eval_datasets=7):

    lib_names = list(all_datasets.keys())
    random.shuffle(lib_names)

    eval_dataset_1 = {}
    eval_dataset_2 = {}

    libs_in_eval_1 = set()
    libs_in_eval_2 = set()

    for name in lib_names:
        if len(eval_dataset_1) >= num_eval_datasets and len(eval_dataset_2) >= num_eval_datasets:
                break
        base_lib = os.path.splitext(name)[0].rsplit('-', 1)[0]
        if base_lib not in libs_in_eval_2 and len(eval_dataset_1) <= num_eval_datasets:
            eval_dataset_1[name] = all_datasets[name]
            libs_in_eval_1.add(base_lib)
            continue
        elif base_lib not in libs_in_eval_1 and len(eval_dataset_2) <= num_eval_datasets:
            eval_dataset_2[name] = all_datasets[name]
            libs_in_eval_2.add(base_lib)            

    return eval_dataset_1, eval_dataset_2


jar_path = "data/jars"
batch_size = 15
processed_dir = os.path.join(jar_path, 'processed')
os.makedirs(processed_dir, exist_ok=True)

out_dirs_with_lib_names = jar.process_lib_files(jar_path, detecting=False)

#out_dirs_with_lib_names = find_out_directories(data_path)

all_datasets = {}

for out_dir, lib_name in out_dirs_with_lib_names:
    method_graphs = ast.create_dataset_from_dir(out_dir, batch_size, lib_name, eval=False)
    if not method_graphs:
        continue
    all_datasets[lib_name] = method_graphs
eval_datasets_1, eval_datasets_2 = select_for_eval(all_datasets, num_eval_datasets=7)
for train_lib_name, train_dataset in all_datasets.items():
    print(f"training model for '{train_lib_name}'...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    trainer = LibraryGNNTrainer(
        library_name=train_lib_name,
        loader=train_loader,
        batch_size=batch_size
    )
    
    acc = trainer.train_and_evaluate(epochs=300)
    
    all_certainties = [] 

    if train_lib_name in eval_datasets_1.keys():
        eval_datasets = eval_datasets_2
    else:
        eval_datasets = eval_datasets_1

    if eval_datasets:
        eval_dataset = ConcatDataset(list(eval_datasets.values())   )
        eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)
        trainer.model.eval()
        for i in range(5):
            with torch.no_grad():
                for data in eval_loader:
                    data = data.to(trainer.device)
                    out = trainer.model(data)
                    probabilities = torch.softmax(out, dim=1)
                    certainty, pred = torch.max(probabilities, dim=1)
                    
                    for i, graph_certainty in enumerate(certainty):
                        all_certainties.append((graph_certainty.item(), data[i]))
            batch_size = 15
            all_certainties = sorted(all_certainties, key=lambda x: x[0], reverse=True)
            total_certainties = len(all_certainties)

            start_idx = (i * batch_size) % total_certainties
            end_idx = ((i + 1) * batch_size) % total_certainties

            if start_idx < end_idx:
                certainties_batch = all_certainties[start_idx:end_idx]
            else:
                certainties_batch = all_certainties[start_idx:] + all_certainties[:end_idx]


            top_graphs = [graph for _, graph in certainties_batch]

            print(f"adding most confidently incorrect to '{train_lib_name}' as class 0...")
            train_dataset.add_class_0(top_graphs, trainer.device)

            expanded_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

            trainer.loader = expanded_loader

            print(f"training model for '{train_lib_name}' on dataset with class 0...")
            accuracy = trainer.train_and_evaluate(epochs=150)
            if accuracy > 0.85:
                break  
            print(f"Last accuracy: {accuracy}")
    lib_file = train_lib_name + '.jar'
    move_file_after_processing(lib_file, jar_path, processed_dir)