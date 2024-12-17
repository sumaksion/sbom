import os
import torch
import re
import glob
from collections import OrderedDict
from torch_geometric.loader import DataLoader
from models.gcn import GCN
from utils import library_ast_data as ast
from utils import jar_processing as jar
from utils import find_lib_dots as dots
from library_gnn_trainer import LibraryGNNTrainer

def get_dot_files_with_labels(directory, file_sizes, tolerance_kb=7):

    dot_files = glob.glob(os.path.join(directory, '*.dot'))
    dot_files = sorted(dot_files, key=os.path.getsize)
    max_size = max(file_sizes) + 1 
    min_size = min(file_sizes) - tolerance_kb
    dot_files_out = []
    idxs_and_sizes = []
    for file in dot_files[:]:
        size = os.path.getsize(file) // 1024
        if size < min_size:
            dot_files.remove(file)
        if size >= min_size:
            break
    dot_files.reverse()

    for file in dot_files[:]:
        size = os.path.getsize(file) // 1024
        if size > max_size:
            dot_files.remove(file)
        if size <= max_size:
            break

    idx = 0
    label_mapping = {}
    labels = []
    for target_size in file_sizes:

        

        for file in dot_files[:]:
            file_size = os.path.getsize(file) // 1024
            diff = target_size - file_size 
            # methods in memory should be smaller than in the jar
            if diff < -1:
                dot_files.remove(file) # targets are sorted, so won't be useful for future targets
                continue
            # files are sorted, so this is minimum diff for this target
            if diff > tolerance_kb:
                break
            else:
                dot_files_out.append(file)
                idxs_and_sizes.append((idx, file_size) )
                labels.append(idx)
                idx += 1

    for obj in idxs_and_sizes:
        for idx, size in enumerate(file_sizes):
            diff = size - obj[1]
            if diff < -1:
                break
            if diff > tolerance_kb:
                continue
            else:
                if obj[0] not in label_mapping:
                    label_mapping[obj[0]] = []
                label_mapping[obj[0]].append(idx+1)

    return dot_files_out, label_mapping

def detect_library_in_apk(apk_ast_data, model, label_mapping: dict, batch_size=64, num_classes=16, threshold=0.8):
    library_vector = [0] * num_classes
    eval_loader = DataLoader(apk_ast_data, batch_size=batch_size, shuffle=False)
    
    
    with torch.no_grad():
        for data in eval_loader:
            out = model(data.x, data.edge_index, data.batch)
            probabilities = torch.softmax(out, dim=1)
            
            for i, prob in enumerate(probabilities):
                certainty, pred = torch.max(prob, dim=0)

                potential_labels = label_mapping[data.y[i].item()]
                if pred.item() == 0 or pred.item() in potential_labels:
                    if certainty > threshold:
                        library_vector[pred.item()] += 1

    return library_vector

def expand_library_dataset_from_apk(
    apk_ast_data, model, lib_name, batch_size=64, num_classes=16, threshold=0.8, 
    higher_threshold=0.7
):
    library_vector = [0] * num_classes
    eval_loader = DataLoader(apk_ast_data, batch_size=batch_size, shuffle=False)
    lib_dataset_path = f'libraries/{lib_name}/ast_dataset'
    label_counts = {}
    training_data = []
    labels = []
    highest_certainty_items = {}

    with torch.no_grad():
        for data in eval_loader:

            for label in data.y.tolist():
                label_counts[label] = label_counts.get(label, 0) + 1

            out = model(data.x, data.edge_index, data.batch)
            probabilities = torch.softmax(out, dim=1)

            for i, prob in enumerate(probabilities):
                certainty, pred = torch.max(prob, dim=0)
                true_label = data.y[i].item()
                label_probability = prob[true_label].item()

                if pred.item() == 0 or pred.item() == true_label:
                    if certainty > threshold:
                        library_vector[pred.item()] += 1

                if label_counts[true_label] == 1:
                    training_data.append(data[i])
                    labels.append(true_label)
                
                else:
                    if (
                        true_label not in highest_certainty_items 
                        or highest_certainty_items[true_label]["certainty"] < certainty
                    ):
                        highest_certainty_items[true_label] = {
                            "data": data[i],
                            "certainty": certainty
                        }
                    if certainty > higher_threshold:
                        training_data.append(data[i])
                        labels.append(true_label)
                
                print(f"Data index {i} true label: {true_label}")
                print(f"Probability of true label: {label_probability}")
                print(f"All class probabilities: {prob.tolist()}")

            for label, item_info in highest_certainty_items.items():
                if label not in labels:  
                    training_data.append(item_info["data"])
                    labels.append(label)
    try:
        training_dataset = ast.CustomASTDataset.load_dataset_from_root(lib_dataset_path)
        print("Dataset loaded successfully!")
    except FileNotFoundError as e:
        print(e)
    training_dataset.add_data(training_data, labels)
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)  
    
    trainer = LibraryGNNTrainer(
        library_name=lib_name,
        loader=train_loader,
        batch_size=batch_size
    )   
    trainer.train_and_evaluate(epochs=300)
    return library_vector



def collect_file_sizes(base_directory='libraries'):
    file_sizes_dict = {}

    for root, _, files in os.walk(base_directory):
        for file in files:
            if file == "sizes.txt":
                parent_dir = os.path.basename(root)
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()[:-2]  
                        sizes = []
                        for line in lines:
                            if "bytes" in line:
                                size = int(line.split(":")[1].strip().split()[0]) // 1024
                                sizes.append(size)
                        
                        file_sizes_dict[parent_dir] = sizes

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    
    return file_sizes_dict


def extract_graph_name(dot_file_path):

    with open(dot_file_path, 'r') as file:
        first_line = file.readline().strip()
        if first_line.startswith("digraph"):
            graph_name = first_line.split(" ")[1].strip(' {')
            return graph_name
    return None

def get_graph_names_and_indexes(directory, num_files=15):

    dot_files = glob.glob(os.path.join(directory, "*.dot"))
    
    dot_files = sorted(dot_files, key=os.path.getsize, reverse=True)
    
    largest_files = dot_files[:num_files]
    
    result = []
    for idx, file_path in enumerate(largest_files):
        graph_name = extract_graph_name(file_path)
        if graph_name:
            result.append((graph_name, idx))
    
    return result

sizes_dict = collect_file_sizes()
dir = 'data/dexs'
joern_workspace = os.path.join(dir, 'workspace')

all_datasets = {}

if not os.path.exists(joern_workspace):
    os.mkdir(joern_workspace)

out_dirs_apk_names = jar.process_lib_files(dir, joern_workspace=joern_workspace, detecting=True)

#out_dirs_apk_names = [('data/test/apod_md/out', 'apod_md')]

classed_dot_files = [] 
labels = []   

for out_dir, apk_name in out_dirs_apk_names:
    detected_libs_string = []
    detected_libs_model = []
    for lib_name in sizes_dict.keys():

        detection_dataset_root = os.path.join('detection', apk_name, lib_name)
        model = GCN(hidden_channels=16)
        model.load_state_dict(torch.load(f'libraries/{lib_name}/{lib_name}_gnn_weights.pt'))

        search_string = lib_name.split("-")[0]
        lib_number = lib_name.split("-")[-1]
        max_size = max(sizes_dict[lib_name]) + 1
        min_size = min(sizes_dict[lib_name]) - 6
        lib_apk_files = dots.find_lib_dot_files(out_dir, search_string, max_size, min_size)
        lib_dot_dir = os.path.join('data', 'jars', 'workspace', search_string, lib_number, 'out')
        name_tuple = (apk_name, lib_name)
        dataset_name = apk_name + "_" + lib_name

        if lib_apk_files:
            detected_libs_string.append(lib_name)
            if not os.path.exists(os.path.join(detection_dataset_root, 'ast_dataset', 'processed', 'data.pt')):
                print(f"Creating dataset for library '{lib_name}' with data from '{apk_name}'...")
                lib_graph_classes = get_graph_names_and_indexes(lib_dot_dir)  
                
                for graph_name, index in lib_graph_classes:
                    for dot_file in lib_apk_files:
                        apk_graph_name = extract_graph_name(dot_file)
                        if apk_graph_name == graph_name:
                            classed_dot_files.append(dot_file)
                            labels.append(index + 1)
                method_graphs = ast.create_dataset_from_dot_files(classed_dot_files, name_tuple, labels)
            else:
                try:
                    method_graphs = ast.CustomASTDataset.load_dataset_from_root(os.path.join(detection_dataset_root, 'ast_dataset'))
                except FileNotFoundError as e:
                    print(e)
                
            
                
            all_datasets[dataset_name] = method_graphs

            lib_vector = expand_library_dataset_from_apk(method_graphs, model, lib_name)

        else: 
            if not os.path.exists(os.path.join(detection_dataset_root, 'ast_dataset', 'processed', 'data.pt')):  
                sizes = sizes_dict[lib_name]
                dot_files, label_mapping = get_dot_files_with_labels(out_dir, sizes)
                method_graphs = ast.create_dataset_from_dot_files(dot_files, name_tuple, list(label_mapping.keys())) 
                all_datasets[dataset_name] = method_graphs
            else:
                try:
                    method_graphs = ast.CustomASTDataset.load_dataset_from_root(os.path.join(detection_dataset_root, 'ast_dataset'))
                except FileNotFoundError as e:
                    print(e)
        lib_vector = detect_library_in_apk(method_graphs, model, label_mapping)

        for i, int in enumerate(lib_vector):
            if i == 0:
                continue
            if int > 0:
                detected_libs_model.append(lib_name)
        print(f"string search: {detected_libs_string}, model: {lib_vector}") 
"""
models = {}
for name in sizes_dict.keys():
    model = GCN(hidden_channels=16)
    model.load_state_dict(torch.load(f'libraries/{name}/{name}_gnn_weights.pt'))
    models[name] = model

for apk, ast_data in all_datasets.items():
    for name, model in models.items():
            
        lib_label_map = label_map_dict[name]
        vector = detect_library_in_apk(ast_data, model, lib_label_map)
        print(f"Library '{name}' detected in '{apk}': {vector}")"""
