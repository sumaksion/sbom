import os
import torch
import json
import glob
from collections import OrderedDict
from torch_geometric.loader import DataLoader
from models.gcn import ASTNN
from utils import text_stripper as strip
from utils import library_ast_data as ast
from utils import jar_processing as jar
from utils import find_lib_dots as dots
from library_gnn_trainer import LibraryGNNTrainer

def get_dot_files_with_labels(directory, file_sizes: dict, filter: dict, tolerance_kb=7):

    dot_files = glob.glob(os.path.join(directory, '*.dot'))
    dot_files = sorted(dot_files, key=os.path.getsize)
    max_size = max(file_sizes, key=lambda x: x[0])[0] + 1
    min_size = min(file_sizes, key=lambda x: x[0])[0] - tolerance_kb
    dot_files_out = []
    idxs_and_sizes = []
    idx = 0
    label_mapping = {}
    labels = []
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


    for size, label in file_sizes:
        filter_list = filter.get(label, [])

        for file in dot_files[:]:
            file_size = os.path.getsize(file) // 1024
            diff = size - file_size

            # Methods in memory should be smaller than in the jar
            if diff < -1:
                dot_files.remove(file)  # Targets are sorted, so this file won't be useful for future targets
                continue
            
            # Minimum diff for this target (files are sorted)
            if diff > tolerance_kb:
                break
            
            # Check if the file matches the filter criteria
            file_beginning = strip.get_ast_first_children(file)
            if file_beginning == filter_list:
                dot_files_out.append(file)
                idxs_and_sizes.append((idx, file_size, file_beginning))
                labels.append(idx)
                idx += 1

    # Create label mapping
    for obj in idxs_and_sizes:
        for size, label in file_sizes:
            diff = size - obj[1]
            if diff < -1:
                break
            if diff > tolerance_kb:
                continue
            else:
                if obj[2] == filter.get(label, []):
                    if obj[0] not in label_mapping:
                        label_mapping[obj[0]] = []
                    label_mapping[obj[0]].append(label)

    return dot_files_out, label_mapping

def detect_library_in_apk(apk_ast_data, model, lib_name, file_mapping, label_mapping=None, batch_size=64, num_classes=16, threshold=0.99):
    library_vector = [0] * num_classes
    eval_loader = DataLoader(apk_ast_data.to(device), batch_size=batch_size, shuffle=False)
    
    
    with torch.no_grad():
        for data in eval_loader:
            out = model(data)
            probabilities = torch.softmax(out, dim=1)
            
            for i, prob in enumerate(probabilities):
                certainty, pred = torch.max(prob, dim=0)
                if label_mapping is None:
                    potential_label = data.y[i].item()
                    if pred.item() == 0 or pred.item() == potential_label:
                        if certainty > threshold:
                            library_vector[pred.item()] += 1
                else:
                    potential_labels = label_mapping[data.y[i].item()]
                    if pred.item() == 0 or pred.item() in potential_labels:
                        if certainty > threshold:
                            library_vector[pred.item()] += 1
    for idx, field in enumerate(library_vector[1:]):
        if field > 2:
            library_vector[idx+1] = 0
            name_list = lib_name.rsplit('-', 1)
            dot_dir = os.path.join('data', 'jars', 'workspace', name_list[0], name_list[1], 'out')
            if len(file_mapping.keys()) == 15:
                smallest_file_name = file_mapping[15]
            else:
                print('dataset should be of size 15')
                return library_vector
            smallest_file_path = os.path.join(dot_dir, smallest_file_name)
            dot_files = sorted(glob.glob(os.path.join(dot_dir, "*.dot" )), key=os.path.getsize, reverse=True)
            index_smallest_file = dot_files.index(smallest_file_path)
            replacement_file = dot_files[index_smallest_file+1]
            lib_dir = os.path.join('libraries', lib_name, 'ast_dataset')
            dataset = ast.CustomASTDataset.load_dataset_from_root(lib_dir)
            dataset.replace_class(replacement_file, idx+1)
            # only new class
            new_data_list = [data.to(device) for data in dataset if data.y.item() == idx + 1]
            train_loader = DataLoader(new_data_list, batch_size=batch_size, shuffle=True)
            trainer = LibraryGNNTrainer(
                library_name=lib_name,
                loader=train_loader,
                batch_size=batch_size
            )   
            trainer.train_and_evaluate(epochs=100)
    return library_vector

def expand_library_dataset_from_apk(
    apk_ast_data, model, lib_name, batch_size=64, num_classes=16, threshold=0.99, 
    higher_threshold=0.7
):
    library_vector = [0] * num_classes
    eval_loader = DataLoader(apk_ast_data.to(device), batch_size=batch_size, shuffle=False)
    lib_dataset_path = f'libraries/{lib_name}/ast_dataset'
    label_counts = {}
    training_data = []
    labels = []
    highest_certainty_items = {}

    with torch.no_grad():
        for data in eval_loader:

            for label in data.y.tolist():
                label_counts[label] = label_counts.get(label, 0) + 1

            out = model(data)
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
                    if label_probability > higher_threshold:
                        training_data.append(data[i])
                        labels.append(true_label)
                

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
    train_loader = DataLoader(training_dataset.to(device), batch_size=batch_size, shuffle=True)  
    
    trainer = LibraryGNNTrainer(
        library_name=lib_name,
        loader=train_loader,
        batch_size=batch_size
    )   
    trainer.train_and_evaluate(epochs=300)
    return library_vector



def collect_file_sizes(base_directory='libraries'):
    classes_sizes_dict = {}
    files_classes_dict = {}
    for root, _, files in os.walk(base_directory):
        for file in files:
            if file == "sizes.json":
                parent_dir = os.path.basename(root)
                file_path = os.path.join(root, file)
                size_dict = load_dictionary(file_path)
                sorted_size_dict = {label: size_info for label, size_info in sorted(size_dict.items(), key=lambda item: item[1]['size'], reverse=True)}
                size_list = [(size_info['size'] // 1024, label) for label, size_info in sorted_size_dict.items()]
                file_dict = {label: file_info['file'] for label, file_info in sorted_size_dict.items()}
                #size_list.sort(reverse=True, key=lambda x: x[0])
                files_classes_dict[parent_dir] = file_dict
                classes_sizes_dict[parent_dir] = size_list

    return classes_sizes_dict, files_classes_dict

def save_mapping(label_mapping, dataset_root, name):
    label_mapping_path = os.path.join(dataset_root, name)
    with open(label_mapping_path, "w") as f:
        json.dump(label_mapping, f, indent=4)
    print(f"Label mapping saved to {label_mapping_path}")

def load_dictionary(path):

    if os.path.exists(path):
        with open(path, "r") as f:
            dict = {int(k): v for k, v in json.load(f).items()}
        print(f"Dictionary loaded from {path}")
        return dict
    else:
        print(f"No dictionary found at {path}")
        return None
    
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
sizes_dict, files_dict = collect_file_sizes()
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
        model = ASTNN().to(device)
        model.load_state_dict(torch.load(f'libraries/{lib_name}/{lib_name}_gnn_weights.pt', map_location=device))

        search_string = lib_name.split("-")[0]
        lib_number = lib_name.split("-")[-1]
        max_size = max(sizes_dict[lib_name], key=lambda x: x[0])[0] + 1
        min_size = min(sizes_dict[lib_name], key=lambda x: x[0])[0] - 6
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
                lib_vector = expand_library_dataset_from_apk(method_graphs, model, lib_name)
            else:
                try:
                    method_graphs = ast.CustomASTDataset.load_dataset_from_root(os.path.join(detection_dataset_root, 'ast_dataset'))
                    file_mapping = files_dict[lib_name]
                    lib_vector = detect_library_in_apk(method_graphs, model, lib_name, file_mapping)
                except FileNotFoundError as e:
                    print(e)
                
            
                
            all_datasets[dataset_name] = method_graphs


        else: 
            if not os.path.exists(os.path.join(detection_dataset_root, 'ast_dataset', 'processed', 'data.pt')):  
                sizes = sizes_dict[lib_name]
                filter_dict = load_dictionary(os.path.join('libraries', lib_name, 'classes_first_children.json'))
                dot_files, label_mapping = get_dot_files_with_labels(out_dir, sizes, filter_dict)
                file_mapping = files_dict[lib_name]
                if dot_files:
                    method_graphs = ast.create_dataset_from_dot_files(dot_files, name_tuple, list(label_mapping.keys()))
                    save_mapping(label_mapping, detection_dataset_root, 'label_mapping.json')
                    save_mapping(file_mapping, detection_dataset_root, 'file_mapping.json')
                else:
                    continue
                all_datasets[dataset_name] = method_graphs
            else:
                try:
                    method_graphs = ast.CustomASTDataset.load_dataset_from_root(os.path.join(detection_dataset_root, 'ast_dataset'))
                    label_mapping_path = os.path.join(detection_dataset_root, 'label_mapping.json')
                    file_mapping_path = os.path.join(detection_dataset_root, 'file_mapping.json')
                    label_mapping = load_dictionary(label_mapping_path)
                    file_mapping = load_dictionary(file_mapping_path)
                except FileNotFoundError as e:
                    print(e)
            lib_vector = detect_library_in_apk(method_graphs, model, lib_name, file_mapping, label_mapping)

        for field in lib_vector[1:]:
            if field > 0:
                detected_libs_model.append(lib_name)
                break
        print(f"Library Vector for {lib_name}: {lib_vector}")
    print(f"Libraries detected in {apk_name}:")
    print(f"string search: {detected_libs_string}, model: {detected_libs_model}") 
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
