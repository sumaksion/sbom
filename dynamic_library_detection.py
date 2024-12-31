import os
import torch
import json
import glob
import random
import shutil
from torch_geometric.loader import DataLoader
from models.gcn import ASTNN
from utils import text_stripper as strip
from utils import library_ast_data as ast
from utils.jar_processing import process_lib_files
from utils.find_lib_dots import find_lib_dot_files

from library_gnn_trainer import LibraryGNNTrainer

def get_dot_files_with_labels(file_sizes: dict, filter: dict, dot_files, tolerance_kb=7):

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
            if match(file_beginning, filter_list) is True:
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
                if match(obj[2], filter.get(label, [])):
                    if obj[0] not in label_mapping:
                        label_mapping[obj[0]] = []
                    label_mapping[obj[0]].append(label)

    return dot_files_out, label_mapping

def match(file_beginning: list, filter: list):

  if len(file_beginning) != len(filter):
    return False  

  for i in range(len(file_beginning)):
    item_beginning = file_beginning[i]
    item_filter = filter[i]
    if item_beginning == item_filter:
        continue
    filter_parts = item_filter.split(",", 1)
    beginning_parts = item_beginning.split(",", 1)
    if filter_parts[0] != beginning_parts[0]:
        return False  
    filter_parts = filter_parts[1].split(".")
    beginning_parts = beginning_parts[1].split(".")
    if len(filter_parts) != len(beginning_parts):
        return False

  return True
    

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
        if field > 5:
            print(f"suspiciously many instances of class {idx+1}, replacing class...")
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
            # only new class and calss 0
            new_data_list = [data.to(device) for data in dataset if data.y.item() == idx + 1 or data.y.item() == 0]
            train_loader = DataLoader(new_data_list, batch_size=batch_size, shuffle=True)
            trainer = LibraryGNNTrainer(
                library_name=lib_name,
                loader=train_loader,
                batch_size=batch_size
            )   
            trainer.train_and_evaluate(epochs=100)
    return library_vector

def expand_library_dataset_from_apk(
    apk_ast_data, model, lib_name, batch_size=64, num_classes=16, threshold=0.9, 
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
    except FileNotFoundError as e:
        print(e)
    training_dataset.add_data(training_data, labels)
    train_loader = DataLoader(training_dataset.to(device), batch_size=batch_size, shuffle=True)  
    
    trainer = LibraryGNNTrainer(
        library_name=lib_name,
        loader=train_loader,
        batch_size=batch_size
    )
    print("adding ASTs with high probability of matching classes and retraining model")   
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

def load_dictionary(path):

    if os.path.exists(path):
        with open(path, "r") as f:
            dict = {int(k): v for k, v in json.load(f).items()}
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

def move_file_after_processing(file, input_dir, processed_dir):
    source_file = os.path.join(input_dir, file)
    destination_file = os.path.join(processed_dir, file)
    if os.path.exists(source_file) and not os.path.isdir(source_file):
        shutil.move(source_file, destination_file)

def get_graph_names_and_indexes(directory, num_files=15):

    dot_files = glob.glob(os.path.join(directory, "*.dot"))
    
    dot_files = sorted(dot_files, key=os.path.getsize, reverse=True)
    
    largest_files = dot_files[:num_files]
    
    result = []
    for idx, file_path in enumerate(largest_files):
        graph_name = extract_graph_name(file_path)
        if graph_name:
            result.append((graph_name, idx+1))
    
    return result

def output_results_json(output_file, apk_name, best_version, valid_queries, detected_libs_string):
    apk_data = {
        apk_name: [value["name"] for value in best_version.values()]
    }

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            cumulative_data = json.load(f)
    else:
        cumulative_data = {}

    cumulative_data.update(apk_data)

    with open(output_file, "w") as f:
        json.dump(cumulative_data, f, indent=4)

    valid_queries = {}

    combined_results = []
    combined_lib_names = set()

    combined_lib_names.update(value["name"] for value in best_version.values())

    combined_lib_names.update(detected_libs_string)

    for key, objects in valid_queries.items():
        for obj in objects:
            combined_lib_names.add(f"{key}:{obj}")

    for lib_name in combined_lib_names:
        detection_methods = []

        if lib_name in (value["name"] for value in best_version.values()):
            detection_methods.append("model")

        if any(lib_name.startswith(key) for key in valid_queries):
            detection_methods.append("file structure")

        if lib_name in detected_libs_string:
            detection_methods.append("string search")

        combined_results.append({
            "library_name": lib_name,
            "detected_by": detection_methods
        })

    detection_output_file = os.path.join('detection', apk_name, f"{apk_name}_detected_libs.json")
    os.makedirs(os.path.dirname(detection_output_file), exist_ok=True)
    with open(detection_output_file, "w") as f:
        json.dump(combined_results, f, indent=4)


def get_json(json_path):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return None

def load_model(device, lib_name):
    model = ASTNN().to(device)
    try:
        model.load_state_dict(torch.load(f'libraries/{lib_name}/{lib_name}_gnn_weights.pt', map_location=device))
        return model
    except FileNotFoundError:
        print(f"No model has been trained for library: {lib_name}")
        return None

def filter_dictionaries(apk_name, gt_dict, sizes_dict, files_dict, output_file):
    """
    Filters the sizes_dict and files_dict based on relevant libraries and their basenames,
    adding additional libraries with the same basename and random entries if necessary.
    """
    # Check if the apk_name is in gt_dict
    if apk_name in gt_dict:
        # Get the list of relevant keys from gt_dict
        relevant_keys = set(gt_dict[apk_name])
        relevant_basenames = {key.rsplit("-", 1)[0] for key in relevant_keys}
        
        # Initialize filtered dictionaries
        filtered_sizes_dict = {}
        filtered_files_dict = {}

        # Add libraries matching relevant keys and their basenames
        for key in sizes_dict:
            base_name = key.rsplit("-", 1)[0]
            if key in relevant_keys or base_name in relevant_basenames:
                filtered_sizes_dict[key] = sizes_dict[key]
                filtered_files_dict[key] = files_dict[key]

        # Determine additional libraries needed
        already_added = set(filtered_sizes_dict.keys())
        remaining_keys = set(sizes_dict.keys()) - already_added
        additional_keys_needed = max(0, 5 - len(already_added))

        # Add random entries to fill up to 5
        if additional_keys_needed > 0:
            random_keys = random.sample(remaining_keys, min(additional_keys_needed, len(remaining_keys)))
            for key in random_keys:
                filtered_sizes_dict[key] = sizes_dict[key]
                filtered_files_dict[key] = files_dict[key]

        # Track final libraries added for this APK
        final_libs = list(filtered_sizes_dict.keys())

        # Update or create the cumulative JSON file
        cumulative_data = {}
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                cumulative_data = json.load(f)

        cumulative_data[apk_name] = final_libs

        with open(output_file, "w") as f:
            json.dump(cumulative_data, f, indent=4)

        return filtered_sizes_dict, filtered_files_dict
    else:
        return None, None


def scan_apk(sizes_dict, files_dict, device, groundtruth = False):
    results = {}
    processed_dir = 'data/dexs/processed'
    if groundtruth is True:
        gt_dict = get_json('groundtruth.json')

    os.makedirs(processed_dir, exist_ok=True)
    dirs = [dir for dir in os.listdir('data/dexs') if os.path.isdir(os.path.join('data/dexs', dir)) and dir not in ['workspace', 'processed']]
    for dir in dirs:
        dir_path = os.path.join('data', 'dexs', dir)
        detected_libs_model = []
        detected_libs_string = []
        apk_name = os.path.basename(dir_path)
        if gt_dict:
            output_file = os.path.join("all_apks_libs_scanned.json")
            sizes_dict, files_dict = filter_dictionaries(apk_name, gt_dict, sizes_dict, files_dict, output_file)
        mvn_google_json = get_json(os.path.join(dir_path, f'{apk_name}_results.json'))
        
        out_dirs = process_lib_files(dir_path, detecting=True)
        obfuscated = mvn_google_json is None

        print(f"Scanning APK: {apk_name}, Obfuscated: {obfuscated}")

        for lib_name in sizes_dict.keys():
            print(f"Trying to detect {lib_name}")
            detection_dataset_root = os.path.join('detection', apk_name, lib_name)
            classed_dot_files = []
            labels = []
            search_string = lib_name.rsplit("-", 1)[0]
            lib_number = lib_name.rsplit("-", 1)[-1]
            lib_base_name = lib_name.rsplit("-", 1)[0]
            max_size = max(sizes_dict[lib_name], key=lambda x: x[0])[0] + 1
            min_size = min(sizes_dict[lib_name], key=lambda x: x[0])[0] - 6
            lib_dot_dir = os.path.join('data', 'jars', lib_base_name, lib_number)

            if not obfuscated:
                lib_apk_files = []
                dot_dirs = [dir for dir, name in out_dirs]
                lib_apk_files = find_lib_dot_files(dot_dirs, search_string, max_size, min_size)

                if lib_apk_files:
                    detected_libs_string.append(lib_name)
                    if not os.path.exists(os.path.join(detection_dataset_root, 'ast_dataset', 'processed', 'data.pt')):
                        lib_graph_classes = get_graph_names_and_indexes(lib_dot_dir)  
                    
                        for graph_name, index in lib_graph_classes:
                            for dot_file in lib_apk_files:
                                apk_graph_name = extract_graph_name(dot_file)
                                if apk_graph_name == graph_name:
                                    classed_dot_files.append(dot_file)
                                    labels.append(index + 1)

                        if classed_dot_files:
                            method_graphs = ast.create_dataset_from_dot_files(classed_dot_files, (apk_name, lib_name), labels = labels)
                            model = load_model(device, lib_name)
                            lib_vector = expand_library_dataset_from_apk(method_graphs, model, lib_name)

                        else:
                            sizes = sizes_dict[lib_name]
                            filter_dict = load_dictionary(os.path.join('libraries', lib_name, 'classes_first_children.json'))
                            dot_files, label_mapping = get_dot_files_with_labels(sizes, filter_dict, dot_files=lib_apk_files)
                            if dot_files:
                                method_graphs = ast.create_dataset_from_dot_files(dot_files, (apk_name, lib_name), labels=list(label_mapping.keys()))
                                file_mapping = files_dict[lib_name]
                                model = load_model(device, lib_name)
                                lib_vector = detect_library_in_apk(method_graphs, model, lib_name, file_mapping)
                            else:
                                continue
                        if sum(lib_vector[1:]) > 0:
                            detected_libs_model.append((lib_name, lib_vector))

                            
                    else:
                        detected_libs_string.append(lib_name)
                        try:
                            method_graphs = ast.CustomASTDataset.load_dataset_from_root(os.path.join(detection_dataset_root, 'ast_dataset'))
                            file_mapping = files_dict[lib_name]
                            model = load_model(device, lib_name)
                            lib_vector = detect_library_in_apk(method_graphs, model, lib_name, file_mapping)
                        except FileNotFoundError as e:
                            print(e)
                    if sum(lib_vector[1:]) > 0:
                        detected_libs_model.append((lib_name, lib_vector))
                else:
                    obfuscated = True

            if obfuscated is True:
                if not os.path.exists(os.path.join(detection_dataset_root, 'ast_dataset', 'processed', 'data.pt')): 
                    try:
                        sizes = sizes_dict[lib_name]
                        filter_dict = load_dictionary(os.path.join('libraries', lib_name, 'classes_first_children.json'))
                        directories = [dir for dir, name in out_dirs]
                        collected_dot_files = []
                        for directory in directories:
                            dot_files = glob.glob(os.path.join(directory, '*.dot'))
                            collected_dot_files.extend(dot_files)
                        dot_files, label_mapping = get_dot_files_with_labels(sizes, filter_dict, dot_files=collected_dot_files)
                        file_mapping = files_dict[lib_name]

                        if dot_files:
                            method_graphs = ast.create_dataset_from_dot_files(dot_files, (apk_name, lib_name), list(label_mapping.keys()))
                            save_mapping(label_mapping, detection_dataset_root, 'label_mapping.json')
                        else:
                            continue

                    except Exception as e:
                        print(f"Error processing library {lib_name}: {e}")
                else:
                    try:
                        method_graphs = ast.CustomASTDataset.load_dataset_from_root(os.path.join(detection_dataset_root, 'ast_dataset'))
                        label_mapping_path = os.path.join(detection_dataset_root, 'label_mapping.json')
                        file_mapping_path = os.path.join(detection_dataset_root, 'file_mapping.json')
                        label_mapping = load_dictionary(label_mapping_path)
                        file_mapping = files_dict[lib_name]
                    except FileNotFoundError as e:
                        print(e)
                        continue
                model = load_model(device, lib_name)
                lib_vector = detect_library_in_apk(method_graphs, model, lib_name, file_mapping, label_mapping)

                if sum(lib_vector[1:]) > 0:
                    detected_libs_model.append((lib_name, lib_vector))


        best_version = {}
        for lib_name, lib_vector in detected_libs_model:
            base_name = lib_name.rsplit("-", 1)[0]
            vector = lib_vector[1:]
            if base_name not in best_version or sum(1 for v in vector[1:] if v > 0) > sum(1 for v in best_version[base_name]["vector"] if v > 0):
                best_version[base_name] = {"name": lib_name, "vector": vector}

        valid_queries = {}
        if mvn_google_json:
            valid_queries = mvn_google_json.get("valid_queries", {})

        output_file = os.path.join("all_apks_detected_libs.json")
        output_results_json(output_file, apk_name, best_version, valid_queries, detected_libs_string)


        print(f"Detection results for '{apk_name}' saved to {output_file}")

        destination_dir = os.path.join(processed_dir, apk_name)
        shutil.move(dir_path, destination_dir)
        print(f"Moved processed directory '{dir_path}' to '{destination_dir}'")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
sizes_dict, files_dict = collect_file_sizes()
dir = 'data/dexs'

scan_apk(sizes_dict, files_dict, device, groundtruth = True)
