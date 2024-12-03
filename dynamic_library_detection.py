import torch
import re
import os
from torch_geometric.loader import DataLoader
from models.gcn import GCN
from utils import library_ast_data as ast
from utils import jar_processing as jar

def detect_library_in_apk(apk_ast_data, model, batch_size=64, num_classes=16, threshold=0.9):
    """model = GCN(hidden_channels=16)
    model.load_state_dict(torch.load(f'libraries/{library_name}/{library_name}_gnn_weights.pt'))
    model.eval()"""
    
    library_vector = [0] * num_classes
    
    eval_loader = DataLoader(apk_ast_data, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for data in eval_loader:

            out = model(data.x, data.edge_index, data.batch)
            #print(out.pred)
            probabilities = torch.softmax(out, dim=1)
            
            for prob in probabilities:
                certainty, pred = torch.max(prob, dim=0)  
                #print(f"Data object: Prediction: {pred.item()}, Probability: {certainty.item():.4f}")
                if certainty > threshold:
                    library_vector[pred.item()] += 1  
    
    return library_vector

def find_min_max_file_size(base_directory='libraries'):
    min_size = None
    max_size = None
    
    for root, _, files in os.walk(base_directory):
        for file in files:
            if file == "sizes.txt":
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()[::-1]  # Read all lines and reverse them
                        for line in lines:
                            if line.startswith("min_size:"):
                                min_size = int(line.split(":")[1].strip().split()[0])
                            elif line.startswith("max_size:"):
                                max_size = int(line.split(":")[1].strip().split()[0])
                            # Break early if both are found
                            if min_size is not None and max_size is not None:
                                break
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return min_size, max_size

def count_files_min_size(directory, min_size):

    count = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                file_size = os.path.getsize(file_path)
                if file_size >= min_size:
                    count += 1
            except Exception as e:
                print(f"Error getting size of {file_path}: {e}")
    
    return count

dir = 'data/dexs'
joern_workspace = os.path.join(dir, 'workspace')
all_datasets = {}

if not os.path.exists(joern_workspace):
    os.mkdir(joern_workspace)

out_dirs_apk_names = jar.process_lib_files(dir, joern_workspace=joern_workspace, detecting=True)
min_size, max_size = find_min_max_file_size()

for out_dir, apk_name in out_dirs_apk_names:
    print(f"Creating dataset for library '{apk_name}' with data from '{out_dir}'...")
    count = count_files_min_size(out_dir, min_size)
    method_graphs = ast.create_dataset_from_dot(out_dir, count, apk_name, eval=True)
    all_datasets[apk_name] = method_graphs

def init_models(directory='libraries'):
    model = GCN(hidden_channels=16)
    models = {}
    try:
        for name in os.listdir(directory):
            if os.path.isdir(os.path.join(directory, name)):
                model.load_state_dict(torch.load(f'libraries/{name}/{name}_gnn_weights.pt'))
                models[name] = model
        return models
    except FileNotFoundError:
        print(Exception)
        return models
    except PermissionError:
        print(f"Permission denied to access '{directory}'.")
        return models
    
models = init_models()

for apk, ast_data in all_datasets.items():
    for name, model in models.items():
        v = detect_library_in_apk(ast_data, model)
        print(name + " in " + apk)
        print(v)
