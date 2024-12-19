import os
from model2vec import StaticModel
import torch
import glob
import json
import re
from multiprocessing import Pool, cpu_count
import math
from torch_geometric.data import InMemoryDataset, Data
from sentence_transformers import SentenceTransformer
import networkx as nx
from utils import text_stripper as strip 

class CustomASTDataset(InMemoryDataset):
    def __init__(self, root, dot_files=None, transform=None, pre_transform=None, labels = None, dict_path='node_type_dict.json'):
        if dot_files is None:
            dot_files = []
        if labels is not None:
            self.dot_files = [(dot_file, label) for dot_file, label in zip(dot_files, labels)]
        else:
            self.dot_files = [(dot_file, idx + 1) for idx, dot_file in enumerate(dot_files)]
        #self.model = SentenceTransformer("all-MiniLM-L6-v2")  
        self.model = StaticModel.from_pretrained("minishlab/potion-base-2M")
        self.dict_path = dict_path
        self.node_type_dict = self.load_node_type_dict()
        self.labels = labels
        super().__init__(root, transform, pre_transform)

        os.makedirs(self.processed_dir, exist_ok=True)
        
        self.preprocessed_dir = os.path.join(self.root, 'preprocessed')
        os.makedirs(self.preprocessed_dir, exist_ok=True)

        if not self._preprocessed_dataset_exists():
            self.preprocess_dot_files()
            self.data, self.slices = self.process_dot_files()
            self.save_dataset()
        else:
            self.data, self.slices = torch.load(self.processed_paths[0])

    def _preprocessed_dataset_exists(self):
        return os.path.exists(self.processed_paths[0])

    def process_dot_files(self):
        data_list = [self.load_and_convert(dot_file, label) for dot_file, label in self.dot_files]
        return self.collate(data_list)

    def save_dataset(self):
        torch.save((self.data, self.slices), self.processed_paths[0])

    def preprocess_dot_files(self):
        preprocessed_files = []
        for dot_file, class_label in self.dot_files:
            preprocessed_path = os.path.join(self.preprocessed_dir, os.path.basename(dot_file))
            if not os.path.exists(preprocessed_path):
                strip.process_graph_file(dot_file, preprocessed_path)
            preprocessed_files.append((preprocessed_path, class_label))
        self.dot_files = preprocessed_files


    def load_and_convert(self, dot_file, label):
        G = nx.drawing.nx_agraph.read_dot(dot_file)
        """root = [node for node, degree in G.in_degree() if degree == 0][0]
        first_layer_children = list(G.successors(root))
        subgraph_nodes = [root] + first_layer_children
        G = G.subgraph(subgraph_nodes).copy()"""
        G = nx.convert_node_labels_to_integers(G)
        data = self.networkx_to_torch_geometric(G, label=label)
        return data
    
    def load_node_type_dict(self):
        if os.path.exists(self.dict_path):
            with open(self.dict_path, 'r') as f:
                return json.load(f)
        else:
            return {}

    def save_node_type_dict(self):
        with open(self.dict_path, 'w') as f:
            json.dump(self.node_type_dict, f)

    def encode_node_types(self, types):
        encoded_types = []
        for node_type in types:
            if node_type not in self.node_type_dict:
                self.node_type_dict[node_type] = len(self.node_type_dict) + 1
            encoded_types.append(self.node_type_dict[node_type])
        self.save_node_type_dict()
        return encoded_types

    def networkx_to_torch_geometric(self, G, label):
        edge_index = torch.tensor(list(G.edges(data=False)), dtype=torch.long).t().contiguous()
        node_labels = [attr_dict.get('label', '') for _, attr_dict in G.nodes(data=True)]
        x = torch.tensor(self.feature_embedding(node_labels), dtype=torch.float)
        y = torch.tensor([label], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, y=y)

    def feature_embedding(self, sentences):
        embeddings = self.model.encode(sentences)
        return embeddings

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def replace_class(self, dot_file, class_label):
        data_list = self._unpack_data()
        class_label_tensor = torch.tensor([class_label], dtype=torch.long)
        new_data_list = [data for data in data_list if data.y.item() != class_label_tensor]
        
        lib_dir = os.path.dirname(self.root)
        sizes_path = os.path.join(lib_dir, 'sizes.json')

        if os.path.exists(sizes_path):
            with open(sizes_path, 'r') as f:
                sizes_data = json.load(f)
        else:
            print("Can't replace class file in empty dataset!")
            return

        # Find and remove the existing file associated with the class
        if str(class_label) in sizes_data:
            existing_file = sizes_data[str(class_label)]['file']
            preprocessed_path = os.path.join(self.preprocessed_dir, existing_file)
            if os.path.exists(preprocessed_path):
                os.remove(preprocessed_path)

        # Preprocess the new dot file
        preprocessed_path = os.path.join(self.preprocessed_dir, os.path.basename(dot_file))
        if not os.path.exists(preprocessed_path):
            strip.process_graph_file(dot_file, preprocessed_path)

        # Load and convert the new data
        new_data = self.load_and_convert(preprocessed_path, class_label)
        if isinstance(new_data, Data):
            new_data_list.append(new_data)
        else:
            print("Something went wrong")
            return

        self.data, self.slices = self.collate(new_data_list)
        self.save_dataset()

        # Update classes_first_children.json
        classes_first_children_path = os.path.join(lib_dir, 'classes_first_children.json')
        if os.path.exists(classes_first_children_path):
            with open(classes_first_children_path, 'r') as f:
                class_data = json.load(f)
        else:
            print("Can't replace class file in empty dataset!")
            return

        strings_list = strip.get_ast_first_children(dot_file)
        class_data[str(class_label)] = strings_list

        with open(classes_first_children_path, 'w') as f:
            json.dump(class_data, f, indent=4)

        # Update sizes.json
        size = os.path.getsize(dot_file)
        sizes_data[str(class_label)] = {'size': size, 'file': os.path.basename(dot_file)}

        with open(sizes_path, 'w') as f:
            json.dump(sizes_data, f, indent=4)

    
    def add_class_0(self, data_list):
        labels = [0] * len(data_list)
        self.add_data(data_list, labels)

    def add_data(self, data_list, labels):
    
        existing_data = self._unpack_data()

        for idx, data in enumerate(data_list):
            if isinstance(data, Data):
                label = labels[idx]
                data.y = torch.tensor([label], dtype=torch.long)
                existing_data.append(data)
        

        self.data, self.slices = self.collate(existing_data)
        self.remove_duplicates()
        self.save_dataset()

    def _unpack_data(self):
        return [self.get(i) for i in range(len(self))]
    
    @classmethod
    def load_dataset_from_root(cls, root):
        if os.path.exists(os.path.join(root, 'processed', 'data.pt')):
            lib_dir = os.path.dirname(root)
            sizes_path = os.path.join(lib_dir, 'sizes.json')
            instance = cls(root=root)
            if os.path.exists(sizes_path):
                with open(sizes_path, 'r') as f:
                    sizes_data = json.load(f)

                dot_files = []
                for class_label, file_info in sizes_data.items():
                    dot_file = os.path.join(root, 'preprocessed', file_info['file'])
                    dot_files.append((dot_file, int(class_label)))

                instance.dot_files = dot_files
                return instance
            else:
                return instance
        else:
            raise FileNotFoundError(f"No processed dataset found in {root}.")

    def remove_duplicates(self):

        unique_data = []
        unique_hashes = set()

        for data in self._unpack_data():
            data_hash = (data.x.numpy().tobytes(), data.edge_index.numpy().tobytes(), data.y.item())
            if data_hash not in unique_hashes:
                unique_data.append(data)
                unique_hashes.add(data_hash)

        self.data, self.slices = self.collate(unique_data)
        self.save_dataset()
    
def create_dataset_from_dir(directory, top_n, lib_name, eval=True):
    lib_path = f'detection/{lib_name}' if eval else f'libraries/{lib_name}'
    lib_data_path = os.path.join(lib_path, 'ast_dataset', 'processed', 'data.pt')
    if os.path.exists(lib_data_path):
        dataset = CustomASTDataset.load_dataset_from_root(os.path.join(lib_path, 'ast_dataset'))
        return dataset
    dot_files = glob.glob(os.path.join(directory, '*.dot'))
    dot_files = sorted(dot_files, key=os.path.getsize, reverse=True)[:top_n]

    lib_path = f'detection/{lib_name}' if eval else f'libraries/{lib_name}'
    os.makedirs(lib_path, exist_ok=True)

    dataset = CustomASTDataset(root=f'{lib_path}/ast_dataset', dot_files=dot_files)

    classes_first_children_path = os.path.join(lib_path, 'classes_first_children.json')
    classes_first_children = {}

    for idx, dot_file in enumerate(dot_files, start=1):
        strings_list = strip.get_ast_first_children(dot_file)
        classes_first_children[str(idx)] = strings_list

    with open(classes_first_children_path, 'w') as f:
        json.dump(classes_first_children, f, indent=4)

    sizes_path = os.path.join(lib_path, 'sizes.json')
    sizes_data = {str(idx + 1): {'size': os.path.getsize(dot_file), 'file': os.path.basename(dot_file)} for idx, dot_file in enumerate(dot_files)}

    with open(sizes_path, 'w') as f:
        json.dump(sizes_data, f, indent=4)

    return dataset


def create_dataset_from_dot_files(dot_files, name, labels=None, eval=True):
    lib_path = f'detection/{name[0]}/{name[1]}' if eval else f'libraries/{name[0]}'
    os.makedirs(lib_path, exist_ok=True)

    dataset = CustomASTDataset(root=f'{lib_path}/ast_dataset', dot_files=dot_files, labels=labels)
    return dataset

