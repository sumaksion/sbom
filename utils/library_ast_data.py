import os
import torch
import glob
import json
from torch_geometric.data import InMemoryDataset, Data
from sentence_transformers import SentenceTransformer
import networkx as nx

class CustomASTDataset(InMemoryDataset):
    def __init__(self, root, dot_files=None, transform=None, pre_transform=None, eval_dataset=True, dict_path='node_type_dict.json'):
        self.dot_files = dot_files
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  
        self.eval_dataset = eval_dataset
        self.dict_path = dict_path
        self.node_type_dict = self.load_node_type_dict()
        super().__init__(root, transform, pre_transform)

        os.makedirs(self.processed_dir, exist_ok=True)

        if not self._preprocessed_dataset_exists():
            self.data, self.slices = self.process_dot_files()
            self.save_dataset()
        else:
            self.data, self.slices = torch.load(self.processed_paths[0])

    def _preprocessed_dataset_exists(self):
        return os.path.exists(self.processed_paths[0])

    def process_dot_files(self):
        data_list = [self.load_and_convert(idx, dot_file) for idx, dot_file in enumerate(self.dot_files)]
        return self.collate(data_list)

    def save_dataset(self):
        torch.save((self.data, self.slices), self.processed_paths[0])

    def load_and_convert(self, idx, dot_file):
        G = nx.drawing.nx_agraph.read_dot(dot_file)
        root = [node for node, degree in G.in_degree() if degree == 0][0]
        first_layer_children = list(G.successors(root))
        subgraph_nodes = [root] + first_layer_children
        G = G.subgraph(subgraph_nodes).copy()

        G = nx.convert_node_labels_to_integers(G)
        data = self.networkx_to_torch_geometric(G, label=idx + 1)
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
                self.node_type_dict[node_type] = len(self.node_type_dict)
            encoded_types.append(self.node_type_dict[node_type])
        self.save_node_type_dict()
        return encoded_types

    def networkx_to_torch_geometric(self, G, label):
        edge_index = torch.tensor(list(G.edges(data=False)), dtype=torch.long).t().contiguous()
        node_labels = [attr_dict.get('label', '') for _, attr_dict in G.nodes(data=True)]
        x = torch.tensor(self.feature_embedding(node_labels), dtype=torch.float)
        #node_types = [lbl.split(',', 1)[0] for lbl in node_labels]
        #x = torch.tensor(self.encode_node_types(node_types), dtype=torch.float)
        #x = x.unsqueeze(1)
        return Data(x=x, edge_index=edge_index, y=label)

    def feature_embedding(self, sentences):
        embeddings = self.model.encode(sentences, show_progress_bar=False)
        return embeddings

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def get(self, idx):
        # Retrieve data using default slicing logic but explicitly copy the label
        data = super().get(idx)  # Call the base class method
        data.y = self.data.y[self.slices['y'][idx]:self.slices['y'][idx + 1]].clone()
        return data

    def add_class_0(self, data_list):
        """
        Adds graphs from other datasets, modifies their label to 0, and adds them to the current dataset.
        
        Parameters:
        - data_list: A list of `Data` objects representing graphs to be added as class 0.
        """
        # Convert self.data and self.slices back to a list of Data objects
        existing_data = self._unpack_data()
        
        # Modify the label of new graphs and add them to the list
        for data in data_list:
            if isinstance(data, Data):
                print(data.y)
                data.y = torch.tensor([0], dtype=torch.long)
                print(data.y)
                existing_data.append(data)
        
        # Re-collate the data and slices
        self.data, self.slices = self.collate(existing_data)
        
        # Save the updated dataset
        self.save_dataset()

    def _unpack_data(self):
        """
        Unpacks the tensor data and slices into a list of Data objects.
        """
        return [self.get(i) for i in range(len(self))]


def create_dataset_from_dot(directory, top_n, lib_name, eval=True):
    dot_files = glob.glob(os.path.join(directory, '*.dot'))
    dot_files = sorted(dot_files, key=os.path.getsize, reverse=True)[:top_n]
    
    lib_path = f'detection/{lib_name}' if eval else f'libraries/{lib_name}'
    os.makedirs(lib_path, exist_ok=True)
    
    sizes_path = os.path.join(lib_path, 'sizes.txt')
    with open(sizes_path, 'w') as f:
        file_sizes = []
        for dot_file in dot_files:
            size = os.path.getsize(dot_file)
            file_sizes.append(size)
            f.write(f"{os.path.basename(dot_file)}: {size} bytes\n")
        
        f.write(f"\nmax_size: {max(file_sizes)} bytes\n")
        f.write(f"min_size: {min(file_sizes)} bytes\n")
    
    dataset = CustomASTDataset(root=f'{lib_path}/ast_dataset', dot_files=dot_files, eval_dataset=eval)
    return dataset
