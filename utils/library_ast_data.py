import os
import glob
import networkx as nx
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import TUDataset
import torch
import matplotlib.pyplot as plt

class CustomASTDataset(InMemoryDataset):
    def __init__(self, root, dot_files, transform=None, pre_transform=None):
        self.dot_files = dot_files
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = self.process_dot_files()

    def visualize_graph(G, color=1):
        plt.figure(figsize=(7,7))
        plt.xticks([])
        plt.yticks([])
        nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                         
                        node_color=color, cmap="Set2")
        plt.show()


    def process_dot_files(self):
        data_list = []
        for dot_file in self.dot_files:
            # Read the .dot file as a NetworkX graph
            G = nx.drawing.nx_agraph.read_dot(dot_file)
            
            G = nx.relabel_nodes(G, {node: int(node) for node in G.nodes()})
        
        
            """edges = [(int(u), int(v)) for u, v in G.edges()]

            G_cleaned = nx.Graph()
            G_cleaned.add_edges_from(edges)
            print(G_cleaned.edges())
            print(G_cleaned.nodes())"""
            data = self.networkx_to_torch_geometric(G, dot_file)
            data_list.append(data)
            
        return self.collate(data_list)

    def networkx_to_torch_geometric(self, G, label):
        
        #print(list(G.edges))
        edge_index = torch.tensor(list(G.edges(data=False)),dtype=torch.long).t().contiguous()
        #print(edge_index)
        x = G.nodes(data=True)

        return Data(x=x, edge_index=edge_index, y=label)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        pass

def create_dataset_from_dot(directory, top_n):

    dot_files = glob.glob(os.path.join(directory, '*.dot'))
    dot_files = sorted(dot_files, key=os.path.getsize, reverse=True)[:top_n]
    lib_name = 'Picasso-2.5.2'
    dataset = CustomASTDataset(root=f'libraries/{lib_name}/ast_dataset', dot_files=dot_files)
    return dataset
directory = 'data/jars/workspace/picasso/out1'

#dataset = create_dataset_from_dot(directory)
#print(f"Number of graphs in dataset: {len(dataset)}")
#print(f"First graph:\n{dataset[0]}")
