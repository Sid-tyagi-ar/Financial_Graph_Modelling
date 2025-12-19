import os
import torch
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Geometry import Point3D
from rdkit import RDLogger
import imageio
import networkx as nx
import numpy as np
import rdkit.Chem
import wandb
import matplotlib
# matplotlib.use("macOSX")
import matplotlib.pyplot as plt





class MolecularVisualization:
    def __init__(self, remove_h, dataset_infos):
        self.remove_h = remove_h
        self.dataset_infos = dataset_infos

    def mol_from_graphs(self, node_list, adjacency_matrix):
        """
        Convert graphs to rdkit molecules
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        # dictionary to map integer value to the char of atom
        atom_decoder = self.dataset_infos.atom_decoder

        # create empty editable mol object
        mol = Chem.RWMol()

        # add atoms to mol and keep track of index
        node_to_idx = {}
        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            a = Chem.Atom(atom_decoder[int(node_list[i])])
            molIdx = mol.AddAtom(a)
            node_to_idx[i] = molIdx

        for ix, row in enumerate(adjacency_matrix):
            for iy, bond in enumerate(row):
                # only traverse half the symmetric matrix
                if iy <= ix:
                    continue
                if bond == 1:
                    bond_type = Chem.rdchem.BondType.SINGLE
                elif bond == 2:
                    bond_type = Chem.rdchem.BondType.DOUBLE
                elif bond == 3:
                    bond_type = Chem.rdchem.BondType.TRIPLE
                elif bond == 4:
                    bond_type = Chem.rdchem.BondType.AROMATIC
                else:
                    continue
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

        try:
            mol = mol.GetMol()
        except rdkit.Chem.KekulizeException:
            print("Can't kekulize molecule")
            mol = None
        return mol

    def visualize(self, path: str, molecules: list, num_molecules_to_visualize: int, trainer=None, log='graph'):
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        # visualize the final molecules
        print(f"Visializing {num_molecules_to_visualize} of {len(molecules)}")
        if num_molecules_to_visualize > len(molecules):
            print(f"Shortening to {len(molecules)}")
            num_molecules_to_visualize = len(molecules)
        
        can_log=trainer is not None and hasattr(trainer,"logger") and trainer.logger is not None
        for i in range(num_molecules_to_visualize):
            file_path = os.path.join(path, 'molecule_{}.png'.format(i))
            mol = self.mol_from_graphs(molecules[i][0].numpy(), molecules[i][1].numpy())
            try:
                Draw.MolToFile(mol, file_path)
                if can_log:
                    print(f"Saving {file_path} to wandb")
                    trainer.logger.log_image(key=log, images=[file_path])
            except rdkit.Chem.KekulizeException:
                print("Can't kekulize molecule")


    def visualize_chain(self, path, nodes_list, adjacency_matrix, trainer=None):
        RDLogger.DisableLog('rdApp.*')
        # convert graphs to the rdkit molecules
        mols = [self.mol_from_graphs(nodes_list[i], adjacency_matrix[i]) for i in range(nodes_list.shape[0])]

        # find the coordinates of atoms in the final molecule
        final_molecule = mols[-1]
        AllChem.Compute2DCoords(final_molecule)

        coords = []
        for i, atom in enumerate(final_molecule.GetAtoms()):
            positions = final_molecule.GetConformer().GetAtomPosition(i)
            coords.append((positions.x, positions.y, positions.z))

        # align all the molecules
        for i, mol in enumerate(mols):
            AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer()
            for j, atom in enumerate(mol.GetAtoms()):
                x, y, z = coords[j]
                conf.SetAtomPosition(j, Point3D(x, y, z))

        # draw gif
        save_paths = []
        num_frams = nodes_list.shape[0]

        for frame in range(num_frams):
            file_name = os.path.join(path, 'fram_{}.png'.format(frame))
            Draw.MolToFile(mols[frame], file_name, size=(300, 300), legend=f"Frame {frame}")
            save_paths.append(file_name)

        imgs = [imageio.imread(fn) for fn in save_paths]
        gif_path = os.path.join(os.path.dirname(path), '{}.gif'.format(path.split('/')[-1]))
        imgs.extend([imgs[-1]] * 10)
        imageio.mimsave(gif_path, imgs, subrectangles=True, duration=200)

        can_log=trainer is not None and hasattr(trainer,"logger") and trainer.logger is not None
        if can_log:
            print(f"Saving {gif_path} to wandb")
            trainer.logger.experiment.log({'chain': [wandb.Video(gif_path, caption=gif_path, format="gif")]})

        # draw grid image
        try:
            img = Draw.MolsToGridImage(mols, molsPerRow=10, subImgSize=(200, 200))
            img.save(os.path.join(path, '{}_grid_image.png'.format(path.split('/')[-1])))
        except Chem.rdchem.KekulizeException:
            print("Can't kekulize molecule")
        return mols


class NonMolecularVisualization:
    def __init__(self, dataset_infos):
        self.dataset_infos = dataset_infos

    def _decode_node_features(self, node_feature_vector):
        decoded_features = {}
        feature_names = list(self.dataset_infos.num_node_features.keys())
        for i, feature_name in enumerate(feature_names):
            decoded_features[feature_name] = node_feature_vector[i].item()
        return decoded_features

    def _decode_edge_features(self, edge_feature_vector):
        decoded_features = {}
        feature_names = list(self.dataset_infos.num_edge_features.keys())
        for i, feature_name in enumerate(feature_names):
            decoded_features[feature_name] = edge_feature_vector[i].item()
        return decoded_features

    def to_networkx(self, data):
        graph = nx.Graph()
        num_nodes = data.num_nodes

        for node_idx in range(num_nodes):
            node_feature_vector = data.x[node_idx]
            if len(node_feature_vector) == 0:  # Check if node is masked/padded
                continue
            decoded_node_features = self._decode_node_features(node_feature_vector)
            graph.add_node(node_idx, **decoded_node_features)

        # Add edges using the sparse edge_index
        for i in range(data.edge_index.shape[1]):
            u, v = data.edge_index[0, i].item(), data.edge_index[1, i].item()

            # Avoid adding edges twice for undirected graphs
            if u > v:
                continue

            edge_feature_vector = data.edge_attr[i]
            decoded_edge_features = self._decode_edge_features(edge_feature_vector)

            graph.add_edge(u, v, **decoded_edge_features)

        return graph

    def visualize_non_molecule(self, graph, pos, path, iterations=100, node_size=100, largest_component=False):
        if largest_component:
            CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
            CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
            graph = CGs[0]

        # Plot the graph structure with colors
        if pos is None:
            pos = nx.spring_layout(graph, iterations=iterations, seed=42)

        node_colors = [graph.nodes[node]['category_small_id'] for node in graph.nodes()] # Color by transaction category
        
        # Create a color map for edges based on their type
        edge_color_map = []
        for u, v, data in graph.edges(data=True):
            if data.get('is_root_txn_edge') == 1:
                edge_color_map.append('#7A9E9F')  # Muted Blue/Green for root connections
            elif data.get('time_delta_bin', 0) > 0:
                time_bin = data['time_delta_bin']
                if time_bin == 1: # < 5 min
                    edge_color_map.append('#F94144') # Red - Rapid
                elif time_bin == 2: # 5-60 min
                    edge_color_map.append('#F8961E') # Orange
                elif time_bin == 3: # 1-24 hr
                    edge_color_map.append('#F9C74F') # Yellow
                else: # > 24 hr
                    edge_color_map.append('#90BE6D') # Green
            elif data.get('is_txn_txn_edge') == 1:
                edge_color_map.append('#D3D3D3')  # Light Gray for shared merchant/category
            else:
                edge_color_map.append('#E0E0E0') # Default fallback

        plt.figure(figsize=(10, 8))
        nx.draw(graph, pos, font_size=5, node_size=node_size, with_labels=False,
                node_color=node_colors, cmap=plt.cm.viridis, # Viridis for node type
                edge_color=edge_color_map,
                width=1.5)

        plt.tight_layout()
        plt.savefig(path)
        plt.close("all")

    def visualize(self, path: str, graphs: list, num_graphs_to_visualize: int, log='graph', trainer=None):
        from torch_geometric.data import Data
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        # visualize the final molecules
        for i in range(num_graphs_to_visualize):
            file_path = os.path.join(path, 'graph_{}.png'.format(i))
            node_feat, edge_feat = graphs[i]
            if node_feat.shape[0] == 0:
                continue
            
            num_nodes = node_feat.shape[0]
            adj = (edge_feat.argmax(dim=-1) > 0)
            edge_index = adj.nonzero().t().contiguous()
            sparse_edge_attr = edge_feat[edge_index[0], edge_index[1]]
            
            data_obj = Data(x=node_feat, edge_index=edge_index, edge_attr=sparse_edge_attr, num_nodes=num_nodes)

            graph = self.to_networkx(data_obj)
            self.visualize_non_molecule(graph=graph, pos=None, path=file_path)
            im = plt.imread(file_path)
            wandb.log({log: [wandb.Image(im, caption=file_path)]})

    def visualize_chain(self, path, nodes_list, adjacency_matrix, trainer=None):
        from torch_geometric.data import Data
        # convert graphs to networkx
        graphs = []
        for i in range(nodes_list.shape[0]):
            node_feat = nodes_list[i]
            edge_feat = adjacency_matrix[i]
            if node_feat.shape[0] == 0:
                continue

            num_nodes = node_feat.shape[0]
            adj = (edge_feat.argmax(dim=-1) > 0)
            edge_index = adj.nonzero().t().contiguous()
            sparse_edge_attr = edge_feat[edge_index[0], edge_index[1]]
            
            data_obj = Data(x=node_feat, edge_index=edge_index, edge_attr=sparse_edge_attr, num_nodes=num_nodes)
            graphs.append(self.to_networkx(data_obj))

        # find the coordinates of atoms in the final molecule
        final_graph = graphs[-1]
        final_pos = nx.spring_layout(final_graph, seed=0)

        # draw gif
        save_paths = []
        num_frams = nodes_list.shape[0]

        for frame in range(num_frams):
            file_name = os.path.join(path, 'fram_{}.png'.format(frame))
            self.visualize_non_molecule(graph=graphs[frame], pos=final_pos, path=file_name)
            save_paths.append(file_name)

        imgs = [imageio.imread(fn) for fn in save_paths]
        gif_path = os.path.join(os.path.dirname(path), '{}.gif'.format(path.split('/')[-1]))
        imgs.extend([imgs[-1]] * 10)
        imageio.mimsave(gif_path, imgs, subrectangles=True, duration=200)
        wandb.log({'chain': [wandb.Video(gif_path, caption=gif_path, format="gif")]})
        return
