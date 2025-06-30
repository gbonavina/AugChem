import os
from rdkit import RDLogger
from rdkit import Chem
import numpy as np
import pandas as pd
from typing import List
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader
from pathlib import Path

from augchem.modules.smiles.smiles_modules import augment_dataset as augment_smiles_dataset
from augchem.modules.graph.graphs_modules import augment_dataset as augment_graph_dataset  


# disable rdkit warnings
RDLogger.DisableLog('rdApp.*')

class QM9Loader:
    def __init__(self, path: Path):
        self.path = path

    def load_qm9_xyz(self, file_path):
        """Load a single QM9.xyz file."""
        with open(file_path, 'r') as f:
            # Number of atoms
            natoms = int(f.readline())
            # Properties are in the second line
            properties = list(map(float, f.readline().split()[2:]))
            # Read atomic coordinates and types
            atoms = []
            coordinates = []
            smiles1 = ''
            smiles2 = '' 
            inchi1 = ''
            inchi2 = ''
            # print(properties)
            for num_line, line in enumerate(f):
                # print(num_line, line)
                if num_line >= 0 and num_line < natoms:
                    info = line.replace("*^","e").split()
                    atoms.append(info[0])
                    coordinates.append(list(map(float, info[1:-1])))
                if num_line == natoms + 1:
                    smiles1, smiles2 = line.split()
                if num_line == natoms + 2:
                    inchi1, inchi2 = line.split()
            
        return {
            "natoms": natoms,
            "atoms": atoms,
            "coordinates": np.array(coordinates),
            "smiles_1": smiles1,
            "smiles_2": smiles2,
            "inchi_1": inchi1,
            "inchi_2": inchi2,
            "properties": properties
        }

    def load_qm9_dataset(self, directory_path, list_mols=[]):
        """Load the entire QM9 dataset from a directory containing .xyz files."""
        X = []
        Y = []
        S = []
        SMILES1 = []
        SMILES2 = []
        INCHI1 = []
        INCHI2 = []

        for file_name in os.listdir(directory_path):
            if file_name.endswith(".xyz"):
                file_path = os.path.join(directory_path, file_name)
                molecule_data = self.load_qm9_xyz(file_path)
                if molecule_data['natoms'] in list_mols or len(list_mols)==0:
                    X.append([molecule_data['atoms'], molecule_data['coordinates']])
                    Y.append(molecule_data['properties'])
                    S.append(molecule_data['natoms'])
                    SMILES1.append(molecule_data['smiles_1'])
                    SMILES2.append(molecule_data['smiles_2'])
                    INCHI1.append(molecule_data['inchi_1'])
                    INCHI2.append(molecule_data['inchi_2'])

        return X, Y, S, SMILES1, SMILES2, INCHI1, INCHI2
    
    def qm9_to_csv(self, qm9_folder: Path, property: int):
        XYZ, Y, natoms, SMILES_1, SMILES_2, INCHI_1, INCHI_2 = self.load_qm9_dataset(qm9_folder)

        z = Y.tolist()

        all_properties = []

        for i, propriedades in enumerate(z):
            propriedade = propriedades[property]

            all_properties.append(propriedade)

        dataframe = pd.DataFrame(
            {
                'SMILES_1': SMILES_1,
                'SMILES_2': SMILES_2,
                'INCHI_1': INCHI_1,
                'INCHI_2': INCHI_2,
                'Property_0': all_properties
            }
        )

        dataframe.to_csv('QM9.csv', index=True, float_format='%.8e')

class PCQM4mv2Loader:
    def __init__(self, sdf_path: str, transform=None):
        """
        Inicializa o carregador de SDF
        
        Args:
            sdf_path: Caminho para o arquivo SDF
            transform: Transformações do torch_geometric a serem aplicadas
        """
        self.sdf_path = Path(sdf_path)
        self.molecules = []
        self.graphs = []
        self.transform = transform
        
    def load_sdf(self, max_molecules: int = 10000) -> List[Chem.Mol]:
        """
        Carrega moléculas do arquivo SDF
        
        Args:
            max_molecules: Número máximo de moléculas para carregar
        
        Returns:
            Lista de moléculas RDKit
        """
        print(f"Carregando moléculas de: {self.sdf_path}")
        
        if not self.sdf_path.exists():
            raise FileNotFoundError(f"Arquivo SDF não encontrado: {self.sdf_path}")
        
        supplier = Chem.SDMolSupplier(str(self.sdf_path))
        molecules = []
        counter = 0
        
        for i, mol in enumerate(supplier):
            if mol is not None:
                molecules.append(mol)
                counter += 1
                
                if i % 1000 == 0:
                    print(f"Carregadas {counter} moléculas...")
                
                if counter >= max_molecules:
                    print(f"Limite de {max_molecules} moléculas atingido, parando o carregamento.")
                    break
            else:
                if i % 1000 == 0:
                    print(f"Molécula {i} é None, pulando...")
        
        self.molecules = molecules
        print(f"Total de moléculas carregadas: {len(molecules)}")
        return molecules
    

        """
        Converte uma molécula RDKit em um objeto torch_geometric.data.Data
        Usando apenas funcionalidades do torch_geometric
        
        Args:
            mol: Molécula RDKit
            target: Tensor alvo (propriedade) para a molécula
            
        Returns:
            Objeto Data do PyTorch Geometric
        """
        # Características dos átomos usando torch
        atom_features = []
        for atom in mol.GetAtoms():
            features = torch.tensor([
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetNumRadicalElectrons(),
                atom.GetTotalNumHs(),
                int(atom.IsInRing()),
                atom.GetMass(),
            ], dtype=torch.float)
            atom_features.append(features)
        
        # Stack dos features dos átomos
        x = torch.stack(atom_features, dim=0) if atom_features else torch.empty((0, 9))
        
        # Obter as arestas (ligações) como tensores
        edge_indices = []
        edge_attrs = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Adicionar aresta em ambas as direções (grafo não direcionado)
            edge_indices.extend([i, j])
            edge_indices.extend([j, i])
            
            # Características da ligação como tensor
            bond_features = torch.tensor([
                bond.GetBondTypeAsDouble(),
                int(bond.GetIsAromatic()),
                int(bond.IsInRing()),
                int(bond.GetIsConjugated()),
            ], dtype=torch.float)
            
            edge_attrs.append(bond_features)
            edge_attrs.append(bond_features)  # Para ambas as direções
        
        # Converter para tensores do torch_geometric
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).view(2, -1)
            edge_attr = torch.stack(edge_attrs, dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 4), dtype=torch.float)
        
        # Criar objeto Data do torch_geometric
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=mol.GetNumAtoms()
        )
        
        # Adicionar target se fornecido
        if target is not None:
            data.y = target if isinstance(target, torch.Tensor) else torch.tensor([target], dtype=torch.float)
        
        # Aplicar transformações se especificadas
        if self.transform:
            data = self.transform(data)
        
        return data

        """
        Carrega grafos torch_geometric de um arquivo
        
        Args:
            file_path: Caminho do arquivo
            
        Returns:
            Lista de grafos carregados
        """
        # Usar weights_only=False para carregar objetos torch_geometric Data
        graphs = torch.load(file_path, weights_only=False)
        print(f"Grafos torch_geometric carregados de: {file_path}")
        return graphs

class Augmentator:        
    """
    Main class for molecular data augmentation across multiple representation formats.
    
    This class provides a unified interface for augmenting molecular data in different 
    formats through specialized modules. It maintains consistent random state management
    across all modules for reproducible augmentation results.
    
    The class contains three specialized modules:
    - SMILESModule: For augmenting SMILES string representations of molecules
    - GraphsModule: For augmenting molecular graph representations
    - INCHIModule: For augmenting InChI string representations of molecules
    
    Parameters
    ----------
    `seed` : int, default=4123
        Random seed for initializing the random number generators used across
        all augmentation modules
        
    Attributes
    ----------
    `seed` : int
        The random seed used for initialization
    `ss` : numpy.random.SeedSequence
        Seed sequence for generating independent random streams
    `rng` : numpy.random.RandomState
        Random number generator for reproducible random operations
    `SMILES` : SMILESModule
        Module containing methods for SMILES-based augmentation
    `Graphs` : GraphsModule
        Module containing methods for molecular graph-based augmentation
    `INCHI` : INCHIModule
        Module containing methods for InChI-based augmentation
    
    Examples
    --------
    >>> augmentator = Augmentator(seed=42)
    >>> augmented_data = augmentator.SMILES.augment_data(dataset='molecules.csv')
    """
    def __init__(self, seed: int = 42):
        """
        Initialize an Augmentator instance with the specified random seed.
        
        Sets up random number generation infrastructure and initializes the
        specialized augmentation modules.
        
        Parameters
        ----------
        `seed` : int, default=4123
            Random seed for initializing random number generators
        """
        self.seed = seed
        self.ss = np.random.SeedSequence(self.seed)
        self.rng = np.random.RandomState(seed=seed)

        self.SMILES = self.SMILESModule(self)
        self.Graphs = self.GraphsModule(self)

    class SMILESModule:
        """
        Module for augmenting molecular data in SMILES format.
        
        Provides methods for generating augmented SMILES representations using various
        techniques including masking, deletion, swapping, fusion, and enumeration.
        """
        def __init__(self, parent):
            self.parent = parent

        def augment_data(self, dataset: Path, mask_ratio: float = 0.1, delete_ratio: float = 0.3, 
                            augment_percentage: float = 0.2, augmentation_methods: List[str] = ["fusion", "enumeration"], col_to_augment: str = 'SMILES',
                            property_col: str = None) -> pd.DataFrame:
            """
            Augment molecular SMILES data from a CSV file.
            
            Reads SMILES strings from a CSV file, applies specified augmentation methods,
            and returns the augmented dataset. Also saves the augmented dataset to a new CSV file.
            
            Parameters
            ----------
            `dataset` : Path
                Path to the CSV file containing SMILES data to augment

            `mask_ratio` : float, default=0.1
                Fraction of tokens to mask when using masking-based augmentation methods

            `delete_ratio` : float, default=0.3
                Fraction of tokens to delete when using deletion-based augmentation methods

            `seed` : int, default=42
                Random seed for reproducible augmentation

            `augment_percentage` : float, default=0.2
                Target size of augmented dataset as a fraction of original dataset size

            `augmentation_methods` : List[str], default=["fusion", "enumerate"]
                List of augmentation methods to apply. Valid options include: 
                "mask", "delete", "swap", "fusion", "enumeration"

            `col_to_augment` : str, default='SMILES'
                Column name in the CSV file containing SMILES strings to augment

            `property_col` : str, optional
                Column name containing property values to preserve in augmented data
                
            Returns
            -------
            `pd.DataFrame`
                DataFrame containing both original and augmented molecules, with a 'parent_idx'
                column linking augmented molecules to their source molecules
                
            Notes
            -----
            The augmented dataset is automatically saved to "Augmented_QM9.csv" in the
            current working directory.
            """
            df = pd.read_csv(dataset, index_col=0)

            new_df = augment_smiles_dataset(
                col_to_augment=col_to_augment,  
                dataset=df,
                augmentation_methods=augmentation_methods,
                mask_ratio=mask_ratio,
                property_col=property_col,  
                delete_ratio=delete_ratio,
                augment_percentage=augment_percentage,
                seed=self.parent.seed
            )


            new_df = new_df.drop_duplicates()
            new_df.to_csv(f"Augmented_{dataset}", index=True, float_format='%.8e')

            new_data = len(new_df) - len(df)
            print(f"Generated new {new_data} SMILES")

            return new_df
            
    class GraphsModule:
        """
        Module for augmenting molecular data in graphs format.

        Provides methods for generating augmented graph representations using various
        techniques including edge dropping, node dropping, feature masking, and edge perturbation.
        """
        def __init__(self, parent):
            self.parent = parent

        def augment_data(self, dataset: List[Data], augmentation_methods: List[str] = ["edge_drop", "node_drop", "feature_mask", "edge_perturb"],
                        edge_drop_rate: float = 0.1, node_drop_rate: float = 0.1, feature_mask_rate: float = 0.1,
                        edge_add_rate: float = 0.05, edge_remove_rate: float = 0.05, augment_percentage: float = 0.2, 
                        seed: int = 42, save_to_file: bool = False, output_path: str = "augmented_graphs") -> List[Data]:

            """
            Augment molecular graph data using various augmentation methods.

            Parameters
            ----------
            `dataset` : List[Data]
                List of torch_geometric Data objects representing molecular graphs to augment
            `augmentation_methods` : List[str], default=["edge_drop", "node_drop", "feature_mask", "edge_perturb"]
                List of augmentation methods to apply. Valid options include:
                "edge_drop", "node_drop", "feature_mask", "edge_perturb"
            `edge_drop_rate` : float, default=0.1
                Fraction of edges to randomly drop when using edge dropping augmentation
            `node_drop_rate` : float, default=0.1
                Fraction of nodes to randomly drop when using node dropping augmentation
            `feature_mask_rate` : float, default=0.1
                Fraction of node features to randomly mask when using feature masking augmentation
            `edge_add_rate` : float, default=0.05
                Fraction of edges to randomly add when using edge adding augmentation
            `edge_remove_rate` : float, default=0.05
                Fraction of edges to randomly remove when using edge removing augmentation
            `augment_percentage` : float, default=0.2
                Target size of augmented dataset as a fraction of original dataset size
            `seed` : int, default=42
                Random seed for reproducible augmentation
            `save_to_file` : bool, default=False
                Whether to save the augmented graphs to a PyTorch file
            `output_path` : str, default="augmented_graphs"
                Path for output file (without .pt extension)
            
            Returns
            -------
            `List[Data]`
                List of augmented torch_geometric Data objects representing molecular graphs
            """

            augmented_dataset = augment_graph_dataset(graphs=dataset, augmentation_methods=augmentation_methods,
                                               edge_drop_rate=edge_drop_rate, node_drop_rate=node_drop_rate,
                                               feature_mask_rate=feature_mask_rate, edge_add_rate=edge_add_rate,
                                               edge_remove_rate=edge_remove_rate, augment_percentage=augment_percentage,
                                               seed=seed)
            
            if save_to_file:
                self.save_graphs(augmented_dataset, output_path)
            
            return augmented_dataset

        def save_graphs(self, graphs: List[Data], output_path: str = "augmented_graphs"):
            """
            Save augmented graphs using PyTorch's torch.save.
            
            Parameters
            ----------
            `graphs` : List[Data]
                List of torch_geometric Data objects to save
            `output_path` : str, default="augmented_graphs"
                Path for output file (without .pt extension)
            """
            filepath = f"{output_path}.pt"
            torch.save(graphs, filepath)
            print(f"Saved {len(graphs)} augmented graphs to: {filepath}")

        @staticmethod
        def load_graphs(filepath: str) -> List[Data]:
            """
            Load previously saved augmented graphs.
            
            Parameters
            ----------
            `filepath` : str
                Path to the saved PyTorch file (.pt)
                
            Returns
            -------
            `List[Data]`
                List of loaded torch_geometric Data objects
            """
            graphs = torch.load(filepath, weights_only=False)
            print(f"Loaded {len(graphs)} graphs from: {filepath}")
            return graphs

        def create_dataloader(self, graphs: List[Data], batch_size: int = 32, 
                            shuffle: bool = True) -> GeometricDataLoader:
            """
            Create a DataLoader from graphs.
            
            Parameters
            ----------
            `graphs` : List[Data]
                List of torch_geometric Data objects
            `batch_size` : int, default=32
                Batch size for the DataLoader
            `shuffle` : bool, default=True
                Whether to shuffle the data
                
            Returns
            -------
            `GeometricDataLoader`
                Configured DataLoader ready for training
            """
            return GeometricDataLoader(graphs, batch_size=batch_size, shuffle=shuffle)