import torch
from rdkit import Chem
from torch_geometric.data import Data
import numpy as np

def smiles_to_data(smiles: str) -> Data:
    """
    Converte um SMILES em um objeto Data do torch_geometric.
    
    Extraímos os átomos e ligações com o RDKit e armazenamos:
      - 'x': tensor com o número atômico de cada átomo 
      - 'symbol': lista com o símbolo de cada átomo
      - 'atomic_num': tensor com o número atômico 
      - 'edge_type': tensor com o tipo de ligação 
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("SMILES inválido.")
    
    # Processa os átomos
    node_symbols = []
    node_atomic_nums = []
    for atom in mol.GetAtoms():
        node_symbols.append(atom.GetSymbol())
        node_atomic_nums.append(atom.GetAtomicNum())
    
    # Cria as arestas (bidirecionais) e extrai os tipos de ligação
    edge_index = []
    edge_types = []
    
    # Mapeamento dos tipos de ligação para números
    BOND_TYPE_MAPPING = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }
    
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        # Obtém o tipo de ligação e converte para número
        bond_type = bond.GetBondType()
        bond_type_numeric = BOND_TYPE_MAPPING.get(bond_type, None)  # -1 se não estiver mapeado
        
        # Adiciona as arestas bidirecionais com seus respectivos tipos
        # Começo -> fim
        edge_index.append([start, end])
        edge_types.append(bond_type_numeric)

        # Fim -> começo
        edge_index.append([end, start])
        edge_types.append(bond_type_numeric)
    
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_types = torch.tensor(edge_types, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_types = torch.empty((0,), dtype=torch.long)
    
    # Cria o objeto Data com as informações dos nós e arestas
    data = Data(x=torch.tensor(node_atomic_nums, dtype=torch.float).view(-1, 1),
                edge_index=edge_index)
    data.symbol = node_symbols
    data.atomic_num = torch.tensor(node_atomic_nums, dtype=torch.long)
    data.edge_type = edge_types
    
    return data
