import torch
from rdkit import Chem
from torch_geometric.data import Data
from rdkit.Chem import BRICS
import numpy as np

# Mapeamentos para conversão entre o valor numérico e a ordem da ligação
numeric_to_order = {
    0: 1,  # simples
    1: 2,  # dupla
    2: 3,  # tripla
    3: 1   # aromática (tratada como simples)
}
order_to_numeric = {
    1: 0,  # simples
    2: 1,  # dupla
    3: 2   # tripla
}

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
    
    node_symbols = []
    node_atomic_nums = []
    for atom in mol.GetAtoms():
        node_symbols.append(atom.GetSymbol())
        node_atomic_nums.append(atom.GetAtomicNum())
    
    edge_index = []
    edge_types = []
    
    BOND_TYPE_MAPPING = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }
    
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()

        bond_type = bond.GetBondType()
        bond_type_numeric = BOND_TYPE_MAPPING.get(bond_type, None) 
        
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
    
    data = Data(x=torch.tensor(node_atomic_nums, dtype=torch.float).view(-1, 1),
                edge_index=edge_index)
    data.symbol = node_symbols
    data.atomic_num = torch.tensor(node_atomic_nums, dtype=torch.long)
    data.edge_type = edge_types
    
    return data

def atom_masking(SMILES: str, mask_probability: float) -> Data:
    """
    Aplica atom masking diretamente no objeto Data do torch_geometric.
    
    Para cada nó, com probabilidade mask_probability:
      - O símbolo é substituído por '[M]'
      - O número atômico é substituído por 0
    Após o masking, atualiza também o tensor 'x' com os novos valores.
    """
    data = smiles_to_data(SMILES)

    new_symbols = []
    new_atomic_nums = []
    
    for i in range(data.num_nodes):
        if np.random.rand() < mask_probability:
            new_symbols.append('[M]')
            new_atomic_nums.append(0)
        else:
            new_symbols.append(data.symbol[i])
            new_atomic_nums.append(int(data.atomic_num[i]))
    
    data.symbol = new_symbols
    data.atomic_num = torch.tensor(new_atomic_nums, dtype=torch.long)
    data.x = torch.tensor(new_atomic_nums, dtype=torch.float).view(-1, 1)
    
    return data

def bond_deletion(smiles: str, bond_deletion_probability: float) -> Data:
    """
    Aplica deleção de ligações em um objeto Data do torch_geometric.
    
    Para cada ligação (armazenada em pares):
      - Se a ligação for múltipla (ordem > 1), reduz sua ordem em 1.
      - Se for simples (ordem == 1) ou aromática, remove completamente a ligação.
    
    A função retorna um novo objeto Data com as arestas (e seus tipos) atualizados.
    """
    data = smiles_to_data(smiles)
    
    edge_index_list = data.edge_index.t().tolist()  
    edge_type_list = data.edge_type.tolist()           
    
    new_edge_index = []
    new_edge_type = []
    
    i = 0
    while i < len(edge_index_list):
        bond_edge_1 = edge_index_list[i]
        bond_edge_2 = edge_index_list[i+1]
        bond_numeric = edge_type_list[i]  
        order = numeric_to_order.get(bond_numeric, 1)
        
        if np.random.rand() < bond_deletion_probability:
            if order > 1:
                new_order = order - 1
                new_numeric = order_to_numeric.get(new_order, 0)
                new_edge_index.append(bond_edge_1)
                new_edge_index.append(bond_edge_2)
                new_edge_type.append(new_numeric)
                new_edge_type.append(new_numeric)
        else:
            new_edge_index.append(bond_edge_1)
            new_edge_index.append(bond_edge_2)
            new_edge_type.append(bond_numeric)
            new_edge_type.append(bond_numeric)
        
        i += 2 
    
    if new_edge_index:
        new_edge_index_tensor = torch.tensor(new_edge_index, dtype=torch.long).t().contiguous()
        new_edge_type_tensor = torch.tensor(new_edge_type, dtype=torch.long)
    else:
        new_edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        new_edge_type_tensor = torch.empty((0,), dtype=torch.long)
    
    data.edge_index = new_edge_index_tensor
    data.edge_type = new_edge_type_tensor
    
    return data

"""
 2.1.5. Substructure removal
 Substructure removal follows the BRICS decomposition in FP augmentations, where molecular graphs are
 created based on the decomposed fragments from BRICS and are assigned with the same label as the original
 molecule. Fragment graphs contain one or more important functional groups from the molecule, and GNNs
 trained on such augmented data learn to correlate target properties with functional groups
"""

def brics_decompose(smiles: str):
    """
    Realiza a decomposição BRICS de uma molécula representada por SMILES.
    Retorna uma lista de objetos Data do torch_geometric, 
    cada um representando um fragmento da molécula.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("SMILES inválido")
    
    fragments = BRICS.BRICSDecompose(mol)
    
    data_list = [smiles_to_data(fragment) for fragment in fragments]
    
    return data_list