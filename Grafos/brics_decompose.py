from rdkit import Chem
from rdkit.Chem import BRICS, Draw
import networkx as nx 
from smiles_converter import smiles_to_graph

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
    Retorna um conjunto de fragmentos SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("SMILES inválido")

    # Realiza a decomposição BRICS
    fragments = BRICS.BRICSDecompose(mol)
    
    return fragments

def draw_fragments(fragments):
    """
    Desenha os fragmentos gerados pela decomposição BRICS.
    """
    mols = [Chem.MolFromSmiles(frag) for frag in fragments]
    img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(200, 200), legends=[f"Fragment {i+1}" for i in range(len(mols))])
    img.show()


if __name__ == '__main__':
    smiles = "CC(=O)Nc1ccc(C(=O)O)cc1"  # Exemplo: Paracetamol
    fragments = brics_decompose(smiles)
    
    draw_fragments(fragments)
    print("Fragmentos BRICS:", fragments)
