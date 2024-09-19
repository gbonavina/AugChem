import os
import re
from rdkit import Chem
from rdkit import RDLogger
import numpy as np
import time

RDLogger.DisableLog('rdApp.error')

class Loader:
    def __init__(self, path):
        self.path = path
        self.data = None

    def loadSingleFile(self, file):
        """
        Carrega um arquivo de um dataset
        # Parâmetros:
        file: str - caminho do arquivo
        """
        with open(file, 'r') as f:
            # Número de átomos
            natoms = int(f.readline())
            f.readline()  # Pula a segunda linha
            
            smiles = []
        
            for num_line, line in enumerate(f):
                if num_line == natoms + 1:
                    smiles.append(line.split())

            match = re.search(r'dsgdb9nsd_(\d+)', file)
            if match:
                file_number = match.group(1)
            else:
                file_number = "unknown"

            return {
                "file": file_number,
                "smiles": smiles
            }
        
    def loadDataset(self, directory_path, list_mols=[]):
        """Load the entire QM9 dataset from a directory containing .xyz files.
        # Parâmetros:
        directory_path: str - caminho do diretório contendo os arquivos .xyz
        list_mols: list - lista de moléculas
        """

        # Contador para limitar o número de moléculas carregadas para facilitar em testes menores
        cont = 0

        for file in os.listdir(directory_path):
            if file.endswith(".xyz") and cont != 20:
                file_path = os.path.join(directory_path, file)
                mol = self.loadSingleFile(file_path)
                list_mols.append(mol)

                #print(f"Loaded {file}")
                cont += 1

        return list_mols

    def verifyMolecules(self, list_mols):
        """Verifica se as moléculas foram importadas corretamente usando RDKit.
        # Parâmetros:
        list_mols: list - lista de moléculas
        """
        valid_mols = []
        invalid_mols = []

        for mol in list_mols:
            smiles = mol["smiles"][0][0]
            rdkit_mol = Chem.MolFromSmiles(smiles)
            if rdkit_mol:
                valid_mols.append(smiles)
            else:
                print("Invalid molecule: ", smiles)
                invalid_mols.append(smiles)
                break

        return valid_mols, invalid_mols
    
class Augmentator(object):
    def __init__(self, seed, fusion=False):
        self.seed = seed
        self.fusion = fusion

    def slice_smiles(self, smiles):
        """
        Slice a SMILES string into tokens.
        # Params:
        smiles: str - SMILES
        """
        sliced_smiles = []
        i = 0
        while i < len(smiles):
            if smiles[i] == '(' or smiles[i] == '[':
                # Handle parentheses and brackets
                end = smiles.find(')', i) if smiles[i] == '(' else smiles.find(']', i)
                sliced_smiles.append(smiles[i:end+1])
                i = end + 1
            elif smiles[i].isalpha():
                # Handle atoms (single letter or two letters)
                if i + 1 < len(smiles) and smiles[i+1].islower():
                    atom = smiles[i:i+2]
                    i += 2
                else:
                    atom = smiles[i]
                    i += 1
                
                # Check for numbers after the atom
                while i < len(smiles) and smiles[i].isdigit():
                    atom += smiles[i]
                    i += 1
                
                sliced_smiles.append(atom)
            else:
                # Handle other characters (=, #, etc.)
                sliced_smiles.append(smiles[i])
                i += 1

        return sliced_smiles
    
    def tokenizer(self, smiles):
        # Tokenize the SMILES string
        charset = set('()[]=#')
        tokens = []
        non_charset_indices = []

        i = 0
        while i < len(smiles):
            if smiles[i] in charset:
                tokens.append(smiles[i])
                i += 1
            elif smiles[i].isalpha():
                if i + 1 < len(smiles) and smiles[i+1].islower():
                    tokens.append(smiles[i:i+2])
                    i += 2
                else:
                    tokens.append(smiles[i])
                    i += 1
                non_charset_indices.append(len(tokens) - 1)
            elif smiles[i].isdigit():
                num = ''
                while i < len(smiles) and smiles[i].isdigit():
                    num += smiles[i]
                    i += 1
                tokens.append(num)
                non_charset_indices.append(len(tokens) - 1)
            else:
                tokens.append(smiles[i])
                i += 1

        return tokens, non_charset_indices
    
    def mask(self, smiles, mask_ratio=0.05):
        """
        Mask a SMILES string with [M] token.
        # Params:
        smiles: str - SMILES

        mask_ratio: float - ratio of tokens to mask
        """
        np.random.seed(self.seed)
        token = '[M]'
        sliced_smiles = self.slice_smiles(smiles)        

        mask_indices = np.random.choice(len(sliced_smiles), int(len(sliced_smiles) * mask_ratio), replace=False)
        
        for idx in mask_indices:
            sliced_smiles[idx] = token

        return ''.join(sliced_smiles)   
    
    def delete(self, smiles, delete_ratio=0.3):
        """
        Delete tokens from a SMILES string.
        # Params:
        smiles: str - SMILES

        delete_ratio: float - ratio of tokens to delete
        """
        np.random.seed(self.seed)
        sliced_smiles = self.slice_smiles(smiles)

        delete_indices = np.random.choice(len(sliced_smiles), int(len(sliced_smiles) * delete_ratio), replace=False)
        #print(delete_indices)
        
        for idx in delete_indices:
            sliced_smiles[idx] = ''

        return ''.join(sliced_smiles)
    
    def swap(self, smiles):
        """
        Swap two random tokens in a SMILES string.
        # Params:
        smiles: str - SMILES
        """
        
        # Tokenize the SMILES string
        tokens, non_charset_indices = self.tokenizer(smiles)

        # Perform the swap
        np.random.seed(self.seed)
        
        # Randomly select two indices to swap
        idx1, idx2 = np.random.choice(non_charset_indices, 2, replace=False)
        
        # Swap the tokens
        tokens[idx1], tokens[idx2] = tokens[idx2], tokens[idx1]
    
        #print(f"Swapped {tokens[idx1]} with {tokens[idx2]}")

        return ''.join(tokens)
    
aug = Augmentator(seed=65)
mask_test = aug.mask("CC(=O)OC1=CC=CC=C1C(=O)O", mask_ratio=0.1)
delete_test = aug.delete("CC(=O)OC1=CC=CC=C1C(=O)O", delete_ratio=0.3)
swap_test = aug.swap("CC(=O)OC1=CC=CC=C1C(=O)O")

print("Mask:", mask_test)
print("Delete:", delete_test)
print("Swap:", swap_test)