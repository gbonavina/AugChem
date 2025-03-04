import itertools
import numpy as np
import random
from rdkit import Chem
from rdkit import RDLogger
from molvs import tautomer

RDLogger.DisableLog('rdApp.*')

class AtomError(Exception):
    def __init__(self, message: str = "Não há o átomo especificado nesta molécula.") -> None:
        super().__init__(message)

class Inchi_Augment:
    def __init__(self, seed: int = 4123):
        self.seed = seed
        self.ss = np.random.SeedSequence(self.seed)
    
    def augment_inchi(self, smiles: str, changed_atom: str, subs: str):
        """
        Funçao para realizar aumento de dados na representacao InChI. 
        1. Troca de atomos
        2. Protonacao
        3. Isotopos
        4. Tautomeria
        """

        inchi_augmentations = []
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("SMILES inválido. Por favor, forneça um SMILES válido.")
        
        # 1. Aumento de dados baseado na substituiçao de atomos
        atom_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == changed_atom]
        if not atom_indices:
            raise AtomError
        
        random_atom_idx = random.choice(atom_indices)
        mol_editor = Chem.RWMol(mol)
        mol_editor.ReplaceAtom(random_atom_idx, Chem.Atom(subs))
        Chem.SanitizeMol(mol_editor)
        
        inchi_augmentations.append(Chem.MolToInchi(mol_editor))
        
        # 2. Aumento de dados baseado no aumento de hidrogenios na molécula
        atoms_to_protonate = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in ['N', 'O']:
                if atom.GetExplicitValence() < atom.GetTotalValence():
                    atoms_to_protonate.append(atom.GetIdx())
                    
        for i in range(1, len(atoms_to_protonate) + 1):
            for combination in itertools.combinations(atoms_to_protonate, i):
                new_mol = Chem.Mol(mol)
                valid = True
                for atom_idx in combination:
                    atom = new_mol.GetAtomWithIdx(atom_idx)
                    if atom.GetExplicitValence() >= atom.GetTotalValence():
                        valid = False
                        break
                    # "Protona" o átomo aumentando a carga formal
                    atom.SetFormalCharge(atom.GetFormalCharge() + 1)
                    atom.UpdatePropertyCache()
                if not valid:
                    continue
                try:
                    Chem.SanitizeMol(new_mol)
                    inchi_augmentations.append(Chem.MolToInchi(new_mol))
                except Exception as e:
                    print(f"Erro ao sanitizar molécula protonada: {e}")

        # 3. Aumento de dados baseado na geraçao de isotopos da molecula
        isotope_variants = []
        isotope_dict = {'C': 13, 'N': 15, 'O': 18, 'H': 2}  
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol in isotope_dict:
                new_mol = Chem.Mol(mol)
                new_atom = new_mol.GetAtomWithIdx(atom.GetIdx())
                new_atom.SetIsotope(isotope_dict[symbol])
                try:
                    Chem.SanitizeMol(new_mol)
                    isotope_variants.append(new_mol)
                except Exception as e:
                    print(f"Erro ao sanitizar molécula com isótopo: {e}")
        for iso_mol in isotope_variants:
            inchi_augmentations.append(Chem.MolToInchi(iso_mol))

        # 4. Aumento de dados baseado nas multiplas representaçoes da molecula
        try:
            enumerator = tautomer.TautomerEnumerator()
            tautomers = enumerator.enumerate(mol)
            for taut in tautomers:
                try:
                    Chem.SanitizeMol(taut)
                    inchi = Chem.MolToInchi(taut)
                    if inchi not in inchi_augmentations:
                        inchi_augmentations.append(inchi)
                except Exception as e:
                    print(f"Erro ao processar tautômero: {e}")
        except Exception as e:
            print(f"Erro ao gerar tautômeros: {e}")
        
        return inchi_augmentations

# Exemplo de uso:
if __name__ == '__main__':
    smiles_input = "N[C]1C(=C([NH])ON=C1)O" 
    try:
        augmenter = Inchi_Augment()
        augmentations = augmenter.augment_inchi(smiles=smiles_input, changed_atom='C', subs='Si')
        for idx, inchi in enumerate(augmentations):
            print(f"Augmentation {idx+1}: {inchi}")
    except Exception as e:
        print(e)
