from rdkit import Chem
from rdkit.Chem import AllChem

mol = Chem.MolFromSmiles('N[C]1C(=C([NH])ON=C1)O')
mol = Chem.AddHs(mol)

isotope_mol = Chem.Mol(mol)  
isotope_mol.GetAtomWithIdx(2).SetIsotope(2)  
inchi_isotopically_modified = Chem.MolToInchi(isotope_mol)
print(f'InChI com isótopo (deutério): {inchi_isotopically_modified}')