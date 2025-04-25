from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='Augchem',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        "pandas==2.2.3",
        "MolVS==0.1.1",
        "numpy>=1.26.0",
        "rdkit==2024.9.6",
        "torch==2.6.0",
        "torch-geometric==2.6.1"
    ],
    author='Gabriel Bonavina',
    author_email='gabriel.bonavina@unifesp.br',
    description='Toolbox created in a partnership with FAPESP and CINE to facilitate use of Data Augmentation methods for chemical datasets.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/gbonavina/AugChem/tree/develop',
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
