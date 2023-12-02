# Python version: 3.11.5
# Modified Dec 2, 2023
'''
Helper Functions for the Modules
--------------------------------
This script conains functions needed to generate objects or
processes within the main modules.
'''

############ Imports ############
import numpy as np
import scipy as sp
#################################

def get_ground_state(Hamiltonian):
    '''
    Quickly returns the ground state energy and ground state vector without having to call
    scipy or numpy and manually select these values.
    -----------
    Parameters:
    -    Hamiltonian : numpy NDarray
            The Hamiltonian with the desired ground state.

    Returns:
    -    GS_energy : float
            The energy of the ground state of Hamiltonian.
    -    GS_vector : numpy array
            The vector of the ground state of Hamiltonian.
    '''
    if len(Hamiltonian) > 2**10:
        evals,evecs = sp.sparse.linalg.eigsh(Hamiltonian,k=5)
        GS_energy = evals[0]
        GS_vector = evecs[:,0]
    else:
        evals,evecs = sp.linalg.eigh(Hamiltonian)
        GS_energy = evals[0]
        GS_vector = evecs[:,0]

    return GS_energy, GS_vector

def get_overlap_matrix(basis_vectors):
    '''
    Produces the overlap matrix.
    -----------
    Parameters:
    -   basis_vectors : numpy NDarray
            The set of basis vectors.
    
    Returns:
    -   overlap : numpy NDarray
            The matrix describing the overlap of the basis vectors.
    '''
    nvecs = len(basis_vectors)
    overlap = np.zeros([nvecs,nvecs],dtype='complex')

    for i in range(nvecs):

        v_i = basis_vectors[i,:]

        for j in range(i,nvecs):
            v_j = basis_vectors[j,:]

            overlap[i,j] = np.dot(np.conjugate(np.transpose(v_i)),v_j)

            if not i == j:
                overlap[j,i] = np.conjugate(overlap[i,j])

    return overlap
    
def save_dict_to_txt(dictionary, file_name, path):
    '''
    Saves a dictionary to a .txt file in the format:

    key1   value1_1   value1_2  ...  value1_M
    key2   value2_1   value2_2  ...  value2_K 
     ...     ...        ...     ...    ...
    keyN   valueN_1   valueN_2  ...  valueN_L

    Where the values are separated by tabs and each entry is on its own line. 
    ".txt" should be included with file_name
    -----------
    Parameters:
    -   dictionary : dict
            A dictionary containing the data to be saved.
    -   file_name : str
            The name of the file to be saved.
    -   path : str
            The location in the local system to save the data.
    '''
    with open(path+'/'+file_name, 'w') as file:
        for key, values in dictionary.items():
            file.write(key + '\t' + '\t'.join(str(val) for val in values) + '\n')

def load_txt_to_dict(file_name, path):
    '''
    Loads a .txt file in the format:

    key1   value1_1   value1_2  ...  value1_M
    key2   value2_1   value2_2  ...  value2_K 
     ...     ...        ...     ...    ...
    keyN   valueN_1   valueN_2  ...  valueN_L

    to a dictionary. The values are separated by tabs and each entry is on its own line. 
    ".txt" should be included with file_name
    -----------
    Parameters:
    -   file_name : str
            The name of the file to be saved.
    -   folder : str
            The relative folder to save to in the main "data" folder.

    Returns:
    -   data : dict
            The data in 'file_name' as a dictionary.
    '''
    data = {}
    with open(path+'/'+file_name, 'r') as file:
        for line in file:
            parts = line.split('\t')
            key = parts[0]
            values = [float(part.strip()) for part in parts[1:]]
            data[key] = values

    return data

