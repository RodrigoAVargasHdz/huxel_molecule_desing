import jax
import jax.numpy as jnp


class myMolecule:
    '''
    Basic class for individual molecule
    '''
    def __init__(self,id: int,smile: str,atom_types: list,conectivity_matrix: jnp.ones(1),xyz:jnp.ones((1,3))):
        self.id = id
        self.smile = smile
        self.atom_types = atom_types
        self.conectivity_matrix = conectivity_matrix
        self.xyz = xyz
        # self.homo_lumo_grap_ref = homo_lumo_grap_ref
        # self.dm = dm #distance matrix


if __name__ == "__main__":

    atom_types = ['C', 'C', 'C', 'C']

    conectivity_matrix = jnp.array([[0, 1, 0, 0],
                                [1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0]],dtype=int)
    homo_lumo_grap_ref = 1.0

    molec = myMolecule('test',atom_types,conectivity_matrix,2.)
    molecs = [molec,molec]
    print(molecs[0].homo_lumo_grap_ref)