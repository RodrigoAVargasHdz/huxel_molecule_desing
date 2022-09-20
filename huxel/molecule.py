import jax
import jax.numpy as jnp
from typing import Any

from huxel.parameters import Bohr_to_AA


class myMolecule:
    '''
    Basic class for individual molecule
    '''

    def __init__(self,
                 id0: int,
                 smile: str,
                 atom_types: list,
                 connectivity_matrix: jnp.ones((1, 3)),
                 xyz: jnp.ones((1, 3)),
                 dm: Any = None
                 ):
        """base molecule class

        Args:
            id (int): id
            smile (str): smile
            atom_types (list): list of atoms
            connectivity_matrix (jnp.ones): connectivity matrix
            xyz (jnp.ones): XYZ matrix
            dm (Any, optional): distance based matrix
        """

        self.id = id0
        self.smile = smile
        self.atom_types = atom_types
        self.connectivity_matrix = connectivity_matrix
        self.xyz = xyz
        self.dm = dm

    def get_dm(self):
        """distance matrix
        """
        z = self.xyz[:, None] - self.xyz[None, :]
        self.dm_AA = jnp.linalg.norm(z, axis=2)  # compute the bond length

    def get_dm_AA_to_Bohr(self):
        """distance matrix in Bohr
        """
        z = self.xyz[:, None] - self.xyz[None, :]
        dm = jnp.linalg.norm(z, axis=2)  # compute the bond length
        self.dm_Bhor = jnp.divide(dm, Bohr_to_AA)  # Bohr -> AA

    def get_xyz_AA_to_Borh(self):
        """XYZ matrix in Bohr
        """
        self.xyz_Bhor = jnp.divide(self.xyz, Bohr_to_AA)  # Bohr -> AA


if __name__ == "__main__":

    atom_types = ['C', 'C', 'C', 'C']

    connectivity_matrix = jnp.array([[0, 1, 0, 0],
                                     [1, 0, 1, 0],
                                     [0, 1, 0, 1],
                                     [0, 0, 1, 0]], dtype=int)
    homo_lumo_grap_ref = 1.0

    molec = myMolecule('test', atom_types, connectivity_matrix, 2.)
    molecs = [molec, molec]
    print(molecs[0].homo_lumo_grap_ref)
