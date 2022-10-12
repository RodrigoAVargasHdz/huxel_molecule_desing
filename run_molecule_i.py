import os
import argparse
from typing import Tuple

import numpy as onp
import jax.numpy as jnp

from huxel import myMolecule
from huxel import optimization as _opt


def get_jcp_molecule_data(smile_i: int = 2) -> Tuple:
    """Molecules from  (only frameworks 1 to 8)
     Dequan Xiao, Weitao Yang, and David N. Beratan,
     "Inverse molecular design in a tight-binding framework", 
     J. Chem. Phys. 129, 044106 (2008) 
     https://doi.org/10.1063/1.2955756 

    Args:
        smile_i (int, optional): id value. (Default) Benzene -> smile_i = 2

    Returns:
        Tuple: smile string, list of atoms, connectivity matrix, XYZ matrix
    """

    data_d = os.path.join(os.getcwd(), 'molecules')
    d = onp.load(os.path.join(
        data_d, f"smile{smile_i}.npy"), allow_pickle=True)

    smile = d.item()['smile']
    atom_types = d.item()['atoms']
    connectivity_matrix = jnp.array(d.item()['AdjacencyMatrix'], dtype=int)
    xyz = jnp.array(d.item()['xyz'])
    return smile, atom_types, connectivity_matrix, xyz


def main():
    parser = argparse.ArgumentParser(
        description="molecular inverse design Huckel with JAX")
    parser.add_argument("--s", type=int, default=2,
                        help="smile integer, range [1 to 8]", choices=[1, 2, 3, 4, 5, 6, 7, 8],)
    parser.add_argument("--l", type=int, default=0, help="label")
    # parser.add_argument("--lr", type=float, default=2e-2, help="learning rate")
    parser.add_argument("--obj", type=str,
                        default="homo_lumo", help="objective type")
    parser.add_argument("--opt", type=str, default="BFGS",
                        help="optimizer name", choices=['BFGS', 'Adam', 'GD'],)
    parser.add_argument("--extfield", type=float, default=0.,
                        help="external field for polarization")

    args = parser.parse_args()
    smile_i = args.s
    l = args.l
    # lr = args.lr
    obj = args.obj
    opt = args.opt
    ext_field = args.extfield

    # read molecule info
    smile, atom_types, connectivity_matrix, xyz = get_jcp_molecule_data(
        smile_i)

    print(smile)
    print(atom_types)
    print(connectivity_matrix)
    print(xyz)

    molec = myMolecule(
        smile_i,
        smile,
        atom_types,
        connectivity_matrix,
        xyz
    )
    print(molec)

    if obj == 'homo_lumo':
        _opt(l, molec, obj, opt)
    elif obj == 'polarizability':
        _opt(l, molec, obj, opt, ext_field)


if __name__ == "__main__":
    main()
