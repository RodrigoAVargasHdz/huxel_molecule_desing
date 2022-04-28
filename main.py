import os
import argparse
import numpy as onp
import jax.numpy as jnp

from huxel.molecule import myMolecule
from huxel.optimization_inversemol import _optimization_molec as _opt

def get_jcp_molecule_data(smile_i:int=2):# Benzene --> smile_i = 2
    data_d = os.path.join(os.getcwd(),'molecules')
    d = onp.load(os.path.join(data_d,f"smile{smile_i}.npy"),allow_pickle=True)

    smile = d.item()['smile']
    atom_types = d.item()['atoms']
    conectivity_matrix = jnp.array(d.item()['AdjacencyMatrix'],dtype=int)
    xyz = jnp.array(d.item()['xyz'])
    return smile, atom_types, conectivity_matrix, xyz

def main():
    parser = argparse.ArgumentParser(description="molecular inverse design Huckel with JAX")
    parser.add_argument("--s", type=int, default=1, help="smile integer for JCP2008 molecules, range [1 to 8]")
    parser.add_argument("--l", type=int, default=2, help="label")
    # parser.add_argument("--lr", type=float, default=2e-2, help="learning rate")
    parser.add_argument("--obj", type=str, default="homo_lumo", help="objective type")
    parser.add_argument("--opt", type=str, default="BFGS", help="objective type")
    parser.add_argument("--extfield", type=float, default=0.01, help="external field for polarization")

    args = parser.parse_args()
    smile_i = args.s
    l = args.l
    # lr = args.lr
    obj = args.obj
    opt = args.opt
    ext_field = args.extfield

    # read molecule info
    smile, atom_types, conectivity_matrix, xyz = get_jcp_molecule_data(smile_i)

    print(smile)
    print(atom_types)
    print(conectivity_matrix)
    print(xyz)

    molec = myMolecule(
        smile_i,
        smile,
        atom_types,
        conectivity_matrix,
        xyz
    )
    print(molec)
    assert 0



    if obj == 'homo_lumo':
        _opt(l, molec,obj,opt)
    elif obj == 'polarizability':
        _opt(l, molec,obj,opt,ext_field)

if __name__ == "__main__":
    main()
