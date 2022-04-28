import os
import argparse
import numpy as onp
import jax.numpy as jnp

from huxel.molecule import myMolecule
from huxel.optimization_inversemol import _optimization_molec as _opt

def main_new():
    parser = argparse.ArgumentParser(description="molecular inverse design Huckel with JAX")
    parser.add_argument("--s", type=int, default=0, help="smile integer for JCP2008 molecules, range [1 to 8]")
    parser.add_argument("--l", type=int, default=2, help="label")
    # parser.add_argument("--lr", type=float, default=2e-2, help="learning rate")
    parser.add_argument("--obj", type=str, default="homo_lumo", help="objective type")
    parser.add_argument("--opt", type=str, default="BFGS", help="objective type")
    parser.add_argument("--extfield", type=float, default=0.01, help="external field for polarization")

    # read molecule info
    smile_i = 8 # Benzene --> smile_i = 2
    data_d = os.path.join(os.getcwd(),'molecules')
    d = onp.load(os.path.join(data_d,f"smile{smile_i}.npy"),allow_pickle=True)

    smile = d.item()['smile']
    atom_types = d.item()['atoms']
    conectivity_matrix = jnp.array(d.item()['AdjacencyMatrix'],dtype=int)
    xyz = jnp.array(d.item()['xyz'])
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
    print(molec.id,molec.smile)
    assert 0


    args = parser.parse_args()
    l = args.l
    # lr = args.lr
    obj = args.obj
    opt = args.opt
    ext_field = args.extfield

    if obj == 'homo_lumo':
        _opt(l, molec,obj,opt)
    elif obj == 'polarizability':
        _opt(l, molec,obj,opt,ext_field)


def main():
    parser = argparse.ArgumentParser(description="opt overlap NN")
    parser.add_argument("--l", type=int, default=0, help="label")
    # parser.add_argument("--lr", type=float, default=2e-2, help="learning rate")
    parser.add_argument("--obj", type=str, default="homo_lumo", help="objective type")
    parser.add_argument("--opt", type=str, default="BFGS", help="objective type")
    parser.add_argument("--extfield", type=float, default=0.01, help="external field for polarization")

    args = parser.parse_args()
    l = args.l
    # lr = args.lr
    obj = args.obj
    opt = args.opt
    ext_field = args.extfield

    atom_types = ["X", "C", "C", "C", "X", "C"]
    # atom_types = ["X", "X", "X", "X", "X", "X"]
    smile = "C6"

    conectivity_matrix = jnp.array(
        [
            [0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0],
        ],
        dtype=int,
    )
    # Kjell
    xyz = jnp.array([[ 1.40000000e+00,  3.70074342e-17,  0.00000000e+00],
       [ 7.00000000e-01, -1.21243557e+00,  0.00000000e+00],
       [-7.00000000e-01, -1.21243557e+00,  0.00000000e+00],
       [-1.40000000e+00,  2.08457986e-16,  0.00000000e+00],
       [-7.00000000e-01,  1.21243557e+00,  0.00000000e+00],
       [ 7.00000000e-01,  1.21243557e+00,  0.00000000e+00]])

    homo_lumo_grap_ref = -7.01 - (-0.42)

    molec = myMolecule(
        "benzene",
        smile,
        atom_types,
        conectivity_matrix,
        xyz
    )

    if obj == 'homo_lumo':
        _opt(l, molec,obj,opt)
    elif obj == 'polarizability':
        _opt(l, molec,obj,opt,ext_field)


if __name__ == "__main__":
    # main()
    main_new()
