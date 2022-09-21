import os
import argparse
import numpy as onp
import jax.numpy as jnp

from huxel import myMolecule
from huxel import optimization as _opt
from huxel import get_external_field
from huxel import _construct_huckel_matrix_field


def main():
    """Benzene example.

    Connectivity matrix from benzene. All site-atoms are considered "optimizable".
    If you want to only optimize some sites, change "X" to the atom that you want to consider, e.g. "X" -> "C".
    For a single "optimizable" atom-site, atom_types = ["C", "X", "X", "X", "X", "X"].


    """
    parser = argparse.ArgumentParser(description="opt overlap NN")
    parser.add_argument("--l", type=int, default=0, help="label")
    # parser.add_argument("--lr", type=float, default=2e-2, help="learning rate")
    parser.add_argument("--obj", type=str,
                        default="homo_lumo", help="objective type")
    parser.add_argument("--opt", type=str, default="BFGS",
                        help="optimizer name", choices=['BGFS', 'Adam', 'GD'],)
    parser.add_argument("--extfield", type=float, default=0.0,
                        help="external field for polarization")

    args = parser.parse_args()
    l = args.l
    # lr = args.lr
    obj = args.obj
    opt = args.opt
    ext_field = args.extfield

    # atom_types = ["X", "C", "C", "C", "X", "C"]
    atom_types = ["X", "X", "X", "X", "X", "X"]
    smile = "C6"

    connectivity_matrix = jnp.array(
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

    xyz = jnp.array([[1.40000000e+00,  3.70074342e-17,  0.00000000e+00],
                     [7.00000000e-01, -1.21243557e+00,  0.00000000e+00],
                     [-7.00000000e-01, -1.21243557e+00,  0.00000000e+00],
                     [-1.40000000e+00,  2.08457986e-16,  0.00000000e+00],
                     [-7.00000000e-01,  1.21243557e+00,  0.00000000e+00],
                     [7.00000000e-01,  1.21243557e+00,  0.00000000e+00]])

    # homo_lumo_grap_ref = -7.01 - (-0.42)

    molec = myMolecule(
        "benzene_test",
        smile,
        atom_types,
        connectivity_matrix,
        xyz
    )

    if obj == 'homo_lumo':
        _opt(l, molec, obj, opt)
    elif obj == 'polarizability':
        ext_field = get_external_field('polarizability', 0.01)
        print(_construct_huckel_matrix_field(molec, ext_field))
        _opt(l, molec, obj, opt, ext_field)


if __name__ == "__main__":
    main()
