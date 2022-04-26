import time
import argparse
import jax
import jax.numpy as jnp

from huxel.optimization_inversemol import _optimization_molec as _opt
from huxel.optimization_inversemol import _benzene_test

from huxel.prediction import _pred, _pred_def
from huxel.molecule import myMolecule
from huxel.comb_benzene import _all_benzene, single_test


def main():
    atom_types = ["X", "C", "C", "C", "X", "C"]
    smile = "C6"

    atom_t = ["P1", "O1", "Si", "Si", "O1", "P1"]
    smile = 0

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
        homo_lumo_grap_ref,
        jnp.ones((6, 6)),
        xyz
    )
    
    _opt(0, molec)#,'polarizability' 
    # _opt(2, molec,'polarizability',0.01)#,'polarizability' 

    # for l in range(0, 20):
    #     _opt(l, molec,'polarizability')

    # _all_benzene()

    # single_test()


if __name__ == "__main__":
    main()
