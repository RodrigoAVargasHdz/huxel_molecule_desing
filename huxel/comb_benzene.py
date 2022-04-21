import os
import time
import datetime
import numpy as onp
import argparse
from typing import Any

import jax
import jax.numpy as jnp
from jax import random
from jax import lax, value_and_grad
from jax.nn import softmax
from jax.tree_util import tree_flatten, tree_unflatten

import jaxopt
from jaxopt import ScipyMinimize
import optax


from huxel.molecule import myMolecule
from huxel.beta_functions import _f_beta
from huxel.utils import (
    get_huckel_params,
    get_initial_params_b,
    get_initial_params_b_benzene,
)
from huxel.huckel import f_homo_lumo_gap


from jax.config import config

jax.config.update("jax_enable_x64", True)

f_beta = _f_beta("c")


def get_molecule(molec, one_pi_elec):
    # norm_params_b = jax.tree_map(lambda x: softmax(x), params_b)
    # molecule_atoms = []
    atoms = molec.atom_types
    params_one_hot = {}
    for index, a in enumerate(atoms):
        for i, b in enumerate(one_pi_elec):
            if a == b:
                z = -35 * jnp.ones(len(one_pi_elec))
                z = z.at[i].set(35.0)
                params_one_hot.update({index: z})
    return params_one_hot


def single_molecule(molec, one_pi_elec):

    params_extra = get_huckel_params()
    params_one_hot = get_molecule(molec, one_pi_elec)
    y_ev = f_homo_lumo_gap(params_one_hot, params_extra, molec, f_beta)
    return y_ev, params_one_hot


def _all_benzene():

    # r = onp.load("comb_benzene.npy", allow_pickle=True)
    # print(r)
    # for index, key in enumerate(r.item()):
    #     print(r.item()[key])
    # assert 0

    params_extra = get_huckel_params()
    one_pi_elec = params_extra["one_pi_elec"]
    i_m = 0

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
    homo_lumo_grap_ref = -7.01 - (-0.42)
    r = {}

    from itertools import combinations_with_replacement, permutations

    from itertools import product, permutations, combinations_with_replacement

    perm = permutations([0, 1, 2, 3, 4, 5], 6)
    comb = combinations_with_replacement([0, 1, 2, 3, 4, 5], 6)
    all = []
    all.append(list(perm))
    all.append(list(comb))
    all_ = []
    for i, x in enumerate(all):
        for xj in x:
            all_.append(xj)
    # print(len(list(perm)))
    r = {}
    for i_m, ai in enumerate(all_):
        atom_types = []
        print(ai)
        for j in list(ai):
            atom_types.append(one_pi_elec[j])

        smile = "_".join(atom_types)
        molec = myMolecule(
            i_m,
            smile,
            atom_types,
            conectivity_matrix,
            homo_lumo_grap_ref,
            jnp.ones((6, 6)),
        )
        y_ev, params_one_hot = single_molecule(molec, one_pi_elec)
        print(i_m, atom_types, y_ev)
        r.update(
            {
                i_m: {
                    "atoms": atom_types,
                    "i_m": i_m,
                    "params_b": params_one_hot,
                    "homo_lumo": y_ev,
                }
            }
        )
        # onp.save("comb_benzene_perm.npy", r, allow_pickle=True)


def single_test():
    # atom_types = ["Si", "S1", "O1", "Si", "S1", "O1"]
    atom_types = ["C", "P1", "N1", "P1", "P1", "N1"]
    params_extra = get_huckel_params()
    one_pi_elec = params_extra["one_pi_elec"]
    i_m = 0

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
    homo_lumo_grap_ref = -7.01 - (-0.42)

    smile = "_".join(atom_types)
    molec = myMolecule(
        i_m,
        smile,
        atom_types,
        conectivity_matrix,
        homo_lumo_grap_ref,
        jnp.ones((6, 6)),
    )
    y_ev, params_one_hot = single_molecule(molec, one_pi_elec)
    print(atom_types, y_ev)


if __name__ == "__main__":
    _all_benzene()

"""
['Si', 'C', 'O1', 'C', 'Si', 'O1']
['Si', 'S1', 'O1', 'Si', 'S1', 'O1']
['Si', 'S1', 'O1', 'Si', 'S1', 'O1']
['S1', 'Si', 'O1', 'S1', 'Si', 'O1']
['O1', 'Si', 'S1', 'O1', 'Si', 'S1']
['O1', 'S1', 'Si', 'O1', 'S1', 'Si']
['O1', 'S1', 'Si', 'O1', 'S1', 'Si']
['O1', 'Si', 'S1', 'O1', 'Si', 'S1']
['O1', 'S1', 'Si', 'O1', 'S1', 'Si']
['Si', 'O1', 'S1', 'Si', 'O1', 'S1']
['Si', 'O1', 'S1', 'Si', 'O1', 'S1']
['Si', 'O1', 'S1', 'Si', 'O1', 'S1']
['S1', 'Si', 'S1', 'O1', 'Si', 'P1']
['O1', 'Si', 'O1', 'S1', 'Si', 'S1']
['S1', 'O1', 'Si', 'S1', 'O1', 'Si']
['O1', 'Si', 'S1', 'O1', 'Si', 'S1']
['Si', 'S1', 'O1', 'Si', 'S1', 'O1']
['S1', 'O1', 'Si', 'S1', 'O1', 'Si']
['Si', 'O1', 'S1', 'Si', 'O1', 'S1']
['O1', 'C', 'Si', 'O1', 'C', 'Si']
['S1', 'Si', 'O1', 'S1', 'Si', 'O1']
['P1', 'Si', 'O1', 'S1', 'Si', 'S1']
['Si', 'S1', 'O1', 'Si', 'S1', 'O1']
['O1', 'Si', 'S1', 'O1', 'Si', 'S1']
['Si', 'S1', 'O1', 'Si', 'S1', 'O1']
['O1', 'Si', 'C', 'O1', 'C', 'Si']
['Si', 'C', 'O1', 'C', 'Si', 'O1']
['O1', 'Si', 'S1', 'O1', 'Si', 'S1']
['Si', 'S1', 'O1', 'Si', 'S1', 'O1']
['Si', 'S1', 'O1', 'Si', 'S1', 'O1']
['S1', 'Si', 'O1', 'S1', 'Si', 'O1']
['S1', 'Si', 'O1', 'S1', 'Si', 'O1']
['S1', 'O1', 'Si', 'S1', 'O1', 'Si']
['S1', 'Si', 'O1', 'S1', 'Si', 'O1']
['O1', 'S1', 'Si', 'S1', 'O1', 'Si']
['O1', 'Si', 'C', 'O1', 'C', 'Si']
['O1', 'C', 'Si', 'O1', 'Si', 'C']
['O1', 'S1', 'C', 'O1', 'S1', 'C']
['Si', 'S1', 'O1', 'Si', 'S1', 'O1']
['O1', 'S1', 'Si', 'O1', 'S1', 'Si']
['S1', 'O1', 'Si', 'O1', 'S1', 'Si']
['O1', 'S1', 'Si', 'O1', 'S1', 'Si']
['O1', 'Si', 'S1', 'O1', 'S1', 'Si']
['O1', 'S1', 'Si', 'O1', 'S1', 'Si']
['Si', 'S1', 'O1', 'Si', 'S1', 'O1']
['O1', 'Si', 'O1', 'S1', 'Si', 'S1']
['S1', 'O1', 'P1', 'S1', 'O1', 'Si']
['S1', 'Si', 'O1', 'S1', 'Si', 'O1']
['S1', 'Si', 'O1', 'S1', 'Si', 'O1']
['O1', 'Si', 'C', 'O1', 'Si', 'C']
"""
