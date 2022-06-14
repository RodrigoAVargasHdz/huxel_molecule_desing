import os
import time
import datetime
from tracemalloc import get_object_traceback
import numpy as onp
import argparse
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import grad, random
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
    _f_obj, get_external_field, get_molecule,
    get_objective_name,
     _preprocessing_params,
)
from huxel.minimize import opt_obj

from jax.config import config

jax.config.update("jax_enable_x64", True)

# one_hot --> pre softmax

def get_files(smile_i:int, l:int, objective: str='homo_lumo', _minimizer:str='BFGS'):
    head = f'smile{smile_i}_l_{l}_{objective}_{_minimizer}'
    files = {'head':head,
            'out': 'out_'+ head + '.txt',
            'results': head + '.npy',
    }
    return files

def _optimization_molec(l: int, molec:Any, objective: str='homo_lumo', _minimizer:str='BFGS', external_field:float=None):
    now = datetime.datetime.now()
    objective_name = get_objective_name(objective)
    f_beta = _f_beta("c")

    rng = jax.random.PRNGKey(l)
    rng, subkey = jax.random.split(rng)

    # ----------------------------------------------------------------- 
    params_extra = get_huckel_params()

    # ----------------------------------------------------------------- 
    (params_b,params_fixed_atoms), subkey = get_initial_params_b(subkey, molec, params_extra["one_pi_elec"])
    params_total = {**params_b,**params_fixed_atoms}

    init_molecule, init_params_one_hot = get_molecule(
        params_total, params_extra["one_pi_elec"]
    )



    f_obj_all = _f_obj(objective)
    external_field = get_external_field(objective,external_field)

    f_obj = lambda w: f_obj_all(w,params_fixed_atoms,params_extra, molec, f_beta,external_field)
    y_obj_initial = f_obj(params_b)
    
    # minimize
    params_b_opt, opt_molecule, results_dic = opt_obj(f_obj,params_b,params_fixed_atoms,params_extra,_minimizer) 

    # -----------------------------------------------------------------
    files = get_files(molec.id,l,objective,_minimizer)
    print(files)
    cwd = os.getcwd()
    rwd = os.path.join(cwd,'Results')
    file_r = files['results']
    resd = os.path.join(os.getcwd(),'Results')
    jnp.save(
        os.path.join(resd,file_r),
        results_dic,
        allow_pickle=True,
    )
    norm_params_b_opt = jax.tree_map(lambda x: softmax(x), params_b_opt)
    y_ev = f_obj(params_b_opt)


    # -----------------------------------------------------------------
    file_out = os.path.join(resd,files['out'])
    f = open(file_out,'w+')
    print("----------------------------------",file=f)
    print(f"l = {l}",file=f)
    print(f"{molec.smile}",file=f)
    print(f"Smile id = {molec.id}",file=f)
    print(f"(base) {molec.atom_types}",file=f)
    print('\n', file=f)
    print(f"{objective}",file=f)
    print(f"Molecule with min {objective_name}",file=f)
    if objective == 'polarizability':
        print(f"External field {external_field}",file=f)

    print(f"(initial) {objective_name}:", y_obj_initial,file=f)
    print(init_molecule,file=f)
    print(f"(opt) {objective_name}:", y_ev,file=f)
    print(opt_molecule,file=f)
    print('\n', file=f)

    print("atom X", params_extra["one_pi_elec"],file=f)
    print("init params:",file=f)
    norm_params_b = jax.tree_map(lambda x: softmax(x), params_b)
    for index, key in enumerate(norm_params_b):
        print(f"atom {key}",norm_params_b[key],file=f)
    print("final params:",file=f)
    for index, key in enumerate(norm_params_b_opt):
        print(f"atom {key}",norm_params_b_opt[key],file=f)
    print("----------------------------------",file=f)
    print(now,file=f)
    f.close()

    # assert 0
