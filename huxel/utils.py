import os
import datetime
from typing import Any
import numpy as onp

import jax
import jax.numpy as jnp
from jax import jit
from jax.nn import one_hot

from huxel.molecule import myMolecule
from huxel.parameters import H_X, H_XY, N_ELECTRONS
from huxel.parameters import R_XY_Bohr, R_XY_AA
from huxel.parameters import Y_XY_Bohr, Y_XY_AA
from huxel.parameters import h_x_tree, h_x_flat, h_xy_tree, h_xy_flat, r_xy_tree, r_xy_flat
from huxel.parameters import f_dif_pytrees, f_div_pytrees, f_mult_pytrees, f_sum_pytrees
from huxel.parameters import au_to_eV, Bohr_to_AA

from huxel.huckel import f_homo_lumo_gap, f_polarizability

PRNGKey = Any 

def get_huckel_params(objective:str="homo_lumo",bool_preopt:bool=True ):

    # one_pi_elec = []
    # h_x_red = {}
    # for index, key in enumerate(N_ELECTRONS):
    #     if N_ELECTRONS[key] == 1:
    #         one_pi_elec.append(key)
    #         h_x_red.update({key: H_X[key]})

    # one_pi_elec.append("X")
    # h_x_red.update({"X": 0.0})

    # JCP C,N,P

    if bool_preopt:
        params = get_pre_opt_params(objective)
        h_x = params['h_x']
        h_xy = params['h_xy']
    else:
        h_x = H_X
        h_xy = H_XY

    one_pi_elec = ["C", "N1", "P1"]
    h_x_red = {}
    for index, key in enumerate(one_pi_elec):
        if N_ELECTRONS[key] == 1:
            h_x_red.update({key: h_x[key]})

    one_pi_elect_ij = []
    h_xy_red = {}
    for i, ni in enumerate(one_pi_elec):
        for j, nj in enumerate(one_pi_elec[i:]):
            one_pi_elect_ij.append([ni, nj])
            key = frozenset([ni, nj])
            if ni != "X" and nj != "X":
                h_xy_red.update({key: h_xy[key]})
            else:
                h_xy_red.update({key: 0.0})

    hl_a, hl_b = get_init_params_homo_lumo()
    pol_a, pol_b = get_init_params_polarizability()

    params_huckel = {
        "h_x": h_x_red,
        "h_xy": h_xy_red,
        "one_pi_elec": one_pi_elec,
        "hl_params":{"a": hl_a,"b": hl_b},
        "pol_params":{"a": pol_a,"b": pol_b},
    }

    return params_huckel


def get_initial_params_b(subkey:Any, molec:Any, one_pi_elec:list):

    params_b = {}
    params_fixed_atoms = {}
    for ni, c in enumerate(molec.atom_types):
        if c == 'X': 
            b_temp = jax.random.uniform(
                subkey, shape=(len(one_pi_elec),), minval=-1.0, maxval=1.0
            )
            _, subkey = jax.random.split(subkey)
            params_b.update({ni: b_temp})
        else:
            i = one_pi_elec.index(c)
            b_temp = one_hot(i,len(one_pi_elec))
            params_fixed_atoms.update({ni:b_temp})

    return (params_b,params_fixed_atoms), subkey


def get_initial_params_b_benzene(subkey:Any, molec:Any, one_pi_elec:Any):

    params_b = {}
    for ni, c in enumerate(molec.atom_types):
        b_temp = jnp.hstack([35.0, -35.0 * jnp.ones(len(one_pi_elec) - 1)]).ravel()
        # b_temp += jax.random.uniform(
        #     subkey, shape=(len(one_pi_elec),), minval=-1.0, maxval=2.0
        # )
        params_b.update({ni: b_temp})
    return params_b, subkey


def get_molecule(params_b, one_pi_elec):
    norm_params_b = jax.tree_map(lambda x: jax.nn.softmax(x), params_b)
    molecule_atoms = []
    params_one_hot = params_b.copy()
    for index, key in enumerate(norm_params_b):
        imax = jnp.argmax(norm_params_b[key])
        molecule_atoms.append(one_pi_elec[imax])
        z = -35 * jnp.ones_like(norm_params_b[key])
        z = z.at[imax].set(35.0)
        params_one_hot[key] = z
    return molecule_atoms, params_one_hot

def _f_obj(objective:str):
    if objective == 'homo_lumo':
        def wrapper(params_b:Any,*args):
            args_new = {**params_b,**args[0]}
            return f_homo_lumo_gap(args_new,*args[1:-1])
        return wrapper  
    elif objective == 'polarizability':
        def wrapper(params_b:Any,*args):
            args_new = {**params_b,**args[0]}
            return -f_polarizability(args_new,*args[1:])
        return wrapper  

def get_external_field(objective:str='homo_lumo',magnitude:Any=0.):
    if objective.lower() == 'polarizability' or objective.lower() == 'pol':
        if isinstance(magnitude, float):
            return magnitude*jnp.ones(3)
        elif isinstance(magnitude, list):
            return jnp.asarray(magnitude)
        else: #default
            return jnp.zeros(3)
    else:
        return None   

def _preprocessing_params(objective:str):
    if objective.lower() == 'homo_lumo' or objective.lower() == 'hl':
        def wrapper(*args):
            return normalize_params_wrt_C(*args)
        return wrapper         
    elif objective.lower() == 'polarizability' or objective.lower() == 'pol':
        def wrapper(*args):
            return normalize_params_polarizability(*args)
        return wrapper  

def get_objective_name(objective:str):
    if objective == 'homo_lumo' or objective == 'hl':
        return 'HOMO-LUMO'
    elif objective == 'polarizability' or objective == 'pol':
        return 'Polarizability'


# --------------------------------
#     FILES

def get_r_dir(method:str, bool_randW:bool):
    if bool_randW:
        r_dir = "./Results_{}_randW/".format(method)
    else:
        r_dir = "./Results_{}/".format(method)

    if not os.path.exists(r_dir):
        os.mkdir(r_dir)
    return r_dir


def get_params_file_itr(files, itr):
    # r_dir = './Results_xyz/'
    f_job = files["f_job"]
    r_dir = files["r_dir"]
    file_ = "{}/params_{}_itr_{}.npy".format(r_dir, f_job, itr)
    return file_

# --------------------------------
#     HEAD OF FILE
def print_head(
    files, N, l, lr, w_decay, n_epochs, batch_size, opt_name, beta, list_Wdecay
):
    f = open(files["f_out"], "a+")
    print("-----------------------------------", file=f)
    print("Starting time", file=f)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), file=f)
    print("-----------------------------------", file=f)
    print(files["f_out"], file=f)
    print("N = {}, l = {}".format(N, l), file=f)
    print("lr = {}, w decay = {}".format(lr, w_decay), file=f)
    print("batch size = {}".format(batch_size), file=f)
    print("N Epoch = {}".format(n_epochs), file=f)
    print("Opt method = {}".format(opt_name), file=f)
    print("f beta: {}".format(beta), file=f)
    print("W Decay {}: ".format(list_Wdecay), file=f)
    print("-----------------------------------", file=f)
    f.close()


#     TAIL OF FILE
def print_tail(files):
    f = open(files["f_out"], "a+")
    print("-----------------------------------", file=f)
    print("Finish time", file=f)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), file=f)
    print("-----------------------------------", file=f)
    f.close()


# --------------------------------
#     PARAMETERS
def load_pre_opt_params(files):
    if os.path.isfile(files["f_loss_opt"]):
        D = jnp.load(files["f_loss_opt"], allow_pickle=True)
        epochs = D.item()["epoch"]
        loss_tr = D.item()["loss_tr"]
        loss_val = D.item()["loss_val"]
        return epochs, loss_tr, loss_val


def random_pytrees(_pytree, key, minval=-1.0, maxval=1.0):
    _pytree_flat, _pytree_tree = jax.tree_util.tree_flatten(_pytree)
    _pytree_random_flat = jax.random.uniform(
        key, shape=(len(_pytree_flat),), minval=minval, maxval=maxval
    )
    _new_pytree = jax.tree_util.tree_unflatten(_pytree_tree, _pytree_random_flat)
    _, subkey = jax.random.split(key)
    return _new_pytree, subkey

def get_init_params_homo_lumo():
    # params_lr = onp.load("huxel/data/lr_params.npy", allow_pickle=True)
    alpha = jnp.array([-2.252276274030775]) #params_lr.item()["alpha"] * jnp.ones(1)
    beta = jnp.array([2.053257355175381]) #params_lr.item()["beta"]
    return jnp.array(alpha), jnp.array(beta)


def get_init_params_polarizability():
    # params_lr = onp.load("huxel/data/lr_params.npy", allow_pickle=True)
    alpha = jnp.ones(1)
    beta = jnp.array([116.20943344747411]) #params_lr.item()["beta"]
    return jnp.array(alpha), jnp.array(beta)

def get_y_xy_random(key):
    y_xy_flat, y_xy_tree = jax.tree_util.tree_flatten(Y_XY_AA)
    y_xy_random_flat = jax.random.uniform(
        key, shape=(len(y_xy_flat),), minval=-0.1, maxval=0.1
    )
    y_xy_random_flat = y_xy_random_flat + 0.3
    _, subkey = jax.random.split(key)
    y_xy_random = jax.tree_util.tree_unflatten(y_xy_tree, y_xy_random_flat)
    return y_xy_random, subkey

def get_params_pytrees(hl_a:float, hl_b:float, pol_a:float, pol_b:float, h_x:dict, h_xy:dict, r_xy:dict, y_xy:dict):
    params_init = {
        "hl_params":{"a": hl_a,"b": hl_b},
        "pol_params":{"a": pol_a,"b": pol_b},
        "h_x": h_x,
        "h_xy": h_xy,
        "r_xy": r_xy,
        "y_xy": y_xy,
    }
    return params_init

# include alpha y beta in the new parameters
def get_default_params(objective:str="homo_lumo"):
    params_hl = get_init_params_homo_lumo() #homo_lumo
    params_pol = get_init_params_polarizability() #(jnp.ones(1), jnp.ones(1))
    
    if objective.lower() == 'homo_lumo' or objective.lower() == 'hl':
        R_XY = R_XY_AA
        Y_XY = Y_XY_AA
    elif objective.lower() == 'polarizability' or objective.lower() == 'pol':
        R_XY = R_XY_Bohr
        Y_XY = Y_XY_Bohr

    return get_params_pytrees(params_hl[0], params_hl[1], params_pol[0], params_pol[1], H_X, H_XY, R_XY, Y_XY)



def get_params_bool(params_wdecay_):
    """return params_bool where weight decay will be used. array used in masks in OPTAX"""
    params = get_default_params()
    params_bool = params
    params_flat, params_tree = jax.tree_util.tree_flatten(params)
    params_bool = jax.tree_util.tree_unflatten(
        params_tree, jnp.zeros(len(params_flat), dtype=bool)
    )  # all FALSE

    if len(params_wdecay_) > 0:
        for pb in params_wdecay_:  # ONLY TRUE
            if isinstance(params[pb], dict):
                p_flat, p_tree = jax.tree_util.tree_flatten(params[pb])
                params_bool[pb] = jax.tree_util.tree_unflatten(
                    p_tree, jnp.ones(len(p_flat), dtype=bool)
                )
            else:
                params_bool[pb] = jnp.ones(params[pb].shape, dtype=bool)

    return params_bool


def get_random_params(files:dict, key:PRNGKey):
    if not os.path.isfile(files["f_w"]):
        params_init = get_default_params()
        # params_lr,params_coulson = params_init

        hl_a_random = jax.random.uniform(key, shape=(1,), minval=-1.0, maxval=1.0)
        _, subkey = jax.random.split(key)
        hl_b_random = jax.random.uniform(subkey, shape=(1,), minval=-1.0, maxval=1.0)
        _, subkey = jax.random.split(subkey)

        pol_a_random = jax.random.uniform(key, shape=(1,), minval=-1.0, maxval=1.0)
        _, subkey = jax.random.split(key)
        pol_b_random = jax.random.uniform(subkey, shape=(1,), minval=-1.0, maxval=1.0)
        _, subkey = jax.random.split(subkey)

        h_x = params_init["h_x"]
        h_x_random, subkey = random_pytrees(h_x, subkey, -1.0, 1.0)

        h_xy = params_init["h_xy"]
        h_xy_random, subkey = random_pytrees(h_xy, subkey, 0.0, 1.0)

        r_xy = params_init["r_xy"]
        r_xy_random, subkey = random_pytrees(r_xy, subkey, 1.0, 3.0)

        y_xy = params_init["y_xy"]
        y_xy_random, subkey = get_y_xy_random(subkey)

        params = get_params_pytrees(
            hl_a_random, hl_b_random, pol_a_random , pol_b_random, h_x_random, h_xy_random, r_xy_random, y_xy_random
        )

        f = open(files["f_out"], "a+")
        print("Random initial parameters", file=f)
        print("-----------------------------------", file=f)
        f.close()
        return params, subkey
    else:
        params = get_init_params(files)
        return params, key

def get_pre_opt_params(objective:str="homo_lumo"):
    cwd = os.getcwd()
    params_d = os.path.join(cwd, "huxel/data")

    params_onp = onp.load(os.path.join(params_d, f"params_opt_{objective}.npy"), allow_pickle=True)

    hl_a = params_onp.item()["hl_params"]["a"]
    hl_b = params_onp.item()["hl_params"]["b"]
    pol_a = params_onp.item()["pol_params"]["a"]
    pol_b = params_onp.item()["pol_params"]["b"]

    h_x = params_onp.item()["h_x"]
    h_xy = params_onp.item()["h_xy"]
    r_xy = params_onp.item()["r_xy"]
    y_xy = params_onp.item()["y_xy"]

    params = get_params_pytrees(hl_a, hl_b, pol_a, pol_b, h_x, h_xy, r_xy, y_xy)

    if objective.lower() == 'homo_lumo' or objective.lower() == 'hl':
        params = normalize_params_wrt_C(params) 
    elif objective.lower() == 'polarizability' or objective.lower() == 'pol': 
        params = normalize_params_polarizability(params)

    return params

def get_init_params(files:dict, obs:str="homo_lumo"):
    # params_init = get_default_params()
    params_init = get_pre_opt_params()
    if os.path.isfile(files["f_w"]):
        params = onp.load(files["f_w"], allow_pickle=True)
        print(files["f_w"])
        # params_lr,params_coulson = params
        hl_a = params.item()["hl_params"]["a"]
        hl_b = params.item()["hl_params"]["b"]
        pol_a = params.item()["pol_params"]["a"]
        pol_b = params.item()["pol_params"]["b"]

        h_x = params.item()["h_x"]
        h_xy = params.item()["h_xy"]
        r_xy = params.item()["r_xy"]
        y_xy = params.item()["y_xy"]

        params = get_params_pytrees(hl_a, hl_b, pol_a, pol_b, h_x, h_xy, r_xy, y_xy)

        f = open(files["f_out"], "a+")
        print("Reading parameters from prev. optimization", file=f)
        print("-----------------------------------", file=f)
        f.close()

        return params
    else:
        f = open(files["f_out"], "a+")
        print("Standard initial parameters", file=f)
        print("-----------------------------------", file=f)
        f.close()
        return params_init


@jit
def update_h_x(h_x:dict):
    xc = h_x["C"]
    xc_tree = jax.tree_unflatten(h_x_tree, xc * jnp.ones_like(jnp.array(h_x_flat)))
    return jax.tree_map(f_dif_pytrees, xc_tree, h_x)

@jit
def update_h_xy(h_xy:dict):
    key = frozenset(["C", "C"])
    xcc = h_xy[key]
    xcc_tree = jax.tree_unflatten(h_xy_tree, xcc * jnp.ones_like(jnp.array(h_xy_flat)))
    return jax.tree_map(f_div_pytrees, xcc_tree, h_xy)

@jit
def update_h_x_au_to_eV(h_x:dict, pol_a:Any):
    x_tree = jax.tree_unflatten(h_x_tree, (pol_a/au_to_eV) * jnp.ones_like(jnp.array(h_x_flat)))
    return jax.tree_map(f_mult_pytrees, x_tree, h_x)

@jit
def update_h_xy_au_to_eV(h_xy:dict, pol_a:Any):
    xy_tree = jax.tree_unflatten(h_xy_tree, (pol_a/au_to_eV) * jnp.ones_like(jnp.array(h_xy_flat)))
    return jax.tree_map(f_mult_pytrees, xy_tree, h_xy)

@jit
def update_r_xy_Bohr_to_AA(r_xy:dict):
    xy_tree = jax.tree_unflatten(r_xy_tree, (Bohr_to_AA) * jnp.ones_like(jnp.array(r_xy_flat)))
    return jax.tree_map(f_div_pytrees, xy_tree, r_xy)

@jit
def normalize_params_wrt_C(params:dict):
    h_x =  update_h_x(params["h_x"])
    h_xy = update_h_xy(params["h_xy"])

    new_params = get_params_pytrees(
        params["hl_params"]["a"], 
        params["hl_params"]["b"], 
        params["pol_params"]["a"], 
        params["pol_params"]["b"], 
        h_x, 
        h_xy, 
        params["r_xy"], 
        params["y_xy"],
    )
    return new_params

@jit
def normalize_params_polarizability(params:dict):
    params_norm_c = normalize_params_wrt_C(params)
    pol_a = params_norm_c["pol_params"]["a"]

    h_x = update_h_x_au_to_eV(params_norm_c["h_x"], pol_a)
    h_xy = update_h_xy_au_to_eV(params_norm_c["h_xy"], pol_a)
    
    new_params = get_params_pytrees(
        params_norm_c["hl_params"]["a"], 
        params_norm_c["hl_params"]["b"], 
        params_norm_c["pol_params"]["a"], 
        params_norm_c["pol_params"]["b"], 
        h_x, 
        h_xy, 
        params_norm_c["r_xy"], 
        params_norm_c["y_xy"],
    )
    return new_params
