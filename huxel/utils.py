import os
import datetime
from typing import Any
import numpy as onp

import jax
import jax.numpy as jnp
from jax import jit

from huxel.molecule import myMolecule
from huxel.parameters import H_X, H_XY, R_XY, Y_XY, N_ELECTRONS
from huxel.parameters import h_x_tree, h_x_flat, h_xy_tree, h_xy_flat
from huxel.parameters import f_dif_pytrees, f_div_pytrees

# r_dir = './Results_xyz_constant_random_params/'
#'./Results_xyz/'
#'./Results_xyz_linear/'
#'./Results_xyz_constant/'
#'./Results_xyz_constant_random_params'/


def get_huckel_params():

    # one_pi_elec = []
    # h_x_red = {}
    # for index, key in enumerate(N_ELECTRONS):
    #     if N_ELECTRONS[key] == 1:
    #         one_pi_elec.append(key)
    #         h_x_red.update({key: H_X[key]})

    # one_pi_elec.append("X")
    # h_x_red.update({"X": 0.0})

    # JCP C,N,P
    one_pi_elec = ["C", "N1", "P1"]
    h_x_red = {}
    for index, key in enumerate(one_pi_elec):
        if N_ELECTRONS[key] == 1:
            h_x_red.update({key: H_X[key]})

    one_pi_elect_ij = []
    h_xy_red = {}
    for i, ni in enumerate(one_pi_elec):
        for j, nj in enumerate(one_pi_elec[i:]):
            one_pi_elect_ij.append([ni, nj])
            key = frozenset([ni, nj])
            if ni != "X" and nj != "X":
                h_xy_red.update({key: H_XY[key]})
            else:
                h_xy_red.update({key: 0.0})

    a, b = get_init_params_lr()

    params_huckel = {
        "h_x": h_x_red,
        "h_xy": h_xy_red,
        "one_pi_elec": one_pi_elec,
        "alpha": a,
        "beta": b,
    }

    return params_huckel


def get_initial_params_b(subkey: Any, molec: Any, one_pi_elec: Any):

    params_b = {}
    for ni, c in enumerate(molec.atom_types):
        b_temp = jax.random.uniform(
            subkey, shape=(len(one_pi_elec),), minval=0.0, maxval=2.0
        )
        _, subkey = jax.random.split(subkey)
        params_b.update({ni: b_temp})
    return params_b, subkey


def get_initial_params_b_benzene(subkey: Any, molec: Any, one_pi_elec: Any):

    params_b = {}
    for ni, c in enumerate(molec.atom_types):
        b_temp = jnp.hstack([35.0, -35.0 * jnp.ones(len(one_pi_elec) - 1)]).ravel()
        # b_temp += jax.random.uniform(
        #     subkey, shape=(len(one_pi_elec),), minval=-1.0, maxval=2.0
        # )
        params_b.update({ni: b_temp})
    return params_b, subkey


# OLD

# --------------------------------
#     FILES
def get_r_dir_old(method):
    if method == "exp":
        return "./Results_xyz/"
    elif method == "linear":
        return "./Results_xyz_linear/"
    elif method == "constant":
        return "./Results_xyz_constant/"
    elif method == "exp_freezeR":
        return "./Results_xyz_freezeR/"
    elif method == "randW":
        return "./Results_xyz_constant_random_params/"


def get_r_dir(method, bool_randW):
    if bool_randW:
        r_dir = "./Results_{}_randW/".format(method)
    else:
        r_dir = "./Results_{}/".format(method)

    if not os.path.exists(r_dir):
        os.mkdir(r_dir)
    return r_dir


def get_files_names(N, l, beta, randW, opt_name="Adam"):
    # r_dir = './Results_xyz/'
    r_dir = get_r_dir(beta, randW)

    f_job = "huckel_xyz_N_{}_l_{}_{}".format(N, l, opt_name)
    f_out = "{}/out_{}.txt".format(r_dir, f_job)
    f_w = "{}/parameters_{}.npy".format(r_dir, f_job)
    f_pred = "{}/Prediction_{}.npy".format(r_dir, f_job)
    f_data = "{}/Data_{}.npy".format(r_dir, f_job)
    f_loss_opt = "{}/Loss_tr_val_itr_{}.npy".format(r_dir, f_job)

    files = {
        "f_job": f_job,
        "f_out": f_out,
        "f_w": f_w,
        "f_pred": f_pred,
        "f_data": f_data,
        "f_loss_opt": f_loss_opt,
        "r_dir": r_dir,
    }
    return files


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
#     DATA
def batch_to_list(batch):
    # numpy array to list
    batch = batch.tolist()
    for b in batch:
        if not isinstance(b["atom_types"], list):
            b["atom_types"] = b["atom_types"].tolist()
    return batch


def batch_to_list_class(batch):
    #     pytree to class-myMolecule
    batch = batch_to_list(batch)
    batch_ = []
    for b in batch:
        m = myMolecule(
            b["id"],
            b["smiles"],
            b["atom_types"],
            b["conectivity_matrix"],
            b["homo_lumo_grap_ref"],
            b["dm"],
        )
        batch_.append(m)
    return batch_


def save_tr_and_val_data(files, D_tr, D_val, n_batches):
    file = files["f_data"]
    D = {
        "Training": D_tr,
        "Validation": D_val,
        "n_batches": n_batches,
    }
    jnp.save(file, D, allow_pickle=True)


def save_tr_and_val_loss(files, loss_tr, loss_val, n_epochs):
    epochs = jnp.arange(n_epochs + 1)
    loss_tr_ = jnp.asarray(loss_tr).ravel()
    loss_val_ = jnp.asarray(loss_val).ravel()

    if os.path.isfile(files["f_loss_opt"]):
        pre_epochs, pre_loss_tr, pre_loss_val = load_pre_opt_params(files)
        loss_tr_ = jnp.append(loss_tr_, pre_loss_tr)
        loss_val_ = jnp.append(loss_val_, pre_loss_val)
        # epochs = jnp.arange(0,pre_epochs.shape[0] + epochs.shape[0])
        epochs = jnp.arange(loss_tr_.shape[0])

    D = {
        "epoch": epochs,
        "loss_tr": loss_tr_,
        "loss_val": loss_val_,
        # 'loss_test':loss_test,
    }
    jnp.save(files["f_loss_opt"], D, allow_pickle=True)


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


def get_init_params_lr():
    params_lr = onp.load("huxel/data/lr_params.npy", allow_pickle=True)
    alpha = params_lr.item()["alpha"].reshape(1,)
    beta = params_lr.item()["beta"]
    return jnp.asarray(alpha), jnp.asarray(beta)


def get_y_xy_random(key):
    y_xy_flat, y_xy_tree = jax.tree_util.tree_flatten(Y_XY)
    y_xy_random_flat = jax.random.uniform(
        key, shape=(len(y_xy_flat),), minval=-0.1, maxval=0.1
    )
    y_xy_random_flat = y_xy_random_flat + 0.3
    _, subkey = jax.random.split(key)
    y_xy_random = jax.tree_util.tree_unflatten(y_xy_tree, y_xy_random_flat)
    return y_xy_random, subkey


def get_params_pytrees(alpha, beta, h_x, h_xy, r_xy, y_xy):
    params_init = {
        "alpha": alpha,
        "beta": beta,
        "h_x": h_x,
        "h_xy": h_xy,
        "r_xy": r_xy,
        "y_xy": y_xy,
    }
    return params_init


# include alpha y beta in the new parameters
def get_default_params():
    params_lr = get_init_params_lr()
    # params_init = (params_lr,params_coulson)
    params_init = {
        "alpha": params_lr[0],
        "beta": params_lr[1],
        "h_x": H_X,
        "h_xy": H_XY,
        "r_xy": R_XY,
        "y_xy": Y_XY,
    }
    return get_params_pytrees(params_lr[0], params_lr[1], H_X, H_XY, R_XY, Y_XY)


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


def get_random_params(files, key):
    if not os.path.isfile(files["f_w"]):
        params_init = get_default_params()
        # params_lr,params_coulson = params_init

        alpha_random = jax.random.uniform(key, shape=(1,), minval=-1.0, maxval=1.0)
        _, subkey = jax.random.split(key)
        beta_random = jax.random.uniform(subkey, shape=(1,), minval=-1.0, maxval=1.0)
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
            alpha_random, beta_random, h_x_random, h_xy_random, r_xy_random, y_xy_random
        )

        f = open(files["f_out"], "a+")
        print("Random initial parameters", file=f)
        print("-----------------------------------", file=f)
        f.close()
        return params, subkey
    else:
        params = get_init_params(files)
        return params, key


def get_init_params(files):
    params_init = get_default_params()
    if os.path.isfile(files["f_w"]):
        params = jnp.load(files["f_w"], allow_pickle=True)
        print(files["f_w"])
        # params_lr,params_coulson = params
        alpha = params.item()["alpha"]
        beta = params.item()["beta"]
        h_x = params.item()["h_x"]
        h_xy = params.item()["h_xy"]
        r_xy = params.item()["r_xy"]
        y_xy = params.item()["y_xy"]

        params = get_params_pytrees(alpha, beta, h_x, h_xy, r_xy, y_xy)

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
def update_h_x(h_x):
    xc = h_x["C"]
    xc_tree = jax.tree_unflatten(h_x_tree, xc * jnp.ones_like(jnp.array(h_x_flat)))
    return jax.tree_map(f_dif_pytrees, xc_tree, h_x)


@jit
def update_h_xy(h_xy):
    key = frozenset(["C", "C"])
    xcc = h_xy[key]
    xcc_tree = jax.tree_unflatten(h_xy_tree, xcc * jnp.ones_like(jnp.array(h_xy_flat)))
    return jax.tree_map(f_div_pytrees, xcc_tree, h_xy)


@jit
def update_params_all(params):
    h_x = h_x = update_h_x(params["h_x"])
    h_xy = update_h_xy(params["h_xy"])
    new_params = get_params_pytrees(
        params["alpha"], params["beta"], h_x, h_xy, params["r_xy"], params["y_xy"]
    )
    return new_params
