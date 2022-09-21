import os
import time
import datetime
from xmlrpc.client import boolean

import matplotlib
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from huxel.data_utils import get_raw_data, batch_to_list_class
from huxel.huckel import f_homo_lumo_gap
from huxel.beta_functions import _f_beta
from huxel.utils import (
    get_files_names,
    get_init_params,
    get_default_params,
)

from jax.config import config

jax.config.update("jax_enable_x64", True)


def get_r_dir(method: str, list_Wdecay: list, bool_randW: bool) -> str:
    """name of results folder

    Args:
        method (str): optimization method
        list_Wdecay (list): list of parameters for weight decay
        bool_randW (bool): boolean for random initial parameters

    Returns:
        str: name of results folder
    """
    flat_list_Wdecay = "_".join(list_Wdecay)

    if len(list_Wdecay) > 0:
        r0_dir = "Results_L2reg_1E-4_{}".format(flat_list_Wdecay)
    else:
        r0_dir = "Results_L2reg_1E-4"

    if bool_randW:
        r_dir = "./{}/Results_{}_randW/".format(r0_dir, method)
    else:
        r_dir = "./{}/Results_{}/".format(r0_dir, method)

    # if not os.path.exists(r_dir):
    # os.mkdir(r_dir)
    return r_dir


def get_files_names(N: int,
                    l: int,
                    beta: str,
                    list_Wdecay: list = None,
                    opt_name: str = "AdamW",
                    randW: bool = False) -> dict:
    """dictionary with all needed files' names

    Args:
        N (int): training data
        l (int): label
        beta (str): _description_
        list_Wdecay (list, optional): list of parameters for weight decay. Defaults to None.
        opt_name (str, optional): optimization method. Defaults to "AdamW".
        randW (bool, optional): boolean for random initial parameters. Defaults to False.

    Returns:
        dict: _description_
    """
    # r_dir = './Results_xyz/'
    r_dir = get_r_dir(beta, list_Wdecay, randW)

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


def _pred(n_tr=50, l=0, beta="exp", list_Wdecay=None, bool_randW=False):
    opt_name = "AdamW"
    # files
    files = get_files_names(n_tr, l, beta, list_Wdecay, opt_name, bool_randW)

    # print info about the optimiation
    # print_head(files,n_tr,l,lr,w_decay,n_epochs,batch_size,opt_name)

    if os.path.isfile(files["f_pred"]):
        print("File {} exists!".format(files["f_pred"]))
        assert 0
    if os.path.isfile(files["f_w"]) == False:
        print("File {} does not exists!".format(files["f_w"]))
        assert 0

    # initialize parameters
    params_init = get_init_params(files)
    # params0 = get_default_params()

    # get data
    _, D = get_raw_data()
    # D = batch_to_list_class(D)

    f_beta = _f_beta(beta)

    # linear batches
    def get_batches(D, batch_size):
        N = len(D)
        n_complete_batches, leftover = divmod(N, batch_size)
        n_batches = n_complete_batches + bool(leftover)

        def data_stream():
            while True:
                # perm = rng.permutation(N)
                # perm = jax.random.permutation(key, jnp.arange(N))
                idx = jnp.arange(0, N, dtype=int)
                for i in range(n_batches):
                    batch_idx = idx[i * batch_size: (i + 1) * batch_size]
                    yield D[batch_idx.tolist()]

        batches = data_stream()
        return batches, n_batches

    # prediction
    batches, n_batches = get_batches(D, 1000)
    y_pred_tot = jnp.ones(1)
    z_pred_tot = jnp.ones(1)
    y_true_tot = jnp.ones(1)
    n = 0
    for itr in range(n_batches):
        batch = next(batches)
        d = batch_to_list_class(batch)
        temp_y_pred, temp_z_pred, temp_y_true = linear_model_pred(
            params_init, d, f_beta
        )
        print(temp_y_pred.shape, temp_z_pred.shape, temp_y_true.shape)
        y_pred_tot = jnp.append(y_pred_tot, temp_y_pred)
        z_pred_tot = jnp.append(z_pred_tot, temp_z_pred)
        y_true_tot = jnp.append(y_true_tot, temp_y_true)

    # prediction
    # y_pred, z_pred, y_true = linear_model_pred(params_init, D, f_beta)

    # prediction original parameters
    # params0 = get_default_params()
    # params_lr, params = params0
    # alpha,beta = params_lr
    # y_pred,z_pred,y_true = linear_model_pred(params0,D)

    print("finish prediction")

    R = {
        "y_pred": y_pred_tot[1:],
        "z_pred": z_pred_tot[1:],
        "y_true": y_true_tot[1:],
        # 'y0_pred': y0_pred,
        # 'z0_pred': z0_pred,
        # 'y0_true': y0_true,
    }

    jnp.save(files["f_pred"], R)
    # jnp.save('./Results/Prediction_coulson.npy',R)


# --------------------------------------------------------


def _pred_def(beta="exp"):
    opt_name = "AdamW"
    # files
    r_dir = "Results_default/"

    f_job = "huckel_xyz_default".format(opt_name)
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

    # initialize parameters
    params0 = get_default_params()

    # get data
    _, D = get_raw_data()
    D = batch_to_list_class(D)

    f_beta = _f_beta(beta)

    y_pred, z_pred, y_true = linear_model_pred(params0, D, f_beta)

    print("finish prediction")

    R = {
        "y_pred": y_pred,
        "z_pred": z_pred,
        "y_true": y_true,
        # 'y0_pred': y0_pred,
        # 'z0_pred': z0_pred,
        # 'y0_true': y0_true,
    }

    jnp.save(files["f_pred"], R)
