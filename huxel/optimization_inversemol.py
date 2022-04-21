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
from huxel.huckel import f_homo_lumo_gap, f_polarizability


from jax.config import config

jax.config.update("jax_enable_x64", True)


def get_molecule(params_b, one_pi_elec):
    norm_params_b = jax.tree_map(lambda x: softmax(x), params_b)
    molecule_atoms = []
    params_one_hot = params_b.copy()
    for index, key in enumerate(norm_params_b):
        imax = jnp.argmax(norm_params_b[key])
        molecule_atoms.append(one_pi_elec[imax])
        z = -35 * jnp.ones_like(norm_params_b[key])
        z = z.at[imax].set(35.0)
        params_one_hot[key] = z
    return molecule_atoms, params_one_hot


def _optimization_molec(l: int, molec=Any, objective: str='homo_lumo'):

    rng = jax.random.PRNGKey(l)
    rng, subkey = jax.random.split(rng)

    params_extra = get_huckel_params()
    print(params_extra)

    params_b, subkey = get_initial_params_b(subkey, molec, params_extra["one_pi_elec"])

    init_molecule, init_params_one_hot = get_molecule(
        params_b, params_extra["one_pi_elec"]
    )

    f_beta = _f_beta("c")

    if objective == 'homo_lumo':
        y0_ev = f_homo_lumo_gap(params_b, params_extra, molec, f_beta)
        f_obj = lambda w: f_homo_lumo_gap(w, params_extra, molec, f_beta)
        objective_name = 'HOMO-LUMO'
    elif objective == 'polarizability':
        external_field = 0.01*jnp.ones(3)# [V/A] from J. Chem. Phys. 129, 044106 (2008)
        y0_ev = f_polarizability(params_b, params_extra, molec, f_beta,external_field)
        f_obj = lambda w: -1.*f_polarizability(w, params_extra, molec, f_beta, external_field)
        objective_name = 'Polarizability'


    def opt_gd(params_b: Any, ntr: int, lr: 0.1):
        v_and_g_obj = value_and_grad(f_obj)

        def inner_update(p, lr):
            v, grads = v_and_g_obj(p)
            inner_sgd_fn = lambda g, state: (state - lr * g)
            return jax.tree_multimap(inner_sgd_fn, grads, p), v

        params_b_opt0 = params_b.copy()
        hl_gap0 = f_obj(params_b)
        opt_molec0 = get_molecule(params_b, params_extra["one_pi_elec"])

        print(f"0, {hl_gap0}, {opt_molec0}")
        for itr in range(ntr):
            params_b, hl_gap = inner_update(params_b, lr)
            opt_molecule = get_molecule(params_b, params_extra["one_pi_elec"])

            if hl_gap < hl_gap0:
                params_b_opt0 = params_b.copy()
                hl_gap0 = hl_gap
            if opt_molecule != opt_molec0:
                opt_molec0 = opt_molecule
                print(f"{itr}, {hl_gap}, {opt_molecule}")
            elif itr % 10 == 0:
                print(f"{itr}, {hl_gap}, {opt_molecule}")

        return params_b_opt0, get_molecule(params_b_opt0, params_extra["one_pi_elec"])

    def opt_adam(params_b: Any, ntr: int, lr: 0.1):
        v_and_g_obj = value_and_grad(f_obj)

        optimizer = optax.adam(learning_rate=lr)
        opt_state = optimizer.init(params_b)

        params_b_opt0 = params_b.copy()
        hl_gap0 = f_obj(params_b)
        opt_molec0, _ = get_molecule(params_b, params_extra["one_pi_elec"])

        print(f"0, {hl_gap0}, {opt_molec0}")
        for itr in range(ntr):
            hl_gap, g_params_b = v_and_g_obj(params_b)
            updates, opt_state = optimizer.update(g_params_b, opt_state)
            params_b = optax.apply_updates(params_b, updates)
            opt_molecule = get_molecule(params_b, params_extra["one_pi_elec"])

            if hl_gap < hl_gap0:
                params_b_opt0 = params_b.copy()
                hl_gap0 = hl_gap
            if opt_molecule != opt_molec0:
                opt_molec0 = opt_molecule
                print(f"{itr}, {hl_gap}, {opt_molecule} *")
            elif itr % 25 == 0:
                print(f"{itr}, {hl_gap}, {opt_molecule}")

        return params_b_opt0, get_molecule(params_b_opt0, params_extra["one_pi_elec"])

    def opt_jaxopt(params_b_init):
        def callbackF(xi):
            print(xi)
            global Nfeval
            opt_molecule = get_molecule(xi, params_extra["one_pi_elec"])
            print("{0:4d}   {s}   {4: 3.6f}".format(Nfeval, opt_molecule, f_obj(xi)))
            Nfeval += 1

        # scipy JAXOPT
        opt = ScipyMinimize(
            method="BFGS",
            fun=f_obj,
            jit=False,
            options={"maxiter": 1},
        )

        params_b = params_b_init
        r = {}
        for i in range(20):
            opt_res = opt.run(params_b)
            params_b = opt_res[0]
            opt_molecule, params_opt_one_hot = get_molecule(
                params_b, params_extra["one_pi_elec"]
            )
            y_obj = f_obj(params_b)
            y_obj_one_hot = f_obj(params_opt_one_hot)
            print(i, y_obj, opt_molecule, y_obj_one_hot)  # opt_res,
            print(opt_res)
            r.update(
                {
                    i: {
                        "molecule": opt_molecule,
                        "params_b": params_b,
                        "objective": y_obj,
                        "objective_one_hot": y_obj_one_hot,
                    }
                }
            )

        params_b_opt = opt_res[0]

        opt_molecule, _ = get_molecule(params_b_opt, params_extra["one_pi_elec"])
        y_ev = f_obj(params_b_opt)
        return params_b_opt, opt_molecule, r

    # opt_gd(params_b, 1000, 0.5)
    # opt_adam(params_b, 1000, 0.1)
    params_b_opt, opt_molecule, r_dic = opt_jaxopt(params_b)
    # print(r_dic)

    jnp.save(
        f"/h/rvargas/huxel/Results_polarizability_X6/molecule_opt_{l}.npy",
        r_dic,
        allow_pickle=True,
    )
    # jnp.save(f"molecule_opt_benzene.npy", r_dic, allow_pickle=True)

    norm_params_b_opt = jax.tree_map(lambda x: softmax(x), params_b_opt)
    y_ev = f_obj(params_b_opt)

    print(f"Molecule with min {objective_name} gap, l = {l}")
    print(f"(initial) {objective_name}:", y0_ev)
    print(init_molecule)
    print(f"(opt) {objective_name}:", y_ev)
    print(opt_molecule)
    print("init:")
    norm_params_b = jax.tree_map(lambda x: softmax(x), params_b)
    for index, key in enumerate(norm_params_b):
        print(norm_params_b[key])
    print("finals:")
    for index, key in enumerate(norm_params_b_opt):
        print(norm_params_b_opt[key])
    print("----------------------------------")

    # assert 0


def _benzene_test(l: int, molec: Any):

    rng = jax.random.PRNGKey(l)
    rng, subkey = jax.random.split(rng)

    params_extra = get_huckel_params()

    # params_b, subkey = get_initial_params_b(subkey, molec, params_extra["one_pi_elec"])
    params_b, subkey = get_initial_params_b_benzene(
        subkey, molec, params_extra["one_pi_elec"]
    )

    init_molecule, init_params_one_hot = get_molecule(
        params_b, params_extra["one_pi_elec"]
    )
    for index, key in enumerate(params_extra["h_xy"]):
        print(key)
    f_beta = _f_beta("c")
    print(params_extra["one_pi_elec"])
    y0_ev = f_homo_lumo_gap(params_b, params_extra, molec, f_beta)
    print(y0_ev)


def main():
    atom_types = ["C", "C", "C", "C", "C", "C"]
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
    homo_lumo_grap_ref = -7.01 - (-0.42)

    molec = myMolecule(
        "benzene",
        smile,
        atom_types,
        conectivity_matrix,
        homo_lumo_grap_ref,
        jnp.ones((6, 6)),
    )
    # _optimization_molec(molec)
    _benzene_test(0, molec)


if __name__ == "__main__":
    main()

"""
def main_old():
    parser = argparse.ArgumentParser(description="opt overlap NN")
    parser.add_argument("--N", type=int, default=5, help="traning data")
    parser.add_argument("--l", type=int, default=0, help="label")
    parser.add_argument("--lr", type=float, default=2e-3, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="batches")
    parser.add_argument("--beta", type=str, default="exp_freezeR", help="beta function")
    parser.add_argument(
        "--randW", type=bool, default=False, help="random initial params"
    )

    # bathch_size = #1024#768#512#256#128#64#32
    args = parser.parse_args()
    l = args.l
    n_tr = args.N
    lr = args.lr
    batch_size = args.batch_size
    beta = args.beta
    bool_randW = args.randW

    _optimization(n_tr, batch_size, lr, l, beta, bool_randW)
"""

"""
   # optimization parameters
    # if n_tr < 100 is considered as porcentage of the training data
    w_decay = 1e-4
    n_epochs = 15
    opt_name = "AdamW"

    # files
    files = get_files_names(n_tr, l, beta, bool_randW, opt_name)

    # print info about the optimiation
    # print_head(
    #     files, n_tr, l, lr, w_decay, n_epochs, batch_size, opt_name, beta, list_Wdecay
    # )

    # training and validation data
    rng = jax.random.PRNGKey(l)
    rng, subkey = jax.random.split(rng)

    D_tr, D_val, batches, n_batches, subkey = get_tr_val_data(
        files, n_tr, subkey, batch_size
    )

    batch = batch_to_list_class(next(batches))
    for b in batch:
        print(b)

    
    # change D-val for list of myMolecules
    batch_val = batch_to_list_class(D_val)

    # initialize parameters
    if bool_randW:
        params_init, subkey = get_random_params(files, subkey)
    else:
        params_init = get_init_params(files)

    params_bool = get_params_bool(list_Wdecay)

    # select the function for off diagonal elements for H
    f_beta = _f_beta(beta)
    # f_loss_batch_ = lambda params,batch: f_loss_batch(params,batch,f_beta)
    grad_fn = value_and_grad(f_loss_batch, argnums=(0,))

    # OPTAX ADAM
    # schedule = optax.exponential_decay(init_value=lr,transition_steps=25,decay_rate=0.1)
    optimizer = optax.adamw(learning_rate=lr, mask=params_bool)
    opt_state = optimizer.init(params_init)
    params = params_init

    # @jit
    def train_step(params, optimizer_state, batch, f_beta):
        loss, grads = grad_fn(params, batch, f_beta)
        updates, opt_state = optimizer.update(grads[0], optimizer_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    loss_val0 = 1e16
    f_params = params_init
    loss_tr_ = []
    loss_val_ = []
    for epoch in range(n_epochs + 1):
        start_time_epoch = time.time()
        loss_tr_epoch = []
        for _ in range(n_batches):
            batch = batch_to_list_class(next(batches))
            params, opt_state, loss_tr = train_step(params, opt_state, batch, f_beta)
            loss_tr_epoch.append(loss_tr)

        loss_tr_mean = jnp.mean(jnp.asarray(loss_tr_epoch).ravel())
        loss_val = f_loss_batch(params, batch_val, f_beta)

        f = open(files["f_out"], "a+")
        time_epoch = time.time() - start_time_epoch
        print(epoch, loss_tr, loss_val, time_epoch, file=f)
        f.close()

        loss_tr_.append(loss_tr_mean)
        loss_val_.append(loss_val)

        if loss_val < loss_val0:
            loss_val0 = loss_val
            f_params = update_params_all(params)
            jnp.save(files["f_w"], f_params)
            # jnp.save(get_params_file_itr(files, epoch), f_params)

    save_tr_and_val_loss(files, loss_tr_, loss_val_, n_epochs + 1)

    print_tail(files)

"""
# label_parmas_all = ['alpha', 'beta', 'h_x', 'h_xy', 'r_xy', 'y_xy']


# def f_loss_batch(params_tot, batch, f_beta):
#     params_tot = update_params_all(params_tot)
#     y_pred, z_pred, y_true = linear_model_pred(params_tot, batch, f_beta)

#     # diff_y = jnp.abs(y_pred-y_true)
#     diff_y = (y_pred - y_true) ** 2
#     return jnp.mean(diff_y)
