import os
from typing import Any

import numpy as onp
import numpy.random as onpr

import jax
import jax.numpy as jnp

from huxel.molecule import myMolecule
from huxel.utils import load_pre_opt_params

PRNGKey = Any 

# --------------------------------
#     DATA


def batch_to_list(batch: Any) -> Any:
    """Atom types of a batch to list

    Args:
        batch (Any): batch

    Returns:
        Any: transformed list
    """
    # numpy array to list
    batch = batch.tolist()
    for b in batch:
        if not isinstance(b["atom_types"], list):
            b["atom_types"] = b["atom_types"].tolist()
    return batch


def batch_to_list_class(batch:Any)->Any:
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


def get_raw_data():
    return jnp.load(
        "/h/rvargas/huxel_data_kjorner/gdb13_list_100000_training.npy",
        allow_pickle=True,
    ), jnp.load(
        "/h/rvargas/huxel_data_kjorner/gdb13_list_100000_test.npy", allow_pickle=True
    )


def get_batches(Dtr, batch_size, key):
    # Dtr = get_data()
    # Xtr,ytr = Dtr
    N = len(Dtr)

    n_complete_batches, leftover = divmod(N, batch_size)
    n_batches = n_complete_batches + bool(leftover)

    def data_stream():
        # rng = onpr.RandomState(0)
        while True:
            # perm = rng.permutation(N)
            perm = jax.random.permutation(key, jnp.arange(N))
            for i in range(n_batches):
                batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                yield Dtr[batch_idx.tolist()]

    batches = data_stream()

    return batches, n_batches


def split_trainig_test(N, key, D=None):
    if D is None:
        D, _ = get_raw_data()
    N_tot = len(D)

    # % of the total data
    if N <= 100:
        N = int(N_tot * N / 100)

    n_val = N + 1000  # extra 1000 points for validation

    # represents the absolute number of test samples
    N_tst = N_tot - N

    j_ = jnp.arange(N_tot)
    j_ = jax.random.shuffle(key, j_, axis=0)
    D_tr = D[:N]
    D_val = D[N:n_val]

    return D_tr, D_val


def get_tr_val_data(files, n_tr, subkey, batch_size):
    if os.path.isfile(files["f_data"]):
        _D = jnp.load(files["f_data"], allow_pickle=True)
        D_tr = _D.item()["Training"]
        D_val = _D.item()["Validation"]
        n_batches = _D.item()["n_batches"]
        batches, n_batches = get_batches(D_tr, batch_size, subkey)
    else:
        D_tr, D_val = split_trainig_test(n_tr, subkey)
        _, subkey = jax.random.split(subkey)  # new key
        batches, n_batches = get_batches(D_tr, batch_size, subkey)
        _, subkey = jax.random.split(subkey)  # new key
        save_tr_and_val_data(files, D_tr, D_val, n_batches)

    return D_tr, D_val, batches, n_batches, subkey
