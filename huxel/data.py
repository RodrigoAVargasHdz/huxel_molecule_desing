import numpy as onp
import numpy.random as onpr
import os

import jax
import jax.numpy as jnp

from huxel.utils import save_tr_and_val_data


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
