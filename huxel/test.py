from logging import raiseExceptions
import os
import time
import datetime
import numpy as onp
import argparse

import jax
import jax.numpy as jnp
from jax import random
from jax import jit,vmap,lax,value_and_grad,jacfwd
from jax.tree_util import tree_flatten, tree_unflatten, tree_multimap

from flax import optim

from huxel.molecule import myMolecule
from huxel.data import get_raw_data,get_batches
from huxel.huckel import linear_model_pred, _construct_huckel_matrix, _f_beta
from huxel.utils import get_default_params, get_files_names, batch_to_list_class, get_init_params, get_random_params
from huxel.utils import print_head, print_tail,get_params_file_itr
from huxel.utils import save_tr_and_val_data, save_tr_and_val_loss

from jax.config import config
jax.config.update('jax_enable_x64', True)


def f_loss_batch(params_tot,batch):

    y_pred,z_pred,y_true = linear_model_pred(params_tot,batch)

    # diff_y = jnp.abs(y_pred-y_true)
    diff_y = (y_pred-y_true)**2
    return jnp.mean(diff_y)

def _test(n_tr=50,batch_size=100,lr=2E-3,l=0):

    # optimization parameters
    # if n_tr < 100 is considered as porcentage of the training data 
    w_decay = 1E-4
    n_epochs = 250
    opt_name = 'Adam'

    # files
    files = get_files_names(n_tr,l,opt_name)

    # print info about the optimiation
    # print_head(files,n_tr,l,lr,w_decay,n_epochs,batch_size,opt_name)

    # training and validation data
    rng = jax.random.PRNGKey(l)
    rng, subkey = jax.random.split(rng)

    if os.path.isfile(files['f_data']):
        _D = jnp.load(files['f_data'],allow_pickle=True)
        D_tr = _D.item()['Training']
        D_val = _D.item()['Validation']
        n_batches = _D.item()['n_batches']
        batches, n_batches = get_batches(D_tr,batch_size,subkey)
    else: 
        raise ValueError('File {} with data not found!'.format(files['f_data']))

    # change D-val for list of myMolecules
    batch_val = batch_to_list_class(D_val)

    # change D-test for list of myMolecules
    _,D_test = get_raw_data()
    batch_test = batch_to_list_class(D_test)

    # initialize parameters
    params_init = get_init_params(files)
    params_lr, params = params_init
    # params_init,subkey = get_random_params(files,subkey)

    beta = 'exp_freezeR'
    f_beta = _f_beta(beta)

    batch =  next(batches)
    b = batch[0]
    molec = myMolecule(b['id'],b['smiles'],b['atom_types'],b['conectivity_matrix'],b['homo_lumo_grap_ref'],b['dm'])
    print(molec.atom_types)
    huckel_matrix = _construct_huckel_matrix(params,molec,f_beta)
    print(huckel_matrix[0])
    print(params[0])
    
    print('----------------')
    beta = 'constant'
    f_beta = _f_beta(beta)
    params_init = get_default_params()
    params_lr, params = params_init
    print(params[0])
    huckel_matrix = _construct_huckel_matrix(params,molec,f_beta)
    print(huckel_matrix[0])
    

def main():
    parser = argparse.ArgumentParser(description='opt overlap NN')
    parser.add_argument('--N', type=int, default=25, help='traning data')
    parser.add_argument('--l', type=int, default=0, help='label')
    parser.add_argument('--lr', type=float, default=2E-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batches')

    # bathch_size = #1024#768#512#256#128#64#32
    # args = parser.parse_args()
    # l = args.l
    # n_tr = args.N
    # lr = args.lr
    # batch_size = args.batch_size

    n_tr = 25
    l = 0
    beta = 'exp'
    _test(n_tr,l,beta)

if __name__ == "__main__":
    main()
