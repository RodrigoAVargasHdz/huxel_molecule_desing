from binascii import Incomplete
import os
import time
import datetime
from tracemalloc import get_object_traceback
import weakref
import numpy as onp
import argparse
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import value_and_grad

import jaxopt
from jaxopt import ScipyMinimize
import optax

from huxel.utils import get_molecule


def opt_obj(f_obj:Callable,params_b_init:Any,params_fixed_atoms:Any,params_extra:Any,opt_method:str='BGFS',ntr:int=15,lr:float=2E-1):
    opt_step = wrapper_opt_method(f_obj,opt_method,lr)
    params_b = params_b_init

    r = {}
    params_b_opt = params_b_init.copy()
    y_obj_opt = jnp.inf
    molecule_opt = []
    for itr in range(0,ntr+1):
        params_b, y_obj = opt_step(params_b)
        molecule_itr, params_b_one_hot = get_molecule(
            {**params_b,**params_fixed_atoms}, params_extra["one_pi_elec"]
        )
        y_obj = f_obj(params_b)
        y_obj_one_hot = f_obj(params_b_one_hot)

        r.update(
            {
                itr: {
                    "molecule": molecule_itr,
                    "params_b": params_b,
                    "objective": y_obj,
                    "objective_one_hot": y_obj_one_hot,
                }
            }
        )

        if y_obj < y_obj_opt:
            y_obj_opt = y_obj
            params_b_opt = params_b
            molecule_opt = molecule_itr
        if itr % 5 == 0:
            print(f"{itr}, {y_obj}, {molecule_itr}, {y_obj_one_hot}")
            print(jax.tree_map(lambda x: jax.nn.softmax(x), params_b))
        
    print('-------------------------------------------------')
    return params_b_opt, molecule_opt, r


def get_max(x:dict):
    flat, tree = jax.tree_flatten(x)
    return jnp.max(jnp.asarray(flat))
def get_sum(x:dict):
    flat, tree = jax.tree_flatten(x)
    return jnp.sum(jnp.asarray(flat))

def f_obj_reg(f_obj:Callable):
    def wrapper(params_b:Any,*args):
        norm_parmas_b = jax.tree_map(lambda x: jnp.linalg.norm(x,ord=1), params_b)    
        max_norm_parmas_b = get_max(jax.lax.stop_gradient(norm_parmas_b))
        # reg_coeff = get_sum(norm_parmas_b) 
        return f_obj(params_b) + get_sum(norm_parmas_b)
    return wrapper
# f_obj_new = lambda params_b: f_obj(params_b) + f_obj_reg(params_b)

def wrapper_opt_method(f_obj:Callable,method:str='BGFS',lr:float=2E-1):
    if method == 'BGFS':
        def wrapper(params_b:Any,*args):
                optimizer = ScipyMinimize(
                    method="BFGS",
                    fun=f_obj,
                    jit=False,
                    options={"maxiter": 1},
                    )
                res = optimizer.run(params_b)
                y_obj = res.state[0]
                params_b = res.params
                return params_b, y_obj
        return wrapper
    elif method == 'GD' or method == 'gradient_descent':
        def wrapper(params_b:Any,*args):
            def gd_step(params_b:Any):
                y_obj, grads = value_and_grad(f_obj)(params_b)
                inner_sgd_fn = lambda g, state: (state - lr * g)
                return jax.tree_multimap(inner_sgd_fn, grads, params_b), y_obj
            return gd_step(params_b)
        return wrapper
    elif method == 'Adam' or method == 'adam': # (Incomplete)
        def wrapper(params_b:Any,*args):
            optimizer = optax.adam(learning_rate=lr)
            def adam_step(params_b:Any):
                opt_state = optimizer.init(params_b)
                y_obj, g_params_b = value_and_grad(f_obj)(params_b)
                updates, opt_state = optimizer.update(g_params_b, opt_state)
                return optax.apply_updates(params_b, updates), y_obj
            return adam_step(params_b)
        return wrapper