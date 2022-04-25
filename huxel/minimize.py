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


def opt_obj(f_obj:Callable,params_b_init:Any,params_fixed_atoms:Any,params_extra:Any,opt_method:str='BGFS',ntr:int=5,lr:float=1E-3):
    opt_step = wrapper_opt_method(f_obj,opt_method,lr)
    params_b = params_b_init
    
    r = {}
    params_b_opt = params_b_init.copy()
    y_obj_opt = jnp.inf
    for itr in range(0,ntr):
        params_b, y_obj = opt_step(params_b)
        opt_molecule, params_opt_one_hot = get_molecule(
            {**params_b,**params_fixed_atoms}, params_extra["one_pi_elec"]
        )
        y_obj = f_obj(params_b)
        y_obj_one_hot = f_obj(params_opt_one_hot)
        print(itr, y_obj, opt_molecule, y_obj_one_hot)  # opt_res,
        print(params_b)

    print('-------------------------------------------------')
    return None


def wrapper_opt_method(f_obj:Callable,method:str='BGFS',lr:float=1E-3):
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

    # def opt_adam(params_b: Any, ntr: int, lr: 0.1):
    #     v_and_g_obj = value_and_grad(f_obj)

    #     optimizer = optax.adam(learning_rate=lr)
    #     opt_state = optimizer.init(params_b)

    #     params_b_opt0 = params_b.copy()
    #     hl_gap0 = f_obj(params_b)
    #     opt_molec0, _ = get_molecule(params_b, params_extra["one_pi_elec"])

    #     print(f"0, {hl_gap0}, {opt_molec0}")
    #     for itr in range(ntr):
    #         hl_gap, g_params_b = v_and_g_obj(params_b)
    #         updates, opt_state = optimizer.update(g_params_b, opt_state)
    #         params_b = optax.apply_updates(params_b, updates)
    #         opt_molecule = get_molecule(params_b, params_extra["one_pi_elec"])

    #         if hl_gap < hl_gap0:
    #             params_b_opt0 = params_b.copy()
    #             hl_gap0 = hl_gap
    #         if opt_molecule != opt_molec0:
    #             opt_molec0 = opt_molecule
    #             print(f"{itr}, {hl_gap}, {opt_molecule} *")
    #         elif itr % 25 == 0:
    #             print(f"{itr}, {hl_gap}, {opt_molecule}")

    #     return params_b_opt0, get_molecule(params_b_opt0, params_extra["one_pi_elec"])


'''
def opt_BFGS(f_obj:Callable,params_b_init:Any,params_fixed_atoms:Any,params_extra:Any,ntr:int=20):
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
    for i in range(ntr):
        opt_res = opt.run(params_b)
        params_b = opt_res[0]
        opt_molecule, params_opt_one_hot = get_molecule(
            {**params_b,**params_fixed_atoms}, params_extra["one_pi_elec"]
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
    {**params_b_opt,**params_fixed_atoms}    
    opt_molecule, _ = get_molecule({**params_b_opt,**params_fixed_atoms}    , params_extra["one_pi_elec"])
    y_ev = f_obj(params_b_opt)
    return params_b_opt, opt_molecule, r
'''