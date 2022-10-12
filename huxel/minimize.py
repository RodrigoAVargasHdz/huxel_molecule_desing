import os
import numpy as onp
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jax import value_and_grad

import jaxopt
from jaxopt import ScipyMinimize
import optax

from huxel.utils import get_molecule


def opt_obj(

        f_obj: Callable,
        params_b_init: Any,
        params_fixed_atoms: Any,
        params_extra: Any,
        files: dict,
        _minimizer: str = 'BFGS',
        ntr: int = 30,
        lr: float = 2E-1,

) -> Tuple:
    """Optimization

    Args:
        f_obj (Callable): function to compute target observable
        params_b_init (Any): initial parameters b 
        params_fixed_atoms (Any): fixed atoms 
        params_extra (Any): additional parameters 
        files (dict): dictionary with files names
        _minimizer (str, optional): _description_. Defaults to 'BFGS'.
        ntr (int, optional): epochs. Defaults to 30.
        lr (float, optional): learning rate. Defaults to 2E-1. (only applicable for Adam and GD methods)

    Returns:
        _type_: optimized parameters b, molecule, dictionary with the results
    """

    rwd = files['rwd']
    file_r = files['results']
    file_out = os.path.join(rwd, files['out'])
    files_r = os.path.join(rwd, file_r)

    f = open(file_out, 'a+')
    print("------------------------------------------------", file=f)
    print(f"Optimization with {_minimizer}", file=f)
    f.close()

    opt_step = wrapper_opt_method(f_obj, _minimizer, lr)
    params_b = params_b_init

    params_b, (y_obj, grad_y_obj) = opt_step(params_b)

    molecule_itr, params_b_one_hot = get_molecule(
        {**params_b, **params_fixed_atoms}, params_extra["one_pi_elec"]
    )

    y_obj_one_hot = f_obj(params_b_one_hot)

    r = {0: {
        "molecule": molecule_itr,
        "params_b": params_b,
        "gradient": grad_y_obj,
        "objective": y_obj,
        "objective_one_hot": y_obj_one_hot,
    }}

    f = open(file_out, 'a+')
    print(f"0 | {y_obj}   {y_obj_one_hot} | {molecule_itr}", file=f)
    f.close()

    params_b_opt = params_b_init.copy()
    y_obj_opt = jnp.inf
    molecule_opt = []
    for itr in range(1, ntr+1):
        params_b, (y_obj, grad_y_obj) = opt_step(params_b)
        molecule_itr, params_b_one_hot = get_molecule(
            {**params_b, **params_fixed_atoms}, params_extra["one_pi_elec"]
        )
        y_obj = f_obj(params_b)
        y_obj_one_hot = f_obj(params_b_one_hot)

        r.update(
            {
                itr: {
                    "molecule": molecule_itr,
                    "params_b": params_b,
                    "gradient": grad_y_obj,
                    "objective": y_obj,
                    "objective_one_hot": y_obj_one_hot,
                }
            }
        )

        f = open(file_out, 'a+')
        print(f"{itr} | {y_obj}   {y_obj_one_hot} | {molecule_itr}", file=f)
        f.close()

        if y_obj < y_obj_opt:
            y_obj_opt = y_obj
            params_b_opt = params_b
            molecule_opt = molecule_itr

    f = open(file_out, 'a+')
    print("------------------------------------------------", file=f)
    f.close()
    return params_b_opt, molecule_opt, r


def wrapper_opt_method(f_obj: Callable, method: str = 'BFGS', lr: float = 2E-1) -> callable:
    """wrapper for optimization

    Args:
        f_obj (Callable): function to compute target observable
        method (str, optional): optimization method. Defaults to 'BFGS'.
        lr (float, optional): learning rate. Defaults to 2E-1. (only applicable for Adam and GD methods)

    Returns:
        callable: optimization 
    """

    if method == 'BFGS' or method == 'bfgs':
        def wrapper(params_b: Any, *args):
            optimizer = ScipyMinimize(
                method="BFGS",
                fun=f_obj,
                jit=False,
                options={"maxiter": 1},
            )
            res = optimizer.run(params_b)
            grad_y_obj = jax.grad(f_obj)(params_b)
            y_obj = res.state[0]
            params_b = res.params
            return params_b, (y_obj, grad_y_obj)
        return wrapper

    elif method == 'GD' or method == 'SG' or method == 'gradient_descent':
        def wrapper(params_b: Any, *args):
            def gd_step(params_b: Any):
                y_obj, grad_y_obj = value_and_grad(f_obj)(params_b)
                def inner_sgd_fn(g, state): return (state - lr * g)
                return jax.tree_multimap(inner_sgd_fn, grad_y_obj, params_b), (y_obj, grad_y_obj)
            return gd_step(params_b)
        return wrapper

    elif method == 'Adam' or method == 'adam':
        def wrapper(params_b: Any, *args):
            optimizer = optax.adam(learning_rate=lr)

            def adam_step(params_b: Any):
                opt_state = optimizer.init(params_b)
                y_obj, grad_y_obj = value_and_grad(f_obj)(params_b)
                updates, opt_state = optimizer.update(grad_y_obj, opt_state)
                return optax.apply_updates(params_b, updates), (y_obj, grad_y_obj)
            return adam_step(params_b)
        return wrapper


'''
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
'''
