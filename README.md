# Molecule inverse design with automatic differentiation: Hückel + JAX

Using JAX and the Hückel model we optimize the type of atoms given an adjacency matrix of a molecular graph and the target observable.


Otpimization of HOMO-LUMO gap ($\epsilon_{HL} = \text{LUMO} - \text{HOMO}$), and polarizability ($\langle \alpha\rangle$),
<p align="center">
<img align="middle" src="./assets/homo_lumo.gif" alt="HOMO_LUMO Demo" width="270" height="250" />
<img align="middle" src="./assets/polarizability.gif" alt="HOMO_LUMO Demo" width="250" height="270" />
</p>


# Test
exectute `main_opt_molec.py` where the options are,
1. `--l`, integer (for random number start `jax.random.PRNGKey(l)`)
2. `--s`, integer for smile data set (range [1,->,8])
3. `--obj`, objective to optimize options [homo_lumo,polarizability]
4. `--opt`, optimization method [adam,GD,BFGS]
5. `--extfield`, external field value (only for polarizability)


## Requirments
```
jax, optax, jaxopt
```