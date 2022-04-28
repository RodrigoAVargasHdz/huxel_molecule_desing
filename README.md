# huxel_molecule_desing
JAX  + Huckel model + J. Chem. Phys. 129, 044106 􏰀2008􏰁 (2008)


# Test
exectute `main_opt_molec.py` where the options are,
1. `--l`, integer (for random number start)
2. `--s`, integer for smile data set (range [1,->,8])
3. `--obj`, objective to optimize options [homo_lumo,polarizability]
4. `--opt`, optimization method [adam,GD,BFGS]
5. `--extfield`, external field value (only for polarizability)