import os
import argparse
import numpy as onp
import jax.numpy as jnp

from huxel.molecule import myMolecule
from huxel.optimization_inversemol import _optimization_molec as _opt

def get_jcp_molecule_data(smile_i:int=2):# Benzene --> smile_i = 2
    data_d = os.path.join(os.getcwd(),'molecules')
    d = onp.load(os.path.join(data_d,f"smile{smile_i}.npy"),allow_pickle=True)

    smile = d.item()['smile']
    atom_types = d.item()['atoms']
    conectivity_matrix = jnp.array(d.item()['AdjacencyMatrix'],dtype=int)
    xyz = jnp.array(d.item()['xyz'])
    return smile, atom_types, conectivity_matrix, xyz

def main():
    parser = argparse.ArgumentParser(description="molecular inverse design Huckel with JAX")
    parser.add_argument("--s", type=int, default=1, help="smile integer for JCP2008 molecules, range [1 to 8]")
    parser.add_argument("--l", type=int, default=0, help="label")
    # parser.add_argument("--lr", type=float, default=2e-2, help="learning rate")
    parser.add_argument("--obj", type=str, default="homo_lumo", help="objective type")
    parser.add_argument("--opt", type=str, default="BFGS", help="objective type")
    parser.add_argument("--extfield", type=float, default=0.01, help="external field for polarization")

    args = parser.parse_args()
    smile_i = args.s
    l = args.l
    # lr = args.lr
    obj = args.obj
    opt = args.opt
    ext_field = args.extfield

    # read molecule info
    smile, atom_types, conectivity_matrix, xyz = get_jcp_molecule_data(smile_i)

    print(smile)
    print(atom_types)
    print(conectivity_matrix)
    print(xyz)

    molec = myMolecule(
        smile_i,
        smile,
        atom_types,
        conectivity_matrix,
        xyz
    )
    print(molec)

    if obj == 'homo_lumo':
        _opt(l, molec,obj,opt)
    elif obj == 'polarizability':
        _opt(l, molec,obj,opt,ext_field)

def main_best():
    # ---------------------------
    # smile1, polarizability, l = 165, y =  1178.337264402567, y_true = 1142.8244196383505
    # smile2, polarizability, l = 39, y =  5.675181519139846, y_true = 5.733011286033663
    # smile3, polarizability, l = 6, y =  30.354181709253716, y_true = 30.471830634333408
    # smile4, polarizability, l = 2, y =  20.19906672810775, y_true = 20.207249425380894
    # smile5, polarizability, l = 40, y =  33.99886063215378, y_true = 31.575829933212795
    # smile6, polarizability, l = 33, y =  42.206361731023236, y_true = 42.26891783553312
    # smile7, polarizability, l = 13, y =  89.89528632533394, y_true = 90.24777285217074
    # smile8, polarizability, l = 238, y =  114.66863327676617, y_true = 107.55964007113984
    # ---------------------------
    # smile1, homo_lumo, l = 20, y =  1.6903396192658715, y_true = 1.6643651033729956
    # smile2, homo_lumo, l = 14, y =  2.5610107499631285, y_true = 2.557615114374589
    # smile3, homo_lumo, l = 135, y =  2.236733190814573, y_true = 2.2355677083042513
    # smile4, homo_lumo, l = 1, y =  2.324648905522813, y_true = 2.322909977302947
    # smile5, homo_lumo, l = 12, y =  1.8190387041596907, y_true = 1.8110723048852058
    # smile6, homo_lumo, l = 15, y =  2.2456615818267043, y_true = 2.236468403891718
    # smile7, homo_lumo, l = 8, y =  2.0543190760417067, y_true = 2.0542667642891743
    # smile8, homo_lumo, l = 5, y =  1.672370007208824, y_true = 1.6695586686152977

    polarizability_best = [165,39,6,2,40,33,13,238]
    for si,l in enumerate(polarizability_best):
        si += 1
        smile, atom_types, conectivity_matrix, xyz = get_jcp_molecule_data(si)
        molec = myMolecule(
            si,
            smile,
            atom_types,
            conectivity_matrix,
            xyz
        )
        _opt(l, molec,'polarizability',"BFGS",0.01)

    homo_lumo_best = [20,14,135,1,12,15,8,5]
    for si,l in enumerate(homo_lumo_best):
        si += 1
        smile, atom_types, conectivity_matrix, xyz = get_jcp_molecule_data(si)
        molec = myMolecule(
            si,
            smile,
            atom_types,
            conectivity_matrix,
            xyz
        )
        _opt(l, molec,'homo_lumo',"BFGS")



if __name__ == "__main__":
    # main()
    main_best()
