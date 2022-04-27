import os
import numpy as onp


def read_molecule(smile_label:int):
    cwd = os.getcwd()
    
    datad = os.path.join(cwd,'molecules')
    print(datad)
    file_xyz = f"smiles{smile_label}_woH_wX.txt"
    file_am = f"smile{smile_label}_AdjacencyMatrix.npy"
    print(os.path.join(datad,file_xyz))
    r = {}
    if os.path.isfile(os.path.join(datad,file_xyz)) and os.path.isfile(os.path.join(datad,file_am)):
        D_xyz = onp.loadtxt(os.path.join(datad,file_xyz),dtype=str)
        D = onp.load(os.path.join(datad,file_am),allow_pickle=True)
        # print(D)
        # print(D_xyz)
        # atoms = D[:,0]
        # xyz = D[:,1:]
        r = {"smile": D.item()[f"smile{smile_label}"],
            "atoms": list(D_xyz[:,0]),
            "AdjacencyMatrix": D.item()["AdjacencyMatrix"],
            "xyz": onp.asarray(D_xyz[:,1:],dtype=float)
        }
        onp.save
        onp.save(os.path.join(datad,f"smile{smile_label}.npy"),r,allow_pickle=True)


def main():
    for i in range(1,9):
        read_molecule(i)

if __name__ == "__main__":
    main()


'''
def main():
    parser = argparse.ArgumentParser(description="opt overlap NN")
    parser.add_argument("--l", type=int, default=0, help="label")
    # parser.add_argument("--lr", type=float, default=2e-2, help="learning rate")
    parser.add_argument("--obj", type=str, default="homo_lumo", help="objective type")
    parser.add_argument("--opt", type=str, default="BFGS", help="objective type")
    parser.add_argument("--extfield", type=float, default=0.01, help="external field for polarization")

    args = parser.parse_args()
    l = args.l
    # lr = args.lr
    obj = args.obj
    opt = args.opt
    ext_field = args.extfield

    atom_types = ["X", "C", "C", "C", "X", "C"]
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
    # Kjell
    xyz = jnp.array([[ 1.40000000e+00,  3.70074342e-17,  0.00000000e+00],
       [ 7.00000000e-01, -1.21243557e+00,  0.00000000e+00],
       [-7.00000000e-01, -1.21243557e+00,  0.00000000e+00],
       [-1.40000000e+00,  2.08457986e-16,  0.00000000e+00],
       [-7.00000000e-01,  1.21243557e+00,  0.00000000e+00],
       [ 7.00000000e-01,  1.21243557e+00,  0.00000000e+00]])

    homo_lumo_grap_ref = -7.01 - (-0.42)

    molec = myMolecule(
        "benzene",
        smile,
        atom_types,
        conectivity_matrix,
        homo_lumo_grap_ref,
        jnp.ones((6, 6)),
        xyz
    )

    if obj == 'homo_lumo':
        _opt(l, molec,obj,opt)
    elif obj == 'polarizability':
        _opt(l, molec,obj,opt,ext_field)

'''
