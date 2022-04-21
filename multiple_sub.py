import os
import time
import numpy as np

w_decay_ = [1e-3, 1e-4, 1e-5, 5e-4, 1e-6, 1e-7]
rho_G_ = [0.0]  # w_decay_


def sh_file(N, l, beta_, l2reg):

    #     BE SURE TO NAME FOR EACH CALCULATION AN INDIVIDUAL XXX.SH FILE CAUSE THAT IS THE ONE
    #     YOU WILL BE SUBMITTING
    l2reg_ = l2reg.replace(" ", "_")

    f_tail = "hxl_n_{}_l_{}_{}_{}_AdamW".format(N, l, beta_, l2reg_)

    f = open("JC_%s.sh" % (f_tail), "w+")
    f.write("#!/bin/bash \n")
    f.write("#SBATCH --nodes=1 \n")
    f.write("#SBATCH --ntasks=1 \n")
    f.write("#SBATCH --cpus-per-task=12  # Cores proportional to GPU \n")
    f.write("#SBATCH --mem=128G \n")  # 64
    f.write("#SBATCH --qos nopreemption  \n")
    f.write("#SBATCH --partition=cpu \n")
    f.write("#SBATCH --job-name={} \n".format(f_tail))
    f.write("#SBATCH --time=3:00:00 \n")
    f.write("#SBATCH --output=out_{}.log \n".format(f_tail))

    f.write("\n")
    #     LOAD MODULES
    f.write("source activate $HOME/huckel-jax \n")

    # f.write(
    #     "/h/rvargas/.conda/envs/huckel-jax/bin/python main.py --N {} --lr 2E-2 --l {} --batch_size 128 --job opt  --beta {} -Wdecay {} \n".format(
    #         N, l, beta_, l2reg
    #     )  # h_x h_xy y_xy
    # )

    f.write(
        "/h/rvargas/.conda/envs/huckel-jax/bin/python main.py --N {} --lr 2E-2 --l {} --job pred  --beta {}  -Wdecay {} \n".format(
            N, l, beta_, l2reg
        )
    )

    f.write("\n")

    f.write("\n")
    f.close()

    if os.path.isfile("JC_%s.sh" % (f_tail)):
        print("Submitting JC_%s.sh" % (f_tail))
        os.system("sbatch JC_%s.sh " % (f_tail))


def main():
    # beta_ = "exp_freezeR"
    # print("caca")
    # sh_file(5, 0, beta_)
    # assert 0

    beta_ = ["c", "exp", "linear"]
    l2reg_ = ["h_x h_xy", "h_x h_xy y_xy", " "]
    n_ = [25, 10, 5]
    for n in n_[:1]:
        for l in range(5):
            for b in beta_:
                for l2reg in l2reg_:
                    sh_file(n, l, b, l2reg)
                    assert 0


if __name__ == "__main__":
    main()
