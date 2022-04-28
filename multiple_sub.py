import os
import time
# import numpy as np

def sh_file(smilei, l, obj, opt = 'BFGS'):
    f_tail = f'smile{smilei}_l_{l}_{obj}_{opt}'
    file_sh = f"Hxl_{f_tail}.sh"
    f=open(file_sh, 'w+')
    f.write('#!/bin/bash \n')
    f.write('#SBATCH --ntasks=1 \n')
#    f.write('#SBATCH --cpus-per-task=12  # Cores proportional to GPU \n')
#    f.write('#SBATCH --mem=64G \n')#64
    f.write('#SBATCH --account=rrg-aspuru\n')
#    f.write('#SBATCH --qos nopreemption  \n')
#    f.write('#SBATCH --partition=cpu \n')
    f.write('#SBATCH --job-name={} \n'.format(f_tail))
    f.write('#SBATCH --time=00:55:00 \n')
    f.write('#SBATCH --output=out_{}.log \n'.format(f_tail))

    f.write('\n')
#     LOAD MODULES
    f.write('module load python/3.9.8 \n')
    f.write('source $HOME/.virtualenvs/jaxenv/bin/activate\n')
    f.write('module load python/3.9.8 \n')
    f.write('\n')

    if obj == 'polarizability':
        ext_field = 0.01 
        f.write(f"python main.py --s {smilei} --l {l}  --obj {obj} --opt {opt} --extfield {ext_field}  \n")
    else:
        f.write(f"python main.py --s {smilei} --l {l}  --obj {obj} --opt {opt} \n")


    f.write('\n')

    f.write('\n')
    f.close()

    if os.path.isfile(file_sh):
        print(f"Submitting {file_sh}")
        os.system(f"sbatch {file_sh}")

def main():
#    beta_ = 'exp_freezeR'
#    print('caca')
#    sh_file(5,0,beta_)
#    assert 0

    obj_ = ['homo_lumo','polarizability']
    opt_ = 'BFGS'

    n_ = [50,25,10,5]
    for si in range(3,10):
        for l in range(1,150):
            sh_file(si,l,obj_[0],opt_)
            sh_file(si,l,obj_[1],opt_)
            assert 0


if __name__== "__main__":
    main()