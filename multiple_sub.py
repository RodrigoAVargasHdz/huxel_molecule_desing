import os
import time
# import numpy as np

def sh_file(smilei,l,opt):
    f_tail = f'smile{smilei}_l_{l}_{opt}'

    f=open('Hxl_%s.sh'%(f_tail), 'w+')
    f.write('#!/bin/bash \n')
    f.write('#SBATCH --ntasks=1 \n')
#    f.write('#SBATCH --cpus-per-task=12  # Cores proportional to GPU \n')
#    f.write('#SBATCH --mem=64G \n')#64
    f.write('#SBATCH --account=rrg-aspuru\n')
#    f.write('#SBATCH --qos nopreemption  \n')
#    f.write('#SBATCH --partition=cpu \n')
    f.write('#SBATCH --job-name={} \n'.format(f_tail))
    f.write('#SBATCH --time=0-25:00 \n')
    f.write('#SBATCH --output=out_{}.log \n'.format(f_tail))

    f.write('\n')
#     LOAD MODULES
    f.write('module load python/3.9.8 \n')
    f.write('source $HOME/.virtualenvs/jaxenv/bin/activate\n')
    f.write('module load python/3.9.8 \n')


    f.write('python main.py --N {} --lr 2E-2 --l {} --batch_size 128 --job opt --beta {} \n'.format(smilei,l,opt))

    f.write('\n')

    f.write('\n')
    f.close()

    if os.path.isfile('JC_%s.sh'%(f_tail)):
        print('Submitting JC_%s.sh'%(f_tail))
        os.system('sbatch JC_%s.sh '%(f_tail))

def main():
#    beta_ = 'exp_freezeR'
#    print('caca')
#    sh_file(5,0,beta_)
#    assert 0

    opt_ = ['homo_lumo','polarizability']

    n_ = [50,25,10,5]
    for si in range(1,10):
        for l in range(1,150):
            sh_file(si,l,opt_[0])
            sh_file(si,l,opt_[1])
    #         # assert 0


if __name__== "__main__":
    main()