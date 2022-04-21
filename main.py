import time
import argparse

# from huxel.optimization import _optimization as _opt
from huxel.optimization_inversemol import _optimization as _opt
from huxel.optimization_inversemol import _all_molecules
from huxel.prediction import _pred, _pred_def


def main():
    parser = argparse.ArgumentParser(description="opt overlap NN")
    parser.add_argument("--N", type=int, default=10, help="traning data")
    parser.add_argument("--l", type=int, default=0, help="label")
    parser.add_argument("--lr", type=float, default=2e-2, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="batches")
    parser.add_argument("--job", type=str, default="opt", help="job type")
    parser.add_argument("--beta", type=str, default="c", help="beta function type")

    # bathch_size = #1024#768#512#256#128#64#32
    args = parser.parse_args()
    l = args.l
    n_tr = args.N
    lr = args.lr
    batch_size = args.batch_size
    job_ = args.job
    beta_ = args.beta

    if job_ == "opt":
        _opt(n_tr, batch_size, lr, l, beta_)
    elif job_ == "pred":
        _pred(n_tr, l, beta_)
    elif job_ == "pred_def":
        _pred_def(beta_)


if __name__ == "__main__":
    main()
