import sys, os, logging
from argparse import ArgumentParser
import numpy as np, pandas as pd

WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path += [f'{WORKDIR}', f'{WORKDIR}/mtpc']
from src.utils.logger import get_logger, add_stream_handler
from predict.ml.linear import predict_patch


dfp = pd.read_csv(f"{WORKDIR}/mtpc/data/target/patch.csv", index_col=0)
dfa = pd.read_csv(f"{WORKDIR}/mtpc/data/target/add_patch.csv", index_col=0)

def predict_all(fname, seed, splits):
    
    # Load feature
    X_main = np.load(f"{WORKDIR}/mtpc/featurize/{fname}/feat_all.npy").astype(np.float32)
    try:
        X_add = np.load(f"{WORKDIR}/mtpc/featurize/{fname}/feat_added.npy").astype(np.float32)
    except FileNotFoundError:
        print(f"{fname}/feat_added.npy not found and passed.")
        return
        
    tgt = 'dyskeratosis'
    unit = 'data_wsi'
    for split in splits:
        
        X_train = X_main
        X_test = X_add
        y_train = dfp['n_ak'].values
        y_test = dfa[tgt]

        folds = np.load(f"{WORKDIR}/mtpc/data/split/main/wsi/{split}.npy")
        fold_mask = folds >= 0
        X_train = X_train[fold_mask]
        y_train = y_train[fold_mask]

        mask_trainval = np.isfinite(y_train)
        mask_test = np.isfinite(y_test)
        X_train = X_train[mask_trainval]
        y_train = y_train[mask_trainval]
        X_test = X_test[mask_test]
        y_test = y_test[mask_test]

        result_dir = f"feature_linear/{fname}/{tgt}/{unit}/{split}" 
        predict_patch(X_train, X_test, y_train, y_test, is_reg=True, result_dir=result_dir, seed=seed)

if __name__ == '__main__':
    logging.captureWarnings(True)
    logger = get_logger()
    add_stream_handler(logger)

    parser = ArgumentParser()
    parser.add_argument('--fnames', nargs='+')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--splits', nargs='+', default=['n_ak_bin_noout/0', 'n_ak_bin/0'])
    args = parser.parse_args()
    for fname in args.fnames:
        predict_all(fname, args.seed, args.splits)
