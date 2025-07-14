import sys, os, logging, pickle
from typing import Optional
from argparse import ArgumentParser
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, \
        mean_squared_error, mean_absolute_error, r2_score
import cuml

WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path += [f'{WORKDIR}', f'{WORKDIR}/mtpc']
from src.utils.logger import get_logger, add_stream_handler

def predict_patch(X_train: np.ndarray, X_test: np.ndarray, 
        y_train: np.ndarray, y_test: np.ndarray, 
        is_reg: bool, result_dir: str):


    logger = get_logger('predict_patch')
    if os.path.exists(f"{result_dir}/score.csv") \
            and os.path.exists(f"{result_dir}/model.pkl"):
        logger.info(f"all results of {result_dir} already exists.")
        return
    
    logger.info(f"predicting {result_dir}...")
    os.makedirs(result_dir, exist_ok=True)

    # check if X and y are finite
    nf_path = f"{result_dir}/nonfinite_params.txt"
    n_nf = 0
    with open(nf_path, 'w') as f:
        for param, name in zip([X_train, X_test, y_train, y_test], 
                ['X_train', 'X_test', 'y_train', 'y_test']):
            if np.any(~np.isfinite(param)):
                logger.warning(f"{name} contains nonfinite values.")
                f.write(param+'\n')
                n_nf += 1
    if n_nf > 0: 
        return
    else:
        os.remove(nf_path)

    if is_reg:
        model = cuml.LinearRegression(copy_X=True)
        model.fit(X_train.copy(), y_train.copy())
        y_pred_test = model.predict(X_test)
        res = spearmanr(y_pred_test, y_test)
        df_score = pd.DataFrame({'score': {
            'RMSE': mean_squared_error(y_test, y_pred_test)**0.5,
            'MAE': mean_absolute_error(y_test, y_pred_test),
            'R^2': r2_score(y_test, y_pred_test), 
            'rho': res.statistic,
            'p_rho': res.pvalue
        }})

    else:
        model = cuml.LogisticRegression()
        model.fit(X_train.copy(), y_train.copy())
        y_pred = model.predict(X_test)
        df_score = pd.DataFrame({'score': {
            'AUROC': roc_auc_score(y_test, y_pred),
            'AUPR': average_precision_score(y_test, y_pred)
        }})
    df_score.to_csv(f"{result_dir}/score.csv")
    with open(f"{result_dir}/model.pkl", 'wb') as f:
        pickle.dump(model, f)


dfp = pd.read_csv(f"{WORKDIR}/mtpc/data/target/patch.csv", index_col=0)
dfa = pd.read_csv(f"{WORKDIR}/mtpc/data/target/add_patch.csv", index_col=0)

default_units = ['wsi', 'cond', 'cond_wsi', 'comp_wsi', 'comp_cond'] # no 'patch', 'region'
unit2n_split = {unit: 2 if 'comp' in unit else 10 for unit in default_units}

tgt2str_tgts = \
    {tgt: [tgt, 'find_name'] for tgt in ['is_Normal', 'is_Mild', 'is_Moderate', 'is_Severe']} \
    |{'cls_ak': ['cls_ak']}\
    |{'n_ak': ['cls_ak', 'n_ak_bin', 'n_ak_bin_noout'], 
      'area_ak': ['cls_ak', 'area_ak_bin', 'area_ak_bin_noout']}\
    |{tgt: [f"{tgt}_bin"] for tgt in ['bio', 'acantholysis', 'dyskeratosis']}
default_tgts = list(tgt2str_tgts.keys())

reg_tgts = ['n_ak', 'area_ak', 'bio', 'acantholysis', 'dyskeratosis']
add_tgts = ['bio', 'acantholysis', 'dyskeratosis']

def predict_all(fname, tgts: list[str]=default_tgts, units: list[str]=default_units):

    # check args
    assert set(tgts) <= set(default_tgts)
    assert set(units) <= set(default_units)

    # Load feature
    X_main = np.load(f"{WORKDIR}/mtpc/featurize/{fname}/feat_all.npy").astype(np.float32)
    try:
        X_add = np.load(f"{WORKDIR}/mtpc/featurize/{fname}/feat_added.npy").astype(np.float32)
    except FileNotFoundError:
        print(f"{fname}/feat_added.npy not found and passed.")
        X_add = None

        
    # データで分割するもの (data_wsi)
    if X_add is not None:
        tgt = 'dyskeratosis'
        unit = 'data_wsi'
        for str_tgt in ['n_ak_bin_noout', 'n_ak_bin']:
            
            X_train = X_main
            X_test = X_add
            y_train = dfp['n_ak'].values
            y_test = dfa[tgt]

            folds = np.load(f"{WORKDIR}/mtpc/data/split/main/wsi/{str_tgt}/0.npy")
            fold_mask = folds >= 0
            X_train = X_train[fold_mask]
            y_train = y_train[fold_mask]

            mask_trainval = np.isfinite(y_train)
            mask_test = np.isfinite(y_test)
            X_train = X_train[mask_trainval]
            y_train = y_train[mask_trainval]
            X_test = X_test[mask_test]
            y_test = y_test[mask_test]

            result_dir = f"feature_linear/{fname}/{tgt}/{unit}/{str_tgt}/0" 
            predict_patch(X_train, X_test, y_train, y_test, is_reg=True, result_dir=result_dir)

    for tgt in tgts:

        is_add = tgt in add_tgts
        if is_add and X_add is None: continue
        if is_add:
            X = X_add
            y = dfa[tgt].values
        else:
            X = X_main
            y = dfp[tgt].values
        is_reg = tgt in reg_tgts
        is_cls = not is_reg

        for unit in units:
            for str_tgt in tgt2str_tgts[tgt]:
                for i_split in range(10):
                    fold_path = f"{WORKDIR}/mtpc/data/split/{'add' if is_add else 'main'}" \
                        f"/{unit}/{str_tgt}/{i_split}.npy"
                    
                    if not os.path.exists(fold_path): 
                        logger.warning(f"fold_path for {tgt=} {unit=}, {str_tgt=}, {i_split=} does not exist.")
                        continue
                    
                    folds = np.load(fold_path)
                    X_train = X[folds > 0]
                    X_test = X[folds == 0]
                    y_train = y[folds > 0]
                    y_test = y[folds == 0]
                    if is_cls and (np.all(y_test == y_test[0])
                            or np.all(y_train == y_train[0])):
                        logger.info(f"Only {y_test[0]} in {tgt=} {unit=}, {str_tgt=}, {i_split=}")
                        continue
                    assert all([np.all(np.isfinite(y0)) for y0 in [y_train, y_test]])

                    result_dir = f"feature_linear/{fname}/{tgt}/{unit}/{str_tgt}/{i_split}"        
                    predict_patch(X_train, X_test, y_train, y_test, is_reg, result_dir)


if __name__ == '__main__':
    logging.captureWarnings(True)
    logger = get_logger()
    add_stream_handler(logger)

    parser = ArgumentParser()
    parser.add_argument('--fnames', nargs='+')
    parser.add_argument('--tgts', nargs='*', default=default_tgts)
    parser.add_argument('--units', nargs='*', default=default_units)
    args = parser.parse_args()
    for fname in args.fnames:
        predict_all(fname, args.tgts, args.units)
