

def metrics(y_pred_test, y_test):
    rec = spearmanr(y_pred_test, y_test)
    return {
        'RMSE': mean_squared_error(y_test, y_pred_test)**0.5,
        'MAE': mean_absolute_error(y_test, y_pred_test),
        'R^2': r2_score(y_test, y_pred_test),
        'rho': rec.statistic,
        'p_rho': rec.pvalue
    }

def predict_patch(split, fname, model):
    
    logger = get_logger('predict_patch')
    rdir = f"results/feature_ml/{model}/{fname}/{split}"

    # check all exists
    if os.path.exists(f"{rdir}/score.csv"):
        logger.info(f"All results exist for {fname=}, {split=}")
        return
    import optuna, xgboost as xgb
    import cuml

    # load folds
    folds = np.load(f"/workspace/mtpc/data/split/main/{split}.npy")
    X_train_val = np.load(f"/workspace/mtpc/featurize/{fname}/feat_all.npy")
    X_test = np.load(f"/workspace/mtpc/featurize/{fname}/feat_added.npy")
    X_test = X_test[dfa_mask]
    X = np.concatenate([X_train_val, X_test])

    train_mask = np.concatenate([folds > 0, np.full(len(X_test), False)]).astype(bool)
    val_mask = np.concatenate([folds == 0, np.full(len(X_test), False)]).astype(bool)
    test_mask = np.concatenate([np.full(len(folds), False), 
            np.full(len(X_test), True)]).astype(bool)
    train_val_mask = ~test_mask
    
    X_train = X[train_mask]
    X = (X - np.mean(X_train, axis=0)) / np.std(X_train)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    X_train_val, y_train_val = X[train_val_mask], y[train_val_mask]

    for y0 in [y_train, y_val, y_test]:
        assert np.all(np.isfinite(y0))

    logger.info(f"predicting {rdir} ...")

    os.makedirs(f"{rdir}", exist_ok=True)        
    match model:
        case 'Linear':
            model = cuml.LinearRegression(copy_X=True)
            model.fit(X_train_val.copy(), y_train_val.copy())
            y_pred = model.predict(X)
            y_pred_test = y_pred[test_mask]
        case 'LinearSVR':
            def objective(trial: optuna.Trial):
                model0 = cuml.LinearSVR(C=trial.suggest_float('C', 1e-4, 1e4, log=True), 
                    loss=trial.suggest_categorical('loss', 
                    ['epsilon_insensitive', 'squared_epsilon_insensitive']))
                model0.fit(X_train.copy(), y_train.copy())
                score = mean_squared_error(y_val, model0.predict(X_val.copy()))
                return score
            
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=100)
            model = cuml.LinearSVR(**study.best_params)
            model.fit(X_train_val.copy(), y_train_val.copy())
            y_pred = model.predict(X.copy())
            y_pred_test = y_pred[test_mask]
            study.trials_dataframe().to_csv(f"{rdir}/optuna.csv")
        
        case 'XGB':
            def objective(trial: optuna.Trial):
                model = xgb.XGBRegressor(
                    eta = trial.suggest_float('eta', 1.0e-8, 10.0, log=True),
                    gamma = trial.suggest_float('gamma', 1.0e-8, 1.0, log=True),
                    max_depth = trial.suggest_int('max_depth', 3, 15),
                    max_delta_step = trial.suggest_int('max_delta_step', 1, 10),
                    subsample = trial.suggest_float('subsample', 0.6, 1.0),
                    reg_lambda = trial.suggest_float('reg_lambda', 1.0e-8, 0.1, log=True),
                    alpha = trial.suggest_float('alpha', 1.0e-8, 1.0, log=True)
                )
                model.fit(X_train, y_train)
                return mean_squared_error(y_val, model.predict(X_val))
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=1)
            model = xgb.XGBRegressor(**study.best_params)
            model.fit(X_train_val, y_train_val)
            y_pred = model.predict(X)
            y_pred_test = y_pred[test_mask]
            study.trials_dataframe().to_csv(f"{rdir}/optuna.csv")

        case _:
            raise ValueError

    with open(f"{rdir}/model.pkl", 'wb') as f:
        pickle.dump(model, f)
    pd.DataFrame({'true': y, 'pred': y_pred}).to_csv(f"{rdir}/pred.csv", index=False)
    pd.DataFrame({'score': metrics(y_pred_test, y_test)}) \
            .to_csv(f"{rdir}/score.csv")

if __name__ == '__main__':

    import sys, os, logging
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--fname', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--model', default='Linear')
    args = parser.parse_args()
    
    rdir = f"results/feature_ml/{args.model}/{args.fname}/{args.split}"
    if os.path.exists(f"{rdir}/score.csv"):
        print(f"All results exist for fname={args.fname}, split={args.split}")
        sys.exit()

    import pickle
    import numpy as np, pandas as pd
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from scipy.stats import spearmanr
    sys.path += ['/workspace', '/workspace/mtpc']
    from src.utils.logger import get_logger, add_stream_handler

    logging.captureWarnings(True)
    logger = get_logger()
    add_stream_handler(logger)

    df = pd.read_csv("/workspace/mtpc/data/target/patch.csv", index_col=0)
    dfa = pd.read_csv("/workspace/mtpc/data/target/add_patch.csv", index_col=0)
    dfa_mask = np.isfinite(dfa['dyskeratosis'])
    dfa = dfa[dfa_mask]
    y_train_val = df['n_ak']
    y_test = dfa['dyskeratosis']
    y = np.concatenate([y_train_val, y_test])

    predict_patch(**vars(args))
