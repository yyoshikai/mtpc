import numpy as np
from .utils.logger import get_logger


def get_mask(y: np.ndarray, folds: np.ndarray, reg: bool) \
        -> tuple[np.ndarray|None]:
    logger = get_logger(__name__)

    test_mask = folds == 0
    if reg:
        val_fold = 1
    else:
        y_test = y[test_mask]
        if np.all(y_test == y_test[0]):
            logger.info(f"y_test only contains {y_test[0]}")
            return None, None, None
        for val_fold in range(1, 5):
            val_mask = folds == val_fold
            train_mask = (folds > 0)&(folds != val_fold)
            y_val = y[val_mask]
            y_train = y[train_mask]
            if not (np.all(y_train == y_train[0]) or np.all(y_val == y_val[0])):
                break
        else:
            logger.info(f"No val_fold could split property.")
            return None, None, None
    val_mask = folds == val_fold
    train_mask = (folds > 0)&(folds != val_fold)
    assert not np.any(train_mask&val_mask)
    assert not np.any(val_mask&test_mask)
    assert not np.any(test_mask&train_mask)
    return train_mask, val_mask, test_mask