import sys, os
from glob import glob
WORKDIR = os.environ.get('WORKDIR', "/workspace")

for path in sorted(glob(f"{WORKDIR}/mtpc/featurize/250715_main/**/feat_added.npy", recursive=True)):
    name = path.removeprefix(f"{WORKDIR}/mtpc/featurize/") \
            .removesuffix("/feat_added.npy")
    for split in ['n_ak_bin', 'n_ak_bin_noout']:
        if not os.path.exists(f"{WORKDIR}/mtpc/predict/ml/feature_linear/{name}/dyskeratosis/data_wsi/{split}/0/score.csv"):
            print(name)
            break