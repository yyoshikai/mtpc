import numpy as np, pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class TGGATEDataset(Dataset[Image.Image]):
    def __init__(self, pdir):
        """
        各WSIから1つずつサンプリングするデータセット。
        一般的なものにする予定。
        - int_to_floatはデフォルトでtrueになっている。
        
        Parameters
        ----------
        pdir: preprocessのパス。
        ppi: 1WSIあたりのパッチ数。
        split: Input to self.calc_split
        """

        self.pdir = pdir
        dfcase = pd.read_csv(f"{pdir}/cases2.csv", index_col=0, keep_default_na=False)
        self.cids = dfcase.index.values
        self.ppi = 512

    def __getitem__(self, idx: int):
        cidx, patch_idx = divmod(idx, self.ppi)
        bytesize = np.dtype('uint8').itemsize
        cid = self.cids[cidx]
        patch = np.fromfile(f"{self.pdir}/sample_patch_agg/{cid}.npy", 
            count=256*256*3, offset=patch_idx*256*256*3*bytesize, 
            dtype=np.uint8).reshape(256, 256, 3)
        return Image.fromarray(patch).convert('RGB')

    def __len__(self):
        return len(self.cids)*self.ppi

data = TGGATEDataset("/workspace/patho/preprocess/results/tggate_liver_late")