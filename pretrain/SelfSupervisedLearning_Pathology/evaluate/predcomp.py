# -*- coding: utf-8 -*-
"""
# Leave One WSI Cross Validation in one dataset
# Clustering Score in one dataset

@author: Katsuhisa MORITA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE, MDS
from sklearn.linear_model import LogisticRegression

from evaluate import utils

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams["font.size"] = 14

# Settings
root = "/workspace/230727_pharm"
random_state=24771
lst_compounds_eisai=[
    "Corn Oil",
    "Bromobenzene",
    "CCl4",
    "Naphthyl isothiocyanate",
    "Methylcellulose",
    "Acetaminophen"
]
lst_compounds_eisai_conv=[
    "vehicle",
    "bromobenzene",
    "carbon tetrachloride",
    "naphthyl isothiocyanate",
    "acetaminophen",
]
lst_compounds_shionogi=[
    "vehicle",
    "voriconazole",
    "thioacetamide",
    "phenylhydrazine",
    "streptozotocin",
]
lst_compounds_rat=[
    "vehicle",
    "thioacetamide",
    "carbon tetrachloride",
    "naphthyl isothiocyanate",
    "acetaminophen",
]
lst_compounds_mouse=[
    "vehicle",
    "thioacetamide",
    "carbon tetrachloride",
    "naphthyl isothiocyanate",
    "acetaminophen",
    "galactosamine",
    "methylene dianiline"
]
lst_compounds_target=[
    "vehicle",
    "bromobenzene",
    "thioacetamide",
    "carbon tetrachloride",
    "naphthyl isothiocyanate",
    "acetaminophen",
]

class LOWCV:
    def __init__(self):
        self.df_info=pd.DataFrame()
        self.lst_arr_x=[]
        self.y=None
        self.le=None
        self.coef=1

    def evaluate(
        self,
        folder="",
        name="",
        pretrained=False,
        layer=4, size=None,
        n_model=5,
        convertz=True,
        compression=False, n_components=16,
        params_lr=dict(),
        plot_heat=False,
        eisai_dataset=False, shionogi_dataset=False,
        tggate_dataset=False, 
        rat_dataset=False, mouse_dataset=False,
        time="24 hr",
        ):
        # data load
        ## features array
        self.lst_arr_x, _ = utils.load_array_preprocess(
            layer=layer, folder=folder, name=name, 
            size=size, pretrained=pretrained, n_model=n_model,
            convertz=convertz,
            compression=compression, n_components=n_components, 
            meta_viz=False, concat=False,
        )
        ## information dataframe
        if size:
            self.coef=int(2000/size)
        else:
            self.coef=10 # already compressed by size=200
        if eisai_dataset:
            self.lst_compounds=lst_compounds_eisai_conv
        elif shionogi_dataset:
            self.lst_compounds=lst_compounds_shionogi
        elif tggate_dataset:
            self.lst_compounds=lst_compounds_target
        elif rat_dataset:
            self.lst_compounds=lst_compounds_rat
        elif mouse_dataset:
            self.lst_compounds=lst_compounds_mouse
        self.df_info=utils.LoadInfo().load(
            coef=self.coef, time=time, conv_name=True,
            lst_compounds=self.lst_compounds,
            eisai_dataset=eisai_dataset, shionogi_dataset=shionogi_dataset, 
            tggate_dataset=tggate_dataset, 
            rat_dataset=rat_dataset, mouse_dataset=mouse_dataset,
            conc="High"
            )
        
        ## Set label encoder and y
        self._set_label()
        # Evaluation
        lst_lst_stats=[]
        for i, arr_x in enumerate(self.lst_arr_x):
            y_pred = self._predict(arr_x[self.df_info["INDEX"].tolist(),:], self.y, params_lr, coef=self.coef)
            lst_stats = utils.calc_stats(self.y, y_pred, self.lst_compounds, self.le)
            lst_lst_stats.append(lst_stats)
            if plot_heat:
                if i==0:
                    y_preds=y_pred/n_model
                else:
                    y_preds+=y_pred/n_model
        if plot_heat:
            self._plot_heat(y_preds)
        return lst_lst_stats

    def _predict(self, x, y, params, coef=5):
        """
        leave one WSI (patch number of WSI=coef) out cross validation
        """ 
        lr = LogisticRegression(**params)
        y_pred=[]
        for i in range(int(len(y)/coef)):
            test=[coef*i+v for v in range(coef)]
            train=list(range(len(y)))
            for v in test:
                train.remove(v)
            lr.fit(x[train,:], y[train])
            y_pred.append(lr.predict_proba(x[test]))
        y_pred=np.concatenate(y_pred)
        return y_pred

    def _plot_heat(self, y_preds):
        lst_name=self.df_info["COMPOUND_NAME"].tolist()
        sns.heatmap(
            y_preds,
            xticklabels=self.le.classes_,
            vmin=0., vmax=1,
            cmap="bwr"
        )
        plt.yticks(
            [i*self.coef+int(self.coef/2) for i in range(int(self.lst_arr_x[0].shape[0]/self.coef))],
            [lst_name[i*self.coef+int(self.coef/2)] for i in range(int(self.lst_arr_x[0].shape[0]/self.coef))],)
        plt.show()

    def _set_label(self):
        le = LabelEncoder()
        self.df_info["y"]=le.fit_transform(self.df_info["COMPOUND_NAME"])
        self.y=self.df_info["y"].values
        self.le=le

class Clustering:
    def __init__(self):
        self.df_info_eisai=pd.DataFrame()
        self.lst_arr_x=[]
        self.arr_embedding=None

    def plot_clustering(
        self,
        folder=f"{root}/data/feature/eisai/feature_all",
        name="_frozen",
        pretrained=False,
        layer=4, size=None,
        n_model=5,      
        concat=False, meta_viz=False,
        number=0,
        title="",
        eisai_dataset=False, shionogi_dataset=False,
        tggate_dataset=False, 
        rat_dataset=False, mouse_dataset=False,
        time="24 hr"
        ):
        # data load
        ## features array
        self.lst_arr_x, self.arr_embedding = utils.load_array_preprocess(
            layer=layer, folder=folder, name=name, 
            size=size, pretrained=pretrained, n_model=n_model,
            convertz=False,
            compression=True, n_components=2, 
            meta_viz=meta_viz, concat=concat,
        )
        ## information dataframe
        if size:
            self.coef=int(2000/size)
        else:
            self.coef=10 # already compressed by size=200
        if eisai_dataset:
            self.lst_compounds=lst_compounds_eisai_conv
        elif shionogi_dataset:
            self.lst_compounds=lst_compounds_shionogi
        elif tggate_dataset:
            self.lst_compounds=lst_compounds_target
        elif rat_dataset:
            self.lst_compounds=lst_compounds_rat
        elif mouse_dataset:
            self.lst_compounds=lst_compounds_mouse
        self.df_info=utils.LoadInfo().load(
            coef=self.coef, time=time, conv_name=True,
            lst_compounds=self.lst_compounds,
            eisai_dataset=eisai_dataset, shionogi_dataset=shionogi_dataset, 
            tggate_dataset=tggate_dataset, 
            rat_dataset=rat_dataset, mouse_dataset=mouse_dataset,
            conc="High"
            )
        self._plot_scatter(embedding=(concat or meta_viz), number=number, title=title)
    
    def calc_f(
        self,
        folder=f"{root}/data/feature/eisai/feature_all",
        name="_frozen",
        pretrained=False,
        layer=4, size=None,
        n_model=5,
        random_f=False,
        convertz=True,
        compression=False, n_components=16,
        eisai_dataset=False, shionogi_dataset=False, 
        tggate_dataset=False, 
        rat_dataset=False, mouse_dataset=False,
        time="24 hr"
        ):
        # data load
        ## features array
        self.lst_arr_x, _ = utils.load_array_preprocess(
            layer=layer, folder=folder, name=name, 
            size=size, pretrained=pretrained, n_model=n_model,
            convertz=convertz,
            compression=compression, n_components=n_components, 
            meta_viz=False, concat=False,
        )
        ## information dataframe
        if size:
            self.coef=int(2000/size)
        else:
            self.coef=10 # already compressed by size=200
        if eisai_dataset:
            self.lst_compounds=lst_compounds_eisai_conv
        elif shionogi_dataset:
            self.lst_compounds=lst_compounds_shionogi
        elif tggate_dataset:
            self.lst_compounds=lst_compounds_target
        elif rat_dataset:
            self.lst_compounds=lst_compounds_rat
        elif mouse_dataset:
            self.lst_compounds=lst_compounds_mouse
        self.df_info=utils.LoadInfo().load(
            coef=self.coef, time=time, conv_name=True,
            lst_compounds=self.lst_compounds,
            eisai_dataset=eisai_dataset, shionogi_dataset=shionogi_dataset, 
            tggate_dataset=tggate_dataset, 
            rat_dataset=rat_dataset, mouse_dataset=mouse_dataset,
            conc="High"
            )
        # calc pseudo F score
        lst_f=[utils.pseudo_F(arr_x, self.df_info, "COMPOUND_NAME") for arr_x in self.lst_arr_x]
        if random_f:
            lst_f_random=[utils.pseudo_F(np.random.default_rng(random_state).permutation(arr_x,axis=0), self.df_info, "COMPOUND_NAME") for arr_x in self.lst_arr_x]
            return lst_f, lst_f_random
        else:
            return lst_f

    def _plot_scatter(self, embedding=False, number=0, title=""):
        if embedding:
            arr_embedding=self.arr_embedding
        else:
            arr_embedding=self.lst_arr_x[number]
        fig=plt.figure(figsize=(9,5.5))
        ax=fig.add_subplot(111)
        colors = sns.color_palette(n_colors=len(self.lst_plot_compounds))
        for i, compound in enumerate(self.lst_plot_compounds):
            arr_index=self.df_info[self.df_info["COMPOUND_NAME"]==compound]["INDEX"].tolist()
            ax.scatter(
                arr_embedding[arr_index,0],
                arr_embedding[arr_index,1],
                s=70, marker="o", 
                label=compound,
                linewidth=4, color="w", ec=colors[i]
            )
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        fig.suptitle(title)
        plt.legend(bbox_to_anchor=(1, 0), loc='lower left')
        plt.tight_layout()
        plt.show()        

