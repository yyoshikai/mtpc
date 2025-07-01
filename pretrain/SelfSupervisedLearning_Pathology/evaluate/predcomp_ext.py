# -*- coding: utf-8 -*-
"""
# Extraporation from TG-GATE to Eisai dataset

@author: Katsuhisa MORITA
"""
import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sklearn.metrics as metrics

from evaluate import utils

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams["font.size"] = 14

# Settings
root = "/workspace/230727_pharm"
lst_compounds_target_eisai=[
    "vehicle",
    "bromobenzene",
    "carbon tetrachloride",
    "naphthyl isothiocyanate",
    "acetaminophen",
]
lst_compounds_target_rat=[
    "vehicle",
    "thioacetamide",
    "carbon tetrachloride",
    "naphthyl isothiocyanate",
    "acetaminophen",
]
lst_compounds_target_mouse=[
    "vehicle",
    "thioacetamide",
    "carbon tetrachloride",
    "naphthyl isothiocyanate",
    "acetaminophen",
]
class PredictCompExt:
    def __init__(self):
        self.lst_arr_x=[]
        self.lst_arr_x2=[]
        self.df_info=pd.DataFrame()
        self.df_info2=pd.DataFrame()
        self.y_train=None
        self.y_test=None
        self.le=None

    def evaluate(
        self,
        folder_tggate=f"{root}/data/feature/target_all",
        folder2=f"{root}/data/feature/eisai/feature_all",
        name="_frozen",
        pretrained=False,
        layer=4, size=None,
        n_model=5,
        convertz=True, z_ctrl=False,
        combat=False,
        compression=False, n_components=16,
        params_lr=dict(),
        plot_heat=False,
        eisai_dataset=False, rat_dataset=False, mouse_dataset=False,
        time="24 hr",
        ):
        """ evaluate compound prediction """
        # data load
        ## information dataframe
        if size:
            self.coef=int(2000/size)
        else:
            self.coef=10 # already compressed by size=200
        if eisai_dataset:
            self.lst_target_pred = lst_compounds_target_eisai
        elif rat_dataset:
            self.lst_target_pred = lst_compounds_target_rat
        elif mouse_dataset:
            self.lst_target_pred = lst_compounds_target_mouse

        self.df_info=utils.LoadInfo().load(
            coef=self.coef, time=time, conv_name=True,
            lst_compounds=self.lst_target_pred,
            tggate_dataset=True, conc="High"
            )

        self.df_info2=utils.LoadInfo().load(
            coef=self.coef, time=time, conv_name=True,
            lst_compounds=self.lst_target_pred,
            eisai_dataset=eisai_dataset,
            rat_dataset=rat_dataset, mouse_dataset=mouse_dataset,
            conc="High"
            )

        ## features array
        self.lst_arr_x, self.lst_arr_x2, self.arr_embedding, self.arr_embedding2 = utils.load_array_preprocess_two(
            layer=layer, size=size, 
            folder1=folder_tggate, folder2=folder2, 
            name=name, pretrained=pretrained, n_model=n_model,
            convertz=convertz, z_ctrl=z_ctrl, 
            ctrl_pos=self.df_info[self.df_info["COMPOUND_NAME"]=="vehicle"]["INDEX"].tolist(), 
            ctrl_pos2=self.df_info2[self.df_info2["COMPOUND_NAME"]=="vehicle"]["INDEX"].tolist(), 
            combat=combat,
            compression=compression, n_components=n_components,
            concat=False, meta_viz=False,
        )
        ## Set label and y
        self._set_label()

        ## Predict / Evaluate
        lst_lst_stats=[]
        for i, (arr_x, arr_x2) in enumerate(zip(self.lst_arr_x, self.lst_arr_x2)):
            y_pred = self._predict(
                arr_x[self.df_info["INDEX"].tolist()], 
                arr_x2[self.df_info2["INDEX"].tolist()], 
                self.y_train, params_lr
                )
            lst_stats = utils.calc_stats(self.y_test, y_pred, self.lst_target_pred, self.le)
            lst_lst_stats.append(lst_stats)
            if plot_heat:
                if i==0:
                    y_preds=y_pred/n_model
                else:
                    y_preds+=y_pred/n_model
        if plot_heat:
            self._plot_heat(y_preds)
        return lst_lst_stats
    
    def _set_label(self,):
        le = LabelEncoder()
        le.fit(self.df_info["COMPOUND_NAME"])
        self.df_info["y"]=le.transform(self.df_info["COMPOUND_NAME"])
        self.df_info2["y"]=le.transform(self.df_info2["COMPOUND_NAME"])
        self.y_train=self.df_info["y"].values
        self.y_test=self.df_info2["y"].values
        self.le=le

    def _predict(self, x_train, x_test, y_train, params):
        """prediction with logistic regression"""
        lr = LogisticRegression(**params)
        lr.fit(x_train, y_train)
        y_pred=lr.predict_proba(x_test)
        return y_pred

    def _plot_heat(self, y_preds):
        plt.figure(figsize=(6,8))
        lst_name=self.df_info2["COMPOUND_NAME"].tolist()
        sns.heatmap(
            y_preds,
            xticklabels=self.le.classes_,
            vmin=0., vmax=1,
            cmap="bwr"
        )
        plt.yticks(
            [i*self.coef+int(self.coef/2) for i in range(int(self.lst_arr_x2[0].shape[0]/self.coef))],
            [lst_name[i*self.coef+int(self.coef/2)] for i in range(int(self.lst_arr_x2[0].shape[0]/self.coef))],)
        plt.ylabel("Ground Truth")
        plt.xlabel("Prediction Label")
        plt.show()

class ClusteringExt:
    def __init__(self):
        self.df_info=pd.DataFrame()
        self.df_info2=pd.DataFrame()
        self.lst_arr_x=None
        self.lst_arr_x_eisai=None
        self.arr_embedding=None
        self.arr_embedding2=None

    def plot_clustering(
        self,
        folder_tggate=f"{root}/data/feature/target_all",
        folder2=f"{root}/data/feature/eisai/feature_all",
        name="_frozen",
        pretrained=False,
        layer=4, size=None,
        n_model=5,      
        combat=False,
        convertz=False, z_ctrl=False,
        concat=False, meta_viz=False,
        number=0,
        title="",
        eisai_dataset=False, rat_dataset=False, mouse_dataset=False,
        time="24 hr"
        ):
        """ evaluate compound prediction """
        # data load
        ## information dataframe
        if size:
            self.coef=int(2000/size)
        else:
            self.coef=10 # already compressed by size=200
        if eisai_dataset:
            self.lst_target_pred = lst_compounds_target_eisai
        elif rat_dataset:
            self.lst_target_pred = lst_compounds_target_rat
        elif mouse_dataset:
            self.lst_target_pred = lst_compounds_target_mou
        self.df_info = utils.load_tggate(
            coef=self.coef, time=time,
            lst_compounds=self.lst_target_pred)
        self.df_info2=utils.LoadInfo().load(
            coef=self.coef, conv_name=True, time=time,
            eisai_dataset=eisai_dataset, rat_dataset=rat_dataset, mouse_dataset=mouse_dataset,
            lst_compounds=self.lst_taret_pred
        )

        ## features array
        self.lst_arr_x, self.lst_arr_x2, self.arr_embedding, self.arr_embedding2 = utils.load_array_preprocess_two(
            layer=layer, size=size, 
            folder1=folder_tggate, folder2=folder2, 
            name=name, pretrained=pretrained, n_model=n_model,
            convertz=convertz, z_ctrl=z_ctrl, 
            ctrl_pos=self.df_info[self.df_info["COMPOUND_NAME"]=="vehicle"]["INDEX"].tolist(), 
            ctrl_pos2=self.df_info2[self.df_info2["COMPOUND_NAME"]=="vehicle"]["INDEX"].tolist(), 
            combat=combat,
            compression=True, n_components=2,
            concat=concat, meta_viz=meta_viz,
        )
        self._plot_scatter(embedding=(concat or meta_viz), number=number, title=title)
    
    def calc_f(
        self,
        folder_tggate=f"{root}/data/feature/target_all",
        folder2=f"{root}/data/feature/eisai/feature_all",
        name="_frozen",
        pretrained=False,
        layer=4, size=None,
        n_model=5,
        random_f=False,
        combat=False,
        convertz=True, z_ctrl=True,
        compression=False,
        n_components=16,
        eisai_dataset=True, rat_dataset=False, mouse_dataset=False,
        time="24 hr",
        random_state=24771,
        ):
        # data load
        ## features array
        if size:
            self.coef=int(2000/size)
        else:
            self.coef=10 # already compressed by size=200
        if eisai_dataset:
            self.lst_target_pred = lst_compounds_target_eisai
        elif rat_dataset:
            self.lst_target_pred = lst_compounds_target_rat
        elif mouse_dataset:
            self.lst_target_pred = lst_compounds_target_mou
        self.df_info = utils.load_tggate(
            coef=self.coef, time=time,
            lst_compounds=self.lst_target_pred)
        self.df_info2=utils.LoadInfo().load(
            coef=self.coef, conv_name=True, time=time,
            eisai_dataset=eisai_dataset, rat_dataset=rat_dataset, mouse_dataset=mouse_dataset,
            lst_compounds=self.lst_taret_pred
        )

        ## features array
        self.lst_arr_x, self.lst_arr_x2, self.arr_embedding, self.arr_embedding2 = utils.load_array_preprocess_two(
            layer=layer, size=size, 
            folder1=folder_tggate, folder2=folder2,
            name=name, pretrained=pretrained, n_model=n_model,
            convertz=convertz, z_ctrl=z_ctrl, 
            ctrl_pos=self.df_info[self.df_info["COMPOUND_NAME"]=="vehicle"]["INDEX"].tolist(), 
            ctrl_pos2=self.df_info2[self.df_info2["COMPOUND_NAME"]=="vehicle"]["INDEX"].tolist(), 
            combat=combat,
            compression=compression, n_components=n_components,
            concat=False, meta_viz=False,
        )
        df_temp=copy.deepcopy(self.df_info2)
        n_tggate=self.lst_arr_x[0].shape[0]
        df_temp["INDEX"]=[int(i+n_tggate) for i in df_temp["INDEX"]]
        df_temp=pd.concat([self.df_info, df_temp],axis=0)
        lst_f=[utils.pseudo_F(
            np.concatenate([arr_x, arr_x2], axis=0), 
            df_temp, 
            "COMPOUND_NAME"
            ) for arr_x, arr_x2 in zip(self.lst_arr_x, self.lst_arr_x2)]
        if random_f:
            lst_f_random=[utils.pseudo_F(
                np.random.default_rng(random_state).permutation(np.concatenate([arr_x, arr_x2], axis=0),axis=0),
                df_temp, 
                "COMPOUND_NAME"
                ) for arr_x, arr_x2 in zip(self.lst_arr_x, self.lst_arr_x2)]
            return lst_f, lst_f_random
        else:
            return lst_f

    def _plot_scatter(self, embedding=False, number=0, title=""):
        if embedding:
            arr_embedding, arr_embedding2=self.arr_embedding, self.arr_embedding2
        else:
            arr_embedding, arr_embedding2=self.lst_arr_x[number], self.lst_arr_x2[number]
        fig=plt.figure(figsize=(10,6))
        ax=fig.add_subplot(111)
        colors = sns.color_palette(n_colors=len(self.lst_compounds_target)*2)
        for i, compound in enumerate(self.lst_compounds_target):
            arr_index=self.df_info[self.df_info["COMPOUND_NAME"]==compound]["INDEX"].tolist()
            ax.scatter(
                arr_embedding[arr_index,0],
                arr_embedding[arr_index,1],
                s=70, marker="o", 
                label=f"{compound}_TG-GATE",
                linewidth=4, color="w", ec=colors[i*2]
            )
            arr_index=self.df_info2[self.df_info2["COMPOUND_NAME"]==compound]["INDEX"].tolist()
            ax.scatter(
                arr_embedding2[arr_index,0],
                arr_embedding2[arr_index,1],
                s=70, marker="o", 
                label=f"{compound}_Ext_Dataset",
                linewidth=4, color="w", ec=colors[i*2+1]
            )
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        fig.suptitle(title)
        plt.legend(bbox_to_anchor=(1, 0), loc='lower left')
        plt.tight_layout()
        plt.show()        

