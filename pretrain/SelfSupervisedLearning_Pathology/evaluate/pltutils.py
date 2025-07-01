# -*- coding: utf-8 -*-
"""
# Plot utils for evaluation

@author: Katsuhisa MORITA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_scatter_circle(
    lst_values, lst_color=[], lst_style=[], ax=None
    ):
    def _plot_scatter_circle(
        arr_values, 
        move=0, position=0, n_plot=0, 
        color="", linestyle="-",
        ax=None, average_line=True,
        ):
        # average line
        if average_line:
            ax.axhline(
                y=arr_values.mean(), 
                xmin=position-move, xmax=position+move, 
                color="darkgrey", linewidth=2.5, zorder=1,
                linestyle=linestyle)
        # point plot
        ax.scatter(
            [position*(n_plot+1)-1]*len(arr_values),
            arr_values,
            s=40, marker="o", 
            color="w", alpha=.8,
            linewidth=3, ec=color, zorder=2
            )
    n_plot=len(lst_values)
    for i, lst_v in enumerate(lst_values):
        _plot_scatter_circle(
            np.array(lst_v),
            move=1/(3*n_plot-1), position=(i+1)/(n_plot+1), n_plot=n_plot,
            color=lst_color[i], linestyle=lst_style[i], ax=ax
            )

def plot_stripplot(
    lst_values, lst_color=[], lst_style=[], ax=None
    ):
    lst_mean=[np.nanmean(i) for i in lst_values]
    v_max=max([len(i) for i in lst_values])
    lst_values=[i+[np.nan]*(v_max-len(i)) for i in lst_values]
    df_melt = pd.DataFrame(dict(enumerate(lst_values))).melt()
    # stripplot
    sns.stripplot(
        x='variable', y='value', hue="variable", data=df_melt, 
        jitter=True, linewidth=1.5, size=9, alpha=0.7,
        palette=lst_color, ax=ax, zorder=2)
    # average line
    for i, v_mean in enumerate(lst_mean):
        plt.hlines(
            y=v_mean, xmin=i-.4, xmax=i+.4, 
            linestyle=lst_style[i], color="dimgrey", 
            alpha=.95, linewidth=3,zorder=1)
    ax.get_legend().remove()

def plot_heat(
    lst_values, 
    target="", name="", title="",
    ylim=[.5,1], figsize=(12,8)
    ):
    df_plt = pd.DataFrame(index=lst_values[0].index.tolist())
    for i, df in enumerate(lst_res):
        df_plt[f"{name}_Layer{i+1}"]=df[f"{target}_mean"]
    plt.rcParams["font.size"] = 14
    fig=plt.figure(figsize=figsize)
    sns.heatmap(
        df_plt.T,
        vmin=ylim[0],vmax=ylim[1],
        cmap="Blues",
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()
        
def plot_scatter_confidence(
    lst_df, 
    target="",
    average_line=True, control=0.5, 
    ylim=[], move=0.25, figsize=(18,10),
    xlabels=[]
    ):
    def plot_stats_ax(
        df, 
        target="", 
        move=0, 
        ax=None, 
        ec="black",
        xlabels=[],
        ):
        arr_values=df[f"{target}_mean"].values
        arr_interval=df[f"{target}_interval"].values
        ax.errorbar(
            [i+move for i in range(len(arr_values))], arr_values,
            yerr=[v_int[1] - v_m for v_m, v_int in zip(arr_values, arr_interval)], 
            lw=.6, capthick=2, capsize=3, ecolor="black",
            color="w", barsabove=True, fmt='.r', markersize=.1, zorder=1
            )
        for i, xlabel in enumerate(x_labels):
            v_v = df.loc[label, f"{target}_v"]
            # dot plot 
            led = ax.scatter(
                [i+move]*len(v_v),
                v_v,
                s=10, marker="o", 
                color="w",
                linewidth=2, ec=ec, zorder=2)
        return led
    fig=plt.figure(figsize=figsize)
    ax=fig.add_subplot(111)
    lst_legend=[]
    cmap = plt.get_cmap('hsv')
    space=1/(len(lst_df))
    for i, df in enumerate(lst_df):
        led = plot_stats_ax(
            df, target=target, 
            move=(-2+i)*space, ax=ax, 
            ec=cmap(i/len(lst_df)),
            xlabels=xlabels,
            )
        lst_legend.append(led)
    # control line
    ax.hlines(y=control,xmin=-1,xmax=len(xlabels), color='grey', linestyles='dashed')
    ax.set_title(f"{target.upper()}")
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlim(-1, len(xlabels))
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=90, fontsize=15)
    delete_frame()
    plt.grid(False)
    plt.legend(legend, [f"layer{i+1}" for i in range(5)])
    plt.tight_layout()
    plt.show()

def plot_violin(
    lst_values, 
    lst_name=list(), 
    ax=None,
    ):
    # Preprocessing
    lst_mean=[np.nanmean(i) for i in lst_values]
    # fill small data list with na
    v_max=max([len(i) for i in lst_values])
    lst_values=[i+[np.nan]*(v_max-len(i)) for i in lst_values]
    df = pd.DataFrame(dict(zip(lst_name, lst_values)))
    df_melt=pd.melt(df)
    # violin plot, transparent
    sns.violinplot(x='variable', y='value', data=df_melt, inner=None, cut=0, scale="count",linewidth=1.5, color="dimgrey")
    plt.setp(ax.collections, alpha=.55)
    # average line
    for i, v_mean in enumerate(lst_mean):
        plt.hlines(y=v_mean, xmin=i-.4, xmax=i+.4, linestyle="dotted", color="black", alpha=0.8, linewidth=3)
    sns.stripplot(x='variable', y='value', data=df_melt, jitter=True, linewidth=1.5, size=9, color="firebrick")

def delete_frame():
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')

def change_lst_order(lst, lst_order=[]):
    """for ssl comparison"""
    return [lst[i] for i in lst_order]

def set_yticks(ymin=0, ymax=1, intv=.1):
    n_t=int((ymax-ymin-.0001)//intv)+2
    yticks=[(ymax*1000-int(i*intv*1000))/1000 for i in reversed(range(n_t))]
    yticks=[i for i in yticks if i>=0]
    return yticks

class PlotPredComp:
    def __init__(self):
        return
    def plot_result(
        self,
        folder="", 
        dataset="",
        task="",
        pf=False,
        pca="",
        order=True,
        figsize=(9,5.5),
        ylim=[],
        xticks=None,
        yticks=None,
        method="strip",
        ax=None
        ):
        # Names
        target_dict={
            "ssl": [
                ["BarlowTwins","SimSiam","BYOL","SwAV",],
                ["", "\nAdd"]],
            "model": [
                ["ResNet18", "DenseNet121", "EfficientNetB3", "ConvNetTiny", "RegNetY_1.6GRF",],
                ["\nPre-trained", "\nBarlowTwins", "\nBarlowTwins_Add"]],
            "reduced":[
                [0, 100, 200, 500, 1000, 2000, 3000, "Full (6058)"],
                ["", "\nAdd"]],
            "ablation":[
                ["Full", "-Color", "-Blur", "-Color-Blur"],
                ["\nBarlowTwins", "\nBarlowTwins_Add"]]
        }
        dataset_dict={
            "eisai":"Eisai",
            "shionogi":"Shionogi",
            "our4d":"Our 4 Days",
            "our24h":"Our 24 hours",
        }
        # Setup
        if pf:
            title=f"Clustering Separation ({dataset_dict[dataset]} Dataset)"
            ylabel="Pseudo F Score"
            if len(pca)>0:
                filein=f"{folder}/result/{task}/pf_{dataset}_pca{pca}.pickle"
            else:
                filein=f"{folder}/result/{task}/pf_{dataset}.pickle"
        else:
            title=f"Compound Name Classification ({dataset_dict[dataset]} Dataset)"
            ylabel="Accuracy"
            filein=f"{folder}/result/{task}/lowcv_acc_{dataset}.pickle"
        lst_label=[f"{v}{i}" for i in target_dict[task][1] for v in target_dict[task][0]]
        if task=="reduced":
            lst_res=self._load_fullmodel(filein)
        else:
            lst_res=pd.read_pickle(filein)
        # Plot
        lst_color=[
            "firebrick", 
            "darkblue",
            "darkorange",
            "darkgreen",
            "violet",
            ]
        lst_style=[
            "solid",
            "dotted",
            "dashed"
        ]
        n_t=len(target_dict[task][0])
        n_model=len(target_dict[task][1])
        if task=="model":
            lst_color=lst_color[:n_t]*n_model
            lst_style=[v for i in lst_style[:n_model] for v in [i]*n_t]
            lst_order=[v*n_t+i for i in range(n_t) for v in range(n_model)]
        else:
            if n_t>5:
                lst_color=["firebrick"]*len(lst_res)
            lst_color=["dimgrey"]+lst_color[:n_t]*n_model
            lst_style=["solid"]+[v for i in lst_style[:n_model] for v in [i]*n_t]
            lst_order=[0]+[v*n_t+i+1 for i in range(n_t) for v in range(n_model)]
            lst_label=["Pre-trained"]+lst_label
        if order:
            lst_color=change_lst_order(lst_color, lst_order=lst_order)
            lst_style=change_lst_order(lst_style, lst_order=lst_order)
            lst_label=change_lst_order(lst_label, lst_order=lst_order)
            lst_res=change_lst_order(lst_res, lst_order=lst_order)
        if pf:
            lst_res=[[v for i in lst_res for v in i[1]]]+[i[0] for i in lst_res] # [random f] + [others]
            lst_color=["dimgrey"]+lst_color
            lst_style=["solid"]+lst_style
            lst_label=["Control"]+lst_label
        if ax:
            self._plot_res_ax(
                lst_res, lst_label,
                lst_color=lst_color, lst_style=lst_style,
                ylim=ylim, yticks=yticks, xticks=xticks,
                title=title, ylabel=ylabel, 
                method=method, ax=ax
            )
        else:
            self._plot_res(
                lst_res, lst_label,
                lst_color=lst_color, lst_style=lst_style,
                figsize=figsize, ylim=ylim, yticks=yticks,
                title=title, ylabel=ylabel, 
                method=method,
            )

    def _load_fullmodel(self, filein):
        """load fullmodel result for WSI number study"""
        lst=pd.read_pickle(filein)
        fulllst=pd.read_pickle(filein.replace("reduced", "model"))
        lst_res=[]
        lst_res+=[lst[i] for i in range(0,8)]
        lst_res.append(fulllst[5])
        lst_res+=[lst[i] for i in range(8,15)]
        lst_res.append(fulllst[10])
        return lst_res

    def _plot_res(
        self, 
        lst_res, lst_label, 
        lst_color=list(), lst_style=list(),
        figsize=(9,6), ylim=list(), yticks=None,
        title="", ylabel="", method="strip",
        ):
        methods_dict={
            "strip":plot_stripplot,
            "scatter_circle":plot_scatter_circle,
        }
        # Plot
        fig=plt.figure(figsize=figsize)
        ax=fig.add_subplot(111)
        methods_dict[method](
            lst_res, lst_color=lst_color, lst_style=lst_style, ax=ax
        )
        if ylim:
            ax.set_ylim(*ylim)
        if yticks:
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticks)
        ax.set_ylabel(ylabel)
        ax.set_xlim(-1,len(lst_res))
        ax.set_xticks(range(len(lst_res)))
        ax.set_xticklabels(
            lst_label,
            rotation=90)
        delete_frame()
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def _plot_res_ax(
        self,
        lst_res, lst_label, 
        lst_color=list(), lst_style=list(),
        ylim=list(), yticks=None, xticks=True,
        title="", ylabel="", method="strip", ax=None,
        ):
        methods_dict={
            "strip":plot_stripplot,
            "scatter_circle":plot_scatter_circle,
        }
        methods_dict[method](
            lst_res, lst_color=lst_color, lst_style=lst_style, ax=ax,
        )
        if ylim:
            ax.set_ylim(*ylim)
        if yticks:
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticks)
        ax.set_ylabel(ylabel)
        ax.set_xlim(-1,len(lst_res))
        if xticks:
            ax.set_xticks(range(len(lst_res)))
            ax.set_xticklabels(
                lst_label,
                rotation=90)
        delete_frame()
        ax.set_title(title)
        return ax  

class PlotPredFold:
    def __init__(self):
        return

    def plot_result(
        self,
        task="prognosis",
        target="AUROC", ylabel="AUROC", # AUROC, AUPR, mAP, ...
        eval_method="macro", # For moa/compound_name task
        lst_filein=list(),
        lst_name=list(),
        figsize=(9,5.5),
        ylim=[0,1], yticks=None, intv=.05,
        rotation=60,
        ):
        # Set Title
        dict_task={
            "prognosis":["Prognosis Prediction",],
            "finding":["Finding Classification",],
            "moa":["MoA Classification",],
            "compound_name":["Compound Name Classification",],
        }
        dict_position={
            "Accuracy":1,
            "AUROC":3,
            "AUPR":4,
            "Balanced Accuracy":2,
        }
        title=dict_task[task][0]
        # Load
        lst_lst_res=[pd.read_pickle(filein) for filein in lst_filein]
        if task=="moa" or task=="compound_name":
            if eval_method=="macro":
                # evaluate (one indicator for one sample, average across fold)
                lst_lst_res=[[v[0] for v in i] for i in lst_lst_res]
            elif eval_method=="micro":
                # evaluate (one indicator for one fold)
                lst_lst_res=[[v[dict_position[target]] for v in i] for i in lst_lst_res]
        else:
            lst_lst_res=[self._calc_mean(i, target=target) for i in lst_lst_res]

        # Plot
        self._plot_res(
            lst_lst_res, lst_name=lst_name,
            title=title, ylabel=ylabel,
            figsize=figsize, ylim=ylim, yticks=yticks, intv=intv,
            grid=False, rotation=rotation)

    def plot_result_pf(
        self,
        lst_filein=list(),
        lst_name=list(),
        title="Clustering Separation (MoA)",
        figsize=(9,5.5),
        rotation=90,
        ):
        # Load
        lst_res=[pd.read_pickle(filein) for filein in lst_filein]
        lst_res=[[v for i in lst_res for v in i[1]]]+[i[0] for i in lst_res] # [Control] + [Results]
        # Plot
        self._plot_res(
            lst_res, lst_name=lst_name,
            title=title, ylabel="Pseudo F Score",
            figsize=figsize, ylim=None,
            grid=False, rotation=rotation)

    def _calc_mean(self, lst_res, target:str=""):
        df_temp = pd.concat([df.loc[:,f"{target}"] for df in lst_res], axis=1)
        return df_temp.mean(axis=1, skipna=True).tolist()

    def _plot_res(
        self,    
        lst_values, 
        lst_name=list(), 
        title="title", 
        ylabel="label", 
        figsize=(6,6),
        ylim=[0,1], yticks=list(), intv=.05,
        grid=False, rotation=60):
        # Plot
        fig = plt.figure(figsize=figsize)
        ax=fig.add_subplot(111)
        plot_violin(
            lst_values,
            lst_name=lst_name, 
            ax=ax,)
        # set
        if ylim:
            ax.set_ylim(*ylim)
            if not yticks:
                yticks=set_yticks(ymin=ylim[0], ymax=1, intv=intv)
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticks)
        ax.set_xticks(range(len(lst_name)))
        ax.set_xticklabels(lst_name, rotation=rotation)
        ax.set_title(title)
        ax.set_xlabel("Model Name")
        ax.set_ylabel(ylabel)
        if grid:
            ax.grid(color="#ababab",linewidth=0.5)
        delete_frame()
        plt.show()