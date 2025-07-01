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
            color=lst_color[i], linestyle=lst_style[i],
            )

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
    lst_df, 
    lst_name=["Constant", "Label Prediction", "Pre-Trained", "BarlowTwins"], 
    target="auroc", 
    title="title", 
    label="label", 
    ylim=[0,1], figsize=(6,6),
    grid=False,
    ):
    # Preprocessing
    lst_values=[i.loc[:,f"{target}_mean"].values for i in lst_df]
    lst_mean=[np.mean(i) for i in lst_values]
    df = pd.DataFrame(dict(zip(lst_name, lst_values)))
    df_melt=pd.melt(df)
    fig = plt.figure(figsize=figsize)
    # Plot
    ax=fig.add_subplot(111)
    # violin plot, transparent
    sns.violinplot(x='variable', y='value', data=df_melt, inner=None, cut=0, scale="count",linewidth=1.5, color="grey")
    plt.setp(ax.collections, alpha=.55)
    # average line
    for i, v_mean in enumerate(lst_mean):
        plt.hlines(y=v_mean, xmin=i-.4, xmax=i+.4, linestyle="dotted", color="black", alpha=0.8, linewidth=3)
    sns.stripplot(x='variable', y='value', data=df_melt, jitter=True, linewidth=1.5, size=9, color="firebrick")
    # set
    ax.set_title(title,fontsize=16)
    ax.set_xlabel("Model Name",fontsize=16)
    ax.set_ylabel(label,fontsize=16)
    ax.set_xticklabels(lst_name, fontsize=16, rotation=60)
    if grid:
        ax.grid(color="#ababab",linewidth=0.5)
    ax.set_ylim(*ylim)
    delete_frame()
    plt.show()


def delete_frame():
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')

