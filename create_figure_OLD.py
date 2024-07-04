# %%
import typing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statannot import add_stat_annotation

from functools import partial

def get_x_name(text):
    text.text
    return


def annotate_p_values(data,ax=None, 
                      x = None,
                      y = None,
                      pairs = None, scale = 0.01, 
                      ylim = None, **kws):
    if ax == None:
        ax = plt.gca()
    xaxis = {x.get_text(): x.get_position()[0] for x in ax.get_xticklabels()}
    print(xaxis)
    if ylim == None:
        ylim = ax.get_ylim()
    range = np.diff(ylim)[0]
    ylim_top = ylim[1]
    lab = ax.get_xticklabels()
    groups = list(data[x].unique())
    print(groups)
    for pair in pairs:
        annot_stat(ax, "***", xaxis[pair[0]],xaxis[pair[1]], ylim_top, range*scale)
        ylim_top += range*scale*10
    return ax

def annot_stat(ax, star, x1, x2, y, h, col='k'):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    ax.text((x1+x2)/2, y+h, star, ha='center', va='bottom', color=col)
    return ax

def pointplot(data: pd.DataFrame,
              y = "values",
              x = "Group",
              hue = "Group",
              palette = sns.color_palette(["#003f5c",
                                 "#ffa600",
                                 "#bc5090"]),
              alpha = 0.5,
              sigpairs = [("X","TP"),
                          ("X","FY")]
              ):
    ax = sns.stripplot(data,
                        x = x,
                        y = y,
                        hue = hue,
                        palette = palette,
                        alpha = alpha,
                        legend = False)
    ax = annotate_p_values(data,ax=ax, data=data, x=x, y=y,
                             pairs = sigpairs)
    ax = sns.pointplot(data,
                       ax = ax,
                        x = x,
                        y = y,
                        hue = hue,
                        palette = palette,
                        legend = False,
                        capsize = 0.1)
    
    return ax

def barplot(data: pd.DataFrame,
              y = "values",
              x = "Group",
              hue = "Group",
              palette = sns.color_palette(["#003f5c",
                                 "#ffa600",
                                 "#bc5090"]),
              sigpairs = [("X","TP"),
                          ("X","FY")]
              ):
    ax = sns.barplot(data,
                        x = x,
                        y = y,
                        hue = hue,
                        palette = palette,
                        legend = False,
                        capsize = 0.1,
                        err_kws = {"color": "black",
                                   "linewidth": 2},
                        linewidth=2,
                        edgecolor="black")
    ax = annotate_p_values(data,ax=ax, data=data, x=x, y=y,
                             pairs = sigpairs)
    
    return ax

def catplot(data: pd.DataFrame,
              y = "values",
              x = "Group1",
              col = "Group2",
              palette = sns.color_palette(["#003f5c",
                                 "#ffa600",
                                 "#bc5090"]),
              sigpairs = [("X","TP"),
                          ("X","FY")]
              ):
    cp = sns.catplot(data,
                     kind = "bar",
                        x = x,
                        y = y,
                        col = col,
                        palette = palette,
                        legend = False,
                        capsize = 0.1,
                        err_kws = {"color": "black",
                                   "linewidth": 2},
                        linewidth=2,
                        edgecolor="black")
    cp.map_dataframe(annotate_p_values, x=x, y=y,
                             pairs = sigpairs, ylim = cp.axes[0][0].get_ylim())
    
    return cp.set_titles(col_template="{col_name}")


class figure:

    def __init__(self) -> None:
        pass

    def add_plot(self, plot, x = 0, y = 0):
        return
    
    def render(self):
        return
    
test_file = r"E:\diploma_thesis\Data\5. Plasmid-mediated degradation does not get affected by Inhibition of Autophagy\Flourometry\Results.csv"
test_df = (pd.read_csv(test_file)
           .loc[:,["Mean_norm_X","Group1","Group2"]]
           .rename(columns={"Mean_norm_X": "values"}))


fig = plt.figure()
ax = catplot(test_df)

fig.add_subplot(ax)
fig.show()

# %%

# %%
