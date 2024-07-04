import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from functools import partial, reduce
from itertools import product

import seaborn as sns


from analyze_data import result

from analyze_data import convert_pvalue_to_asterisks





def adjust_level(pcoords):
    pcoords_adjusted = pcoords.copy()    
    pcoords_adjusted["y_lo"] = pcoords_adjusted.groupby("level")["y_lo"].transform('max')
    pcoords_adjusted["y_hi"] = pcoords_adjusted.groupby("level")["y_hi"].transform('max')
    return pcoords_adjusted

def get_max(ind, max_v):
    if ind[0] < ind[1]:
        return np.max(max_v[ind[0]:ind[1]+1])
    elif ind[0] > ind[1]:
        return np.max(max_v[ind[1]:ind[0]+1])
    else:
        return np.max(max_v[ind[0]])

def create_2d_matrix(vector, 
                     merge_func = partial(np.max, axis =2)):
    matrix = np.vstack([vector for i in range(len(vector))])
    array = merge_func(np.stack([matrix, matrix.T], axis = 2))
    return array

def get_p_coords(ftable, p_table, scaling):
    index_lab = ftable.index.map(lambda x: x if type(x) is not tuple else "_".join([str(y) for y in x]))
    p_coords = pd.DataFrame(index = pd.MultiIndex.from_product([index_lab,
                                                               index_lab], names = ["control", "treatment"]),
                           columns = ["x1", "x2", "y_lo", "y_hi", "p", "star", "text", "level"])
    index_mat = create_2d_matrix(index_lab.values, merge_func = lambda x: x)
    p_array = (p_table
          .reindex(ftable.index, axis = 0)
         .reindex(ftable.index, axis = 1)
         .to_numpy())
    max_v = ftable.to_numpy()
    level_v = np.zeros_like(max_v)
    step = np.max(max_v) * scaling
    tracking = p_array >= 0
    coords = np.indices(tracking.shape).T
    diff = np.abs(np.diff(coords,axis=2).T)[0]
    while np.sum(tracking) > 0:
        max_array = np.ma.masked_array(np.apply_along_axis(partial(get_max, max_v = max_v), 2, coords), mask=~tracking) 
        diff_array = np.ma.masked_array(diff, mask=~(max_array == np.min(max_array)))
        i,j = np.unravel_index(np.argmin(diff_array), diff_array.shape)
        i_s, j_s = tuple(sorted((i,j)))
        max_value = max_array[i,j]
        df_index = tuple(index_mat[i,j])
        p_coords.loc[df_index, "x1"] = i
        p_coords.loc[df_index, "x2"] = j
        p_coords.loc[df_index, "y_lo"] = max_value + step
        p_coords.loc[df_index, "y_hi"] = max_value+step*(1+scaling)
        p_coords.loc[df_index, "p"] = p_array[i,j]
        p_coords.loc[df_index, "text"] = (i+j)/2
        p_coords.loc[df_index, "level"] = np.max(level_v[i_s:j_s+1])
        level_v[i_s:j_s+1] += 1
        max_v[i_s:j_s+1] += step*(1+scaling)
        tracking[i,j] = False
    p_coords["star"] = convert_pvalue_to_asterisks(p_coords["p"])
    return p_coords.dropna()

def add_p_bars(ax, ftable, p_table, scaling, p_coords,
              level_adj):
    if p_coords.empty == True:
        p_coords = get_p_coords(ftable, p_table, scaling)
    if level_adj:
        p_coords = adjust_level(p_coords)
    ax.plot(np.vstack([p_coords.x1,
                      p_coords.x1,
                      p_coords.x2,
                      p_coords.x2]),
           np.vstack([p_coords.y_lo,
                      p_coords.y_hi,
                      p_coords.y_hi,
                      p_coords.y_lo]),
           color = "black")
    for _,row in p_coords.loc[:,["y_hi", "y_lo", "text", "star"]].iterrows():
        ax.text(row.text, row.y_hi + (row.y_hi - row.y_lo), row.star,
               horizontalalignment = "center")
    return ax

def add_multilevel_xticks(ax, labels,
                          col_width = 0.8):
    top_labels = labels[0]
    if labels.shape[0] > 1:
        top_labels = labels[0]
        ind = np.indices(labels[1:,:].shape)
        tick_df = (pd.DataFrame({"label": labels[1:,:].ravel(),
                                "level": ind[0][0],
                                "index": ind[1][0]}).groupby("label")
                   .agg(f = pd.NamedAgg("index", lambda d: np.min(d)),
                        l = pd.NamedAgg("index", lambda d: np.max(d)),
                        place = pd.NamedAgg("index", lambda d: np.mean(d)),
                        level = pd.NamedAgg("level", lambda d: int(np.median(d)+1)*2))
                   .drop("None", axis=0, errors='ignore')
                   .sort_values(by = "place")
                   .reset_index())
        tick_df["labels_spaced"] = tick_df.apply(lambda x: "".join([*(['\n']*x.level),x.label]), axis=1)
        tick_df["int"] = tick_df["place"].apply(lambda x: x.is_integer())
        for _,row in tick_df.iterrows():
            if row.int:
                top_labels[int(row.place)] = "".join([top_labels[int(row.place)], row.labels_spaced])
            ax.hlines(-0.045*row.level, row.f-col_width/2, row.l+col_width/2, color='black', lw=1, clip_on=False, transform=ax.get_xaxis_transform())
        ax.set_xticks(tick_df["place"], labels=tick_df["labels_spaced"], minor = True)
    ax.set_xticks(np.arange(len(top_labels)), labels=top_labels) 
    return ax

def boxplot(ax,
            data,
            table,
            data_c = "Mean",
            title = "",
            x_label = "Plasmid",
            y_label = "values",
              palette = ["#003f5c",
                     "#ffa600",
                     "#bc5090"],
            x_units = ("",1),
            y_units = ("",1),
            scaling = 0.1,
            max_v = "ci_hi",
            p_coords = pd.DataFrame(),
            level_adj = True,
            col_width = 0.7,
            **kwargs
              ):
    ftable = table.copy()
    data_red = data.copy().set_index(ftable.index.names).loc[:,[data_c]].sort_index() * y_units[1]
    ftable["id"] = np.arange(0,len(table))
    ftable = ftable * y_units[1]
    labels = ftable.index.to_frame(index=False).sort_index(axis = 1, ascending = True).to_numpy().T
    datalist = [data_red.loc[x, data_c].to_numpy() for x in ftable.index.values]
    box = ax.boxplot(datalist,
                    positions = range(len(datalist)),
                    patch_artist=True,
                    widths = col_width,
                    showfliers=False,
                  **kwargs
                )
    
    ax.set_title(title)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel(x_label if len(x_units[0])==0 else f"{x_label} [{x_units[0]}]")
    ax.set_ylabel(y_label if len(y_units[0])==0 else f"{y_label} [{y_units[0]}]")
    ax.set_ylim(0)
    return ax

def barplot(ax,
            table,
            p_table,
            title = "",
            x_label = "Plasmid",
            y_label = "values",
              palette = ["#003f5c",
                     "#ffa600",
                     "#bc5090"],
            x_units = ("",1),
            y_units = ("",1),
            no_bars = False,
            scaling = 0.1,
            max_v = "ci_hi",
            p_coords = pd.DataFrame(),
            level_adj = True,
            col_width = 0.7,
            **kwargs
              ):
    ftable = table.copy()
    ftable["id"] = np.arange(0,len(table))
    ftable.loc[:,["mean", "ci_lo", "ci_hi"]] = ftable.loc[:,["mean", "ci_lo", "ci_hi"]] * y_units[1]
    labels = ftable.index.to_frame(index=False).sort_index(axis = 1, ascending = True).to_numpy().T
    nitable = ftable.reset_index()
    ax.bar(nitable["id"], nitable["mean"],
          color = palette,
          yerr = np.abs(nitable.loc[:,["ci_lo", "ci_hi"]].values.T - nitable["mean"].values),
           capsize = 5,
           alpha = 0.9,
           
           width = col_width,
            **kwargs
          )
    ax = add_p_bars(ax, ftable[max_v], p_table, scaling, p_coords, level_adj)
    ax = add_multilevel_xticks(ax, labels, col_width=col_width)
    ax.set_title(title)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel(x_label if len(x_units[0])==0 else f"{x_label} [{x_units[0]}]")
    ax.set_ylabel(y_label if len(y_units[0])==0 else f"{y_label} [{y_units[0]}]")
    return ax

def add_p_points(ax, p_table, max_v, scaling,p_coords,
              level_adj):
    max_s = (max_v.reset_index()
             .drop(max_v.index.names[1], axis = 1)
             .groupby(max_v.index.names[0])
             .max()) + scaling * max_v.max()
    total_max = max_s.max()
    resh = p_table.melt().dropna().pivot_table(values = "value", 
                                                 index = p_table.index.names[0],
                                                 columns = p_table.index.names[1],
                                                 sort = False).apply(convert_pvalue_to_asterisks)
    for i, row in resh.iterrows():
        y = total_max
        if not level_adj:
            y = max_s.loc[i]
        ax.text(i, y, 
                '\n'.join(row.tolist()),
                horizontalalignment = "center")
    ax.set_ylim(top = 1.1*float(total_max))
    return ax

def lineplot(ax,
            table,
            p_table,
            title = "",
            x_label = "Plasmid",
            y_label = "values",
              palette = ["#003f5c",
                     "#ffa600",
                     "#bc5090"],
            x_units = ("",1),
            y_units = ("",1),
            no_bars = False,
            scaling = 0.1,
            max_v = "ci_hi",
            p_coords = pd.DataFrame(),
            level_adj = False,
            col_width = 0.7,
            **kwargs
              ):
    ftable = table.copy().sort_index(ascending = [True, False])
    ftable.loc[:,["mean", "ci_lo", "ci_hi"]] = ftable.loc[:,["mean", "ci_lo", "ci_hi"]] * y_units[1]
    for i, g in enumerate(ftable.index.get_level_values(-1).unique()):
        y = ftable.loc[pd.IndexSlice[:,g],"mean"].values
        x = ftable.loc[pd.IndexSlice[:,g],:].index.get_level_values(0).to_numpy()
        ci_lo = ftable.loc[pd.IndexSlice[:,g],"ci_lo"].values
        ci_hi = ftable.loc[pd.IndexSlice[:,g],"ci_hi"].values
        ax.plot(x,y, color = palette[i],
                label = g,linewidth=2, marker = "d")
        ax.errorbar(x,y,yerr=np.column_stack([y-ci_lo,ci_hi-y]).T,
                    color = palette[i],capsize=4,elinewidth=0.5,linestyle="")
    ax = add_p_points(ax, p_table, ftable["ci_hi"], scaling,p_coords,
              level_adj)
    ax.legend(loc=3)
    ax.set_title(title)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel(x_label if len(x_units[0])==0 else f"{x_label} [{x_units[0]}]")
    ax.set_ylabel(y_label if len(y_units[0])==0 else f"{y_label} [{y_units[0]}]")
    ax.set_ylim(0)
    return ax


import skimage as ski
from skimage.transform import hough_line, hough_line_peaks
from skimage.color import rgb2gray
import scipy.ndimage as ndi



class WesternBlot:
    
    def __init__(self, border = 0.1,
                 linewidth = 1) -> None:
        self.linewidth = linewidth
        self.border = border
        self.x= 12+2*border
        self.y = border
        self.images = {}
        self.lines = []
        self.minor = None
        self.lane_params = pd.DataFrame(
            columns = [
                "height",
                "y_min",
                "weight",
                "protein"
            ]
        )
        self.label_params = {}
        pass
    
    def add_lane(self, 
                 image = None,
                 protein = "???",
                 weight = 0,
                 id = None):
        if type(image) == str:
            image = ski.io.imread(image)
        if id == None:
            id = len(self.images)
        self.images[id] = image
        height = (image.shape[0]/image.shape[1]) * (self.x - 2*self.border)
        self.lane_params.loc[id, "height"] = height
        self.lane_params.loc[id, "y_min"] = self.y
        self.lane_params.loc[id, "protein"] = protein
        self.lane_params.loc[id, "weight"] = f"{str(weight)} kD"
        self.y += (height + 2*self.border)
        return self
    
    def extract_points(self, image, n):
        gray  = rgb2gray(image)
        t = gray < ski.filters.threshold_otsu(gray)
        labeled = ndi.label(t)[0]
        rp = (pd.DataFrame(data=ski.measure.regionprops_table(labeled, properties=("area","centroid")))
            .sort_values("area", ascending=False)
            .iloc[:n,:]
            .sort_values("centroid-1", ascending=True))
        return rp["centroid-1"].values/image.shape[1]
    
    def get_locations(self, n, id):
        points =self.extract_points(self.images[id], n)
        return points
        
    
    def edit_multilevel_labels(self, labels):
        major = labels[0]
        for i in range(1,labels.shape[0]):
            fill_array = np.chararray(labels[0].shape).fill("\n")
            major = (major, fill_array)
            lines_rel = [(np.min(np.argwhere(labels[i] == x)),
                          np.max(np.argwhere(labels[i] == x)),
                          i)
                         for x in np.unique(labels[i])]
            for tup in lines_rel:
                pos = np.mean(np.array([tup[0],tup[1]]))
                
        return major
    
    def add_labels(self, labels,
                   locations = None,
                   id = 0):
        if locations == None:
            self.label_loc = self.get_locations(len(labels), id)*(self.x - 2* self.border) + self.border
        else: self.label_loc = locations
        self.labels = labels
        return self
    
    def plot_lane(self, ax, id, y_min, h):
        inset = ax.inset_axes([self.border, 
                                   y_min,
                                   self.x-2*self.border,
                                   h],
                                  transform = ax.transData)
        ex_in = (0, self.x-2*self.border, 0, h)
        inset.imshow(self.images[id],
                        extent=ex_in)
        inset.tick_params(bottom = False,
                            left = False,
                            labelbottom = False,
                            labelleft = False)
        for side in ["top","bottom","left","right"]:
            inset.spines[side].set_linewidth(self.linewidth)
        return ax
    
    def plot(self, ax = None
                ) -> plt.axis:
        ex = (0, self.x, 0, self.y-self.border)
        if ax == None:
            ax = plt.gca()
        ax.imshow(255 * np.ones([100,100,3],dtype=np.uint8),
                  extent=ex, origin="lower")
        for id, lane in self.lane_params.iterrows():
            h = lane["height"]
            y_min = lane["y_min"]
            ax = self.plot_lane(ax, id, y_min, h)
        ax.tick_params(which = "major", bottom = False,
                              left = False,
                              labelbottom = False,
                              labelleft = True,
                              labeltop = True)
        ax.tick_params(which = "minor",
                       left = False,
                        labelleft = False,
                        labelright = True)
        lp = self.lane_params.copy()
        lp["label_y"] = lp["y_min"] + lp["height"]/2
        ax.set_yticks(lp["label_y"].tolist(), lp["protein"].tolist())
        ax.set_yticks((lp["label_y"] + 0.001).tolist(), lp["weight"].tolist(), minor = True)
        ax.set_xticks(self.label_loc, self.labels,
                      fontsize = "large")
        for side in ["top","bottom","left","right"]:
            ax.spines[side].set_visible(False)
        return ax


