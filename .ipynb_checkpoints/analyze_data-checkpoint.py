import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chisquare
from statsmodels.stats.multitest import fdrcorrection
from pingouin import pairwise_tests
from itertools import product, combinations

np.random.seed(888)

def bootstrap_lo(v, ntries = 1000):
    tries = np.mean(np.random.choice(v, (len(v),ntries)), axis = 0)
    return np.percentile(tries, 2.5)

def bootstrap_hi(v, ntries = 1000):
    tries = np.mean(np.random.choice(v, (len(v),ntries)), axis = 0)
    return np.percentile(tries, 97.5)

def convert_pvalue_to_asterisks(pvalue):
    stars = np.empty(pvalue.shape, dtype = object)
    stars[pvalue >= 0] = "ns"
    stars[pvalue <= 0.05] = "*"
    stars[pvalue <= 0.01] = "**"
    stars[pvalue <= 0.001] = "***"
    stars[pvalue <= 0.0001] = "****"
    return stars


def change_last(tup, ctrl):
    l_tup = list(tup)
    l_tup[-1] = ctrl
    return tuple(l_tup)

def get_sigpairs(table, control):
    if type(list(table.index)[0]) is not tuple:
        return [(x,y) for x,y in product(control, table.index) if x != y]
    else:
        return [(change_last(x, control),x) for x in table.index if x != change_last(x, control)]
    
def get_p_continous(data, sigp, table, values):
    control_v = data.loc[[sigp[0]],values].values
    treatment_v = data.loc[[sigp[1]],values].values
    table.loc[[sigp[1]],"p"] = ttest_ind(treatment_v, control_v, 
                                         alternative = "less",
                                         nan_policy="omit").pvalue
    table.loc[[sigp[1]],"control"] = sigp[0][-1]
    return table

def get_p_categorical(data, sigp, table, values):
    c = data.loc[[sigp[0]],values].mean()
    t = data.loc[[sigp[1]],values].mean()
    table.loc[[sigp[1]],"p"] = chisquare(f_obs = [t, 1-t], f_exp = [c, 1-c], 
                                         alternative = "less",
                                         nan_policy="omit").pvalue
    table.loc[[sigp[1]],"control"] = sigp[0]
    return table


def get_summary(data, 
                groups = ["Group1"],
                values = "values",
                control = "X",
                sigpairs = None,
                continous = True):
    table = (data.groupby(groups).agg(mean = pd.NamedAgg(values, lambda d: np.mean(d)),
                                    sd = pd.NamedAgg(values, lambda d: np.std(d)),
                                    ci_lo = pd.NamedAgg(values, lambda d: bootstrap_lo(d)),
                                   ci_hi = pd.NamedAgg(values, lambda d: bootstrap_hi(d)))
             .assign(p = pd.Series(),
                     p_cor = pd.Series(),
                     control = pd.Series()))
    if sigpairs == None:
        sigpairs = get_sigpairs(table, control)
    data_ind = data.set_index(groups)
    for sigp in sigpairs:
        if continous:
            table = get_p_continous(data_ind, sigp, table, values)
        else:
            table = get_p_categorical(data_ind, sigp, table, values)
    _,p_cor = fdrcorrection(table.p.dropna())
    table.loc[table.p >=0,"p_cor"] = p_cor
    return table.assign(p_stars = lambda d: convert_pvalue_to_asterisks(d.p_cor)).sort_index(ascending = False)
    


df_gfp_f = (pd.read_csv(r'E:\diploma_thesis\Data\1. Plasmid can degrade GFP\Flourometry\Results.csv')
           .loc[:,["Group1","Mean_norm"]])

summary = get_summary(df_gfp_f, values = "Mean_norm")
print(summary)

summary.head()

test_file = r"E:\diploma_thesis\Data\5. Plasmid-mediated degradation does not get affected by Inhibition of Autophagy\Flourometry\Results.csv"
test_df = (pd.read_csv(test_file)
           .loc[:,["Mean_norm_X","Group1","Group2"]])

summary = get_summary(test_df, values = "Mean_norm_X", groups = ["Group2", "Group1"], control = "X")

summary.head()












