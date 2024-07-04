import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, fisher_exact, false_discovery_control, iqr

# from pingouin import pairwise_tests
from itertools import product, combinations

np.random.seed(888)


def convert_pvalue_to_asterisks(pvals):
    ast = np.empty(pvals.shape, dtype="U4")
    ast[pvals >= 0] = "ns"
    ast[pvals <= 0.05] = "*"
    ast[pvals <= 0.01] = "**"
    ast[pvals <= 0.001] = "***"
    ast[pvals <= 0.0001] = "****"
    return ast


class result:

    def __init__(self, df=None):
        self.values = "values"
        self.df = df
        return

    def read(self, filename, read_func=pd.read_csv, **kwargs):
        self.df = read_func(filename, **kwargs)
        return self

    def summarize(self, values=None, **kwargs):
        if values is not None:
            self.values = values
        self.summary = get_summary(self.df, values=self.values, **kwargs)
        return self.summary

    def create_p_table(self, **kwargs):
        self.p_table = get_p_table(self.df, self.summary, values=self.values, **kwargs)
        return (
            pd.melt(self.p_table, ignore_index=False, value_name="p")
            .dropna()
            .assign(star=lambda d: convert_pvalue_to_asterisks(d.p))
        )


def bootstrap_lo(v, ntries=1000):
    tries = np.mean(np.random.choice(v, (len(v), ntries)), axis=0)
    return np.percentile(tries, 2.5)


def bootstrap_hi(v, ntries=1000):
    tries = np.mean(np.random.choice(v, (len(v), ntries)), axis=0)
    return np.percentile(tries, 97.5)


def pv_to_star(pvalue):
    stars = np.empty(pvalue.shape, dtype=object)
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
        return [(x, y) for x, y in product(control, table.index) if x != y]
    else:
        return [
            (change_last(x, control), x)
            for x in table.index
            if x != change_last(x, control)
        ]


def get_p(pair, data, categorical, alternative, ci):
    control_v = data[pair[0]].values
    treatment_v = data.loc[pair[1]].values
    if categorical:
        table = np.array(
            [[treatment_v.sum(), control_v.sum()], [len(treatment_v), len(control_v)]]
        )
        return fisher_exact(table=table, alternative=alternative).pvalue
    if ci:
        t = np.mean(np.random.choice(treatment_v, (len(treatment_v), 10000)), axis=0)
        c = np.mean(np.random.choice(control_v, (len(control_v), 10000)), axis=0)
        if alternative == "less":
            return np.mean(c - t < 0)
        if alternative == "greater":
            return np.mean(t - c < 0)
        else:
            return np.mean(t - c < 0) + np.mean(c - t < 0)

    else:
        return ttest_ind(
            treatment_v, control_v, alternative=alternative, nan_policy="omit"
        ).pvalue


def correct_p_values(p_table):
    return p_table.combine(
        p_table.stack(p_table.index.names)
        .transform(lambda x: false_discovery_control(x))
        .unstack(list(np.arange(len(p_table.index.names))))
        .T,
        lambda x, y: y,
    )


def get_p_table(
    data,
    table,
    values="values",
    sigpairs=None,
    control="X",
    categorical=False,
    ci=False,
    alternative="less",
):
    series = data.set_index(table.index.names).sort_index(ascending=False)[values]
    if sigpairs == None:
        sigpairs = get_sigpairs(table, control)
    p_table = pd.DataFrame(columns=table.index, index=table.index, dtype="float64")
    for pair in sigpairs:
        p_table.loc[pair] = get_p(pair, series, categorical, alternative, ci)
    p_table_corr = correct_p_values(p_table)
    return p_table_corr.sort_index(ascending=False, axis=0).sort_index(
        ascending=False, axis=1
    )


def get_summary(data, groups=["Group1"], values="values"):
    table = data.groupby(groups).agg(
        mean=pd.NamedAgg(values, lambda d: np.mean(d)),
        sd=pd.NamedAgg(values, lambda d: np.std(d)),
        ci_lo=pd.NamedAgg(values, lambda d: bootstrap_lo(d)),
        ci_hi=pd.NamedAgg(values, lambda d: bootstrap_hi(d)),
        count=pd.NamedAgg(values, lambda d: len(d)),
    )
    return table.sort_index(ascending=False)
