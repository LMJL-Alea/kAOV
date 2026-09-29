from kaov import AOV
import numpy as np
import os
import pandas as pd
import random


def sequencing_data():
    """Data and metadata from experimental transcriptomic dataset."""

    # data
    url = "https://raw.githubusercontent.com/LMJL-Alea/ktest/main/tutorials/v5_data/RTqPCR_reversion_logcentered.csv"
    data = pd.read_csv(url, index_col=0)

    # metadata
    meta = pd.Series(data=pd.Series(data.index).apply(lambda x : x.split(sep='.')[1]))
    meta.index = data.index

    # sample names
    sample_names = list(meta.unique())

    # output
    return data, meta, sample_names


def dummy_data(nrow=100, ncol=100, n_factor=1, n_group=[2,]):
    """Generate dummy data for testing"""
    assert n_factor == len(n_group)
    # generate random data (under H0, no separation)
    rng = np.random.default_rng()
    data_array = rng.normal(loc=0, scale=1, size=(nrow, ncol))

    # create a data frame from random gaussian data
    data = pd.DataFrame(
        data=data_array,
        columns=[f"col{i+1}" for i in range(ncol)]
    )

    # create meta data (add factor columns)
    for i in range(n_factor):

        group_ind = list(np.sort(
            np.tile(
                np.array([f"c{i+1}" for i in range(n_group[i])]),
                (nrow+n_group[i]) // n_group[i]
            )[:nrow]
        ))

        random.shuffle(group_ind)

        data[f"factor{i+1}"] = group_ind

    # formulae
    output_formula = " + ".join([f"col{i+1}" for i in range(ncol)])
    input_formula = " + ".join([f"C(factor{i+1}, OneHot)" for i in range(n_factor)])

    # output
    return data, input_formula, output_formula


def run_kaov(data, input_formula, output_formula):
    """Create kaov object and run test"""
    # init object
    kfit_1 = AOV.from_formula(
        f"{output_formula} ~ {input_formula}", data=data, nystrom=True
    )
    # run kfda test
    res_1_pw = kfit_1.test(hypotheses='pairwise')
    # output
    return res_1_pw


if __name__ == "__main__":
    # various data dimension
    for nrow in [100, 1000]:
        for ncol in [10, 100]:
            if ncol > nrow:
                continue
            # verbosity
            print(
                f"Running kaov with dummy data of dim ({nrow}, {ncol})"
            )
            # generate data
            data, input_formula, output_formula = dummy_data(
                nrow, ncol, 2, [2, 3]
            )
            # run
            res = run_kaov(data, input_formula, output_formula)
