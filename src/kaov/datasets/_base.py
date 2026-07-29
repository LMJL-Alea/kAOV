#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 09:42:04 2026

@author: Polina Arsenteva
"""
from importlib import resources
import pandas as pd
import scanpy as sc

def load_reversion():
    """
    Load the reversion dataset from a .csv file.

    Returns
    -------
    data : pandas.DataFrame
        The dataset containing single cell gene expression measurements, with
        cells in rows and genes in columns, along with the metadata as last columns.

    """
    data_path = resources.files("kaov.datasets.data") / "reversion.csv"
    data = pd.read_csv(data_path, index_col=0)
    return data

def load_rabbits_anndata():
    """
    Load the rabbits dataset from a .h5ad file.

    Returns
    -------
    data : AnnData
        The single cell transcriptomics dataset in the AnnData format containing
        raw and pre-processed counts, metadata, UMAP, etc.

    """
    data_path = resources.files("kaov.datasets.data") / "rabbits_ct.h5ad"
    data = pd.read_csv(data_path, index_col=0)
    return data

