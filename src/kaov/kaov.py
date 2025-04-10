#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:28:15 2024

@author: Polina Arsenteva
"""
import itertools
import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
from statsmodels.iolib import summary2
from patsy import dmatrices, DesignInfo, ContrastMatrix
from scipy.stats import chi2, gaussian_kde
import matplotlib.pyplot as plt
from matplotlib import rc, colormaps
from torch import cdist, exp, matmul, diag, trace, sqrt, pow, cat
from torch import eye, ones, tensor, float64, from_numpy, Tensor
from torch.linalg import multi_dot
from apt.eigen_wrapper import eigsy

def ordered_eigsy(matrix):
    """
    Calculates the eigendecomposition of the matrix, using a solver 
    implemented in C++.

    Parameters
    ----------
    matrix : 2-d array_like
        Matrix to decompose.

    Returns
    -------
    sp : torch.Tensor
        Eigenvalues.
    ev : torch.Tensor
        Eigenvectors.
    """
    sp,ev = eigsy(matrix)
    order = sp.argsort()[::-1]
    ev = tensor(ev[:, order], dtype=float64) 
    sp = tensor(sp[order], dtype=float64)
    return(sp, ev)

def convert_to_torch(A):
    """
    Converts A to torch.Tensor.

    Parameters
    ----------
    A : array_like
        Container to convert.

    Returns
    -------
    B : torch.Tensor
        A converted to torch.

    """
    if isinstance(A, pd.Series):
        B = from_numpy(A.to_numpy().reshape(-1,1)).double()
    if isinstance(A, pd.DataFrame):
        B = from_numpy(A.to_numpy()).double()
    if isinstance(A, Tensor):
        B = A.double()
    else:
        try:
            X = A.to_numpy() if not isinstance(A, np.ndarray) else A.copy()
            B = from_numpy(X).double()
        except AttributeError:
            print(f'Unknown data type {type(A)}')
    return B

def distances(x, y=None):
    """
    Computes the distances between each pair of the two collections of row 
    vectors of x and y, or x and itself if y is not provided.
    
    Parameters
    ----------
    x : torch.Tensor
        Input 2-d tensor.
    y : None or torch.Tensor
        Input 2-d tensor. Replaced by `x` if None.

    Returns
    -------
    sq_dists : torch.Tensor
        Distance matrix.
        
    """
    if y is None:
        y = x.clone()
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("2-dimensional input is expected.")
    if x.shape[1] != y.shape[1]:
        raise ValueError("`x` and `y` must have the same second dimension.")
    sq_dists = cdist(x, y, 
                     compute_mode='use_mm_for_euclid_dist_if_necessary').pow(2)  
    return sq_dists

def _median(x, y=None):    
    if y == None:
        dtot = distances(x)
    else:
        dtot = distances(cat((x,y)))
    median = dtot.median()
    if median == 0: 
        warnings.warn('The median is null. To avoid a kernel with zero bandwidth, the median is replaced by the mean.')
        mean = dtot.mean()
        if mean == 0 : 
            warnings.warn('The whole dataset is null.')
        return mean
    else:
        return median

def linear_kernel(x, y=None):
    """
    Computes the standard linear kernel k(x,y)= <x,y>.

    Parameters
    ----------
    x : torch.Tensor
        2-d tensor containing the data to kernalize.
    y : None or torch.Tensor
        2-d tensor containing the data to kernalize. Replaced by `x` if None.

    Returns
    -------
    K : torch.Tensor
        Kernel matrix (gram).
        
    """
    if y is None:
        y = x.clone()
    K = matmul(x, y.T)
    return K

def gauss_kernel(x, y=None, sigma=1):
    """
    Computes the standard Gaussian kernel k(x,y)=exp(- ||x-y||**2 / (2 * sigma**2)).

    Parameters
    ----------
    x : torch.Tensor
        2-d tensor containing the data to kernalize.
    y : None or torch.Tensor
        2-d tensor containing the data to kernalize. Replaced by `x` if None.
    sigma : int
        Standard deviation for Gaussian kernel, 1 by default.

    Returns
    -------
    K : torch.Tensor
        Kernel matrix (gram).

    """
    d = distances(x, y)   # [sq_dists]_ij=||X_j - Y_i \\^2
    K = exp(-d / (2 * sigma**2))  # Gram matrix
    return K

def gauss_kernel_median(x, y=None, bandwidth='median', median_coef=1, 
                        return_bandwidth=False):
    """
    Computes the gaussian kernel with bandwidth set as the median of the 
    distances between pairs of observations (`bandwidth='median'`).

    Parameters
    ----------
    x : torch.Tensor
        2-d tensor containing the data to kernalize.
    y : None or torch.Tensor
        2-d tensor containing the data to kernalize. Replaced by `x` if None.
    bandwidth : 'median' or float
        If 'median' (default), the bandwidth is calculated with the median 
        method. If float, the value is assigned to the bandwidth.
    median_coef : float
        Coefficient in the badwidth calculation, 1 by default.
    return_bandwidth : bool
        Bandwidth, calculated with the median method. False by default.

    Returns
    -------
    kernel : callable
        Kernel function.
    if return_bandwidth=True:
    computed_bandwidth : float
        Bandwidth, calculated with the median method.
    """
    if bandwidth == 'median':
        computed_bandwidth = sqrt(_median(x, y) * median_coef)
    else:
        computed_bandwidth = bandwidth
    kernel = lambda a, b=None : gauss_kernel(x=a, y=b, sigma=computed_bandwidth)
    if return_bandwidth:
        return (kernel, computed_bandwidth)
    else: 
        return kernel
    
def _calculate_XXinv_and_ProjImX(X):
    XX = matmul(X.T, X)
    sp, ev = ordered_eigsy(XX)
    # Cut off the spectrum since the matrix is not full rank:
    cutoff = np.linalg.matrix_rank(X)
    sp = sp[: cutoff]
    ev = ev[:, : cutoff]
    _XXinv = multi_dot([ev, diag(sp ** -1), ev.T])
    _ProjImX = multi_dot([X, _XXinv, X.T])
    return _XXinv, _ProjImX

class OneHot(object):
    """
    Class defining One Hot encoding: a coding scheme for linear models, along 
    with other well known ones such as Treatment or Difference coding.It is to 
    be integrated in a formula defining the linear model, provided by patsy's 
    formula interface. It is recommended to use one hot encoding with the 
    testing framework implemented in AOV, especially with more than one factor.
    
    Example of a formula with OneHot:
        'y1 + y2 ~ C(x1, OneHot) + C(x2, OneHot)'
    See Readme and tutorials for kAOV for more details and examples.
    
    """
    def __init__(self, reference=-1):
        self.reference = reference

    def code_with_intercept(self, levels):
        return ContrastMatrix(np.eye(len(levels)),
                              ["[%s]" % (level,) for level in levels])
    def code_without_intercept(self, levels):
        return self.code_with_intercept(levels)
    
class Data:
    """
    Class containing data-related structures for the kernel analysis of variance.
    
    Parameters
    ----------
    endog : 2-d array_like
        An array_like with dimensions nobs x nvar containing nobs values of 
        nvar dependent variables.
    exog : 2-d array_like
        An array_like with dimensions nobs x nlvl containing nobs values of 
        nlvl independent variables.
    meta : None or 2-d array_like, optional
        An array_like with the metadata for the dataset, i.e. containing
        information on factors. Used for visualizations.
    endog_names : None or 1-d array_like, optional
        A 1-dimensional array_like containing names of the dependent variables.
        If not specified (default), will be retrieved from `endog` or assigned 
        to numbers with respect to the order.
    exog_names : None or 1-d array_like, optional
        A 1-dimensional array_like containing names of the independent variables.
        If not specified (default), will be retrieved from `exog` or assigned 
        to numbers with respect to the order.
    nystrom : bool, optional
        If True, computes the Nystrom landmarks, in which case the observations
        in all attributes correspond to the landmarks and not the original data.
        The default is False.
    n_landmarks: int, optional
        Number of landmarks used in the Nystrom method. If unspecified, one
        fifth of the observations are selected as landmarks.
    random_gen :  int, Generator, RandomState instance or None
        Determines random number generation for the landmarks selection. 
        If None (default), the generator is the RandomState instance used 
        by `np.random`. To ensure the results are reproducible, pass an int
        to instanciate the seed, or a Generator/RandomState instance (recommended).

    Attributes:
    ----------
    endog : 2-d torch.tensor
        A tensor with dimensions nobs x nvar containing nobs values of 
        nvar dependent variables.
    exog : 2-d torch.tensor
        A tensor with dimensions nobs x nlvl containing nobs values of 
        nlvl independent variables.
    meta : None or 2-d array_like
        An array_like with the metadata for the dataset, i.e. containing
        information on factors. Used for visualizations. The default is None.
    nobs : int
        Number of observations. If `nystrom=True`, corresponds to the number 
        of landmarks `n_landmarks`.
    nlvl : int
        Number of independent variables (i.e. levels of all factors).
    nvar : int
        Number of dependent variables.
    index : 1-d array_like
        Observation labels.
    endog_names : 1-d array_like
        A 1-dimensional array_like containing names of the dependent variables.
    exog_names : 1-d array_like
        A 1-dimensional array_like containing names of the independent variables.
    nystrom : bool
        False by default, True if the Nystrom approximation is performed, in 
        which case the observations in all attributes correspond to the 
        landmarks and not the original data.

    """
    def __init__(self, endog, exog, meta=None, endog_names=None, exog_names=None,
                 nystrom=False, n_landmarks=None, random_gen=None):
        self.exog = convert_to_torch(exog)
        self.endog = convert_to_torch(endog)
        self.meta = meta
        self.index = exog.index if hasattr(exog, "index") else range(self.nobs)
        self.nobs = self.exog.shape[0]
        self.nlvl = self.exog.shape[1]
        self.nvar = self.endog.shape[1]
        if endog_names is not None:
            self.endog_names = endog_names
        elif hasattr(endog, "columns"):
            self.endog_names = endog.columns
        else:
            self.endog_names = ['y' + str(x) for x in range(self.nvar)]
            
        if exog_names is not None:
            self.exog_names = exog_names
        elif hasattr(exog, "columns"):
            self.exog_names = exog.columns
        else:
            if hasattr(self, '_factor_info') and 'Intercept' in self._factor_info:
                self.exog_names = ['x' + str(x + 1) for x in range(self.nlvl - 1)]
                self.exog_names.insert(0, 'Intercept')
            else:
                self.exog_names = ['x' + str(x + 1) for x in range(self.nlvl)]
        
        if self.exog.shape[1] != len(self.exog_names):
            raise ValueError("The length of `exog_names` should be equal to "\
                             "the number of columns in `exog`.")
        if self.endog.shape[1] != len(self.endog_names):
            raise ValueError("The length of `endog_names` should be equal to "\
                             "the number of columns in `endog`.")  
        
        # Calculate useful matrices:
        self._XXinv, self._ProjImX = _calculate_XXinv_and_ProjImX(self.exog)
        
        self.nystrom = nystrom
        if self.nystrom:
            self.nobs = (n_landmarks if n_landmarks is not None 
                         else min(self.nobs, self.nlvl * 30))
            generators = (np.random.RandomState, np.random.Generator)
            if isinstance(random_gen, generators):
                rnd_gen = random_gen
            elif isinstance(random_gen, int):
                rnd_gen = np.random.default_rng(random_gen)
            else:
                rnd_gen = np.random
            h_ii = diag(self._ProjImX)
            ny_ind = rnd_gen.choice(np.arange(self.exog.shape[0]), size=self.nobs,
                                    p=h_ii/h_ii.sum(), replace=False)
            ny_ind.sort()
            self.exog = self.exog[ny_ind]
            self.endog = self.endog[ny_ind]
            if isinstance(self.meta, (pd.Series, pd.DataFrame)):
                self.meta = self.meta.iloc[ny_ind]
            else:
                self.meta = self.meta[ny_ind]
            self.index = self.index[ny_ind]
            self._XXinv, self._ProjImX = _calculate_XXinv_and_ProjImX(self.exog)        
        self._ProjImXorthogonal = eye(self.nobs) - self._ProjImX
        
    def _diagonalize_residual_covariance(self, K):
        """
        Computes the spectral decomposition of a matrix that shares the 
        spectrum with the residual covariance operator. A normalized version of
        the eigenvectors is stored since it is used in all calculations.
        
        """
        Kresidual = 1 / self.nobs * multi_dot([self._ProjImXorthogonal,
                                               K, self._ProjImXorthogonal])
        sp, ev = ordered_eigsy(Kresidual)
        sp_power = -1/2 if self.nystrom else -1
        sp12 = sp ** sp_power * self.nobs ** (-1/2)
        to_ignore = sp12.isnan()
        sp12 = sp12[~to_ignore]
        U = sp12 * ev[:,~to_ignore]
        U_norm = matmul(self._ProjImXorthogonal, U)
        return sp, U_norm
        
class AOV:
    """
    Class implementing Kernel Analysis Of Variance.
    
    Parameters
    ----------
    endog : 2-d array_like
        An array_like with dimensions nobs x nvar containing nobs values of 
        nvar dependent variables.
    exog : 2-d array_like
        An array_like with dimensions nobs x nlvl containing nobs values of 
        nlvl independent variables.
    meta : None or 2-d array_like, optional
        An array_like with the metadata for the dataset, i.e. containing
        information on factors. Used for visualizations.
    endog_names : None or 1-d array_like, optional
        A 1-dimensional array_like containing names of the dependent variables.
        If not specified (default), will be retrieved from `endog` or assigned 
        to numbers with respect to the order.
    exog_names : None or 1-d array_like, optional
        A 1-dimensional array_like containing names of the independent variables.
        If not specified (default), will be retrieved from `exog` or assigned 
        to numbers with respect to the order.
    kernel_function : callable or str, optional
        Specifies the kernel function. Acceptable values in the form of a
        string are 'gauss' (default) and 'linear'. Pass a callable for a
        user-defined kernel function.
    kernel_bandwidth : 'median' or float, optional
        Value of the bandwidth for kernels using a bandwidth. If 'median' 
        (default), the bandwidth will be set as the median or its multiple, 
        depending on the value of the parameter `median_coef`. Pass a float
        for a user-defined value of the bandwidth.
    kernel_median_coef : float, optional
        Multiple of the median to compute bandwidth if `kernel_bandwidth='median'`.
        The default is 1. 
    nystrom : bool, optional
        If True, computes the Nystrom landmarks, in which case the observations
        in all attributes correspond to the landmarks and not the original data.
        The default is False.
    n_landmarks: int, optional
        Number of landmarks used in the Nystrom method. If unspecified, one
        fifth of the observations are selected as landmarks.
    random_gen :  int, Generator, RandomState instance or None
        Determines random number generation for the landmarks selection. 
        If None (default), the generator is the RandomState instance used 
        by `np.random`. To ensure the results are reproducible, pass an int
        to instanciate the seed, or a Generator/RandomState instance (recommended).

    Attributes:
    ----------
    data : instance of class Data
        Contains various information on the original dataset, see the 
        documentation of the class Data for more details.
    data_nystrom : None or instance of class Data
        Contains various information on the Nystrom dataset, see the 
        documentation of the class Data for more details. If None, Nystrom is
        not taken into account in the computations.
    kernel_function : callable or str, optional
        Specifies the kernel function.
    kernel_bandwidth : 'median' or float, optional
        Value of the bandwidth for kernels using a bandwidth.
    kernel_median_coef : float, optional
        Multiple of the median to compute bandwidth if `kernel_bandwidth='median'`.
        The default is 1. 
    kernel: callable
        Kernel function used for calculations.
    computed_bandwidth : float
        The value of the kernel bandwidth.
        
    Notes
    -----
    The `from_formula` interface is the recommended method to specify
    a model.

    """
    def __init__(self, endog, exog, meta=None, endog_names=None, exog_names=None,
                 nystrom=False, n_landmarks=None, random_gen=None,
                 kernel_function='gauss', kernel_bandwidth='median',
                 kernel_median_coef=1):
        self.data = Data(endog, exog, meta=meta, endog_names=endog_names, 
                         exog_names=exog_names)
        ### Nystrom:
        self.data_nystrom = None
        if nystrom:
            self.data_nystrom = Data(endog, exog, meta=meta, endog_names=endog_names, 
                                     exog_names=exog_names, nystrom=True, 
                                     n_landmarks=n_landmarks, random_gen=random_gen)
        
        ### Kernel:
        self.kernel_function = kernel_function
        self.kernel_bandwidth = kernel_bandwidth
        self.kernel_median_coef = kernel_median_coef
        
        if self.kernel_function == 'gauss':
            (self.kernel,
             self.computed_bandwidth) = gauss_kernel_median(x=self.data.endog,
                                                            bandwidth=kernel_bandwidth,  
                                                            median_coef=kernel_median_coef,
                                                            return_bandwidth=True)
        elif self.kernel_function == 'linear':
            self.kernel = linear_kernel
        else:
            self.kernel = self.kernel_function
        
        # Diagnostics:
        self.diagnostics = None
        
    @classmethod
    def from_formula(cls, formula, data, kernel_function='gauss', 
                     nystrom=False, n_landmarks=None, random_gen=None,
                     kernel_bandwidth='median', kernel_median_coef=1):
        """
        Creates a kernel linear model from a formula and a dataframe.

        Parameters
        ----------
        formula : str
            The formula specifying the model. For more details on the formula
            interface see https://patsy.readthedocs.io/en/latest/formulas.html.
        data : pandas.DataFrame
            The data for the model. Columns must contain the values for the 
            factors in the formula, with the names matching those in the formula.
        kernel_function : callable or str, optional
            Specifies the kernel function. Acceptable values in the form of a
            string are 'gauss' (default) and 'linear'. Pass a callable for a
            user-defined kernel function.
        kernel_bandwidth : 'median' or float, optional
            Value of the bandwidth for kernels using a bandwidth. If 'median' 
            (default), the bandwidth will be set as the median or its multiple, 
            depending on the value of the parameter `median_coef`. Pass a float
            for a user-defined value of the bandwidth.
        kernel_median_coef : float, optional
            Multiple of the median to compute bandwidth if `kernel_bandwidth='median'`.
            The default is 1.

        Returns
        -------
        aov_obj : instance of AOV
        
        """
        if 'OneHot' in formula:
            formula += ' - 1'
        endog, exog = dmatrices(formula, data, return_type='dataframe', 
                                NA_action='raise')
        # Extract the metadata:
        meta = data[data.columns.difference(endog.columns)]
    
        # Simplify the names for OneHot:
        if 'OneHot' in formula:
            s1 = 'C('
            s2 = ', OneHot)'
            exog.columns = [col.replace(s1, '').replace(s2, '') for col in exog.columns]
            
        aov_obj = cls(endog, exog, meta=meta, nystrom=nystrom, 
                      n_landmarks=n_landmarks, random_gen=random_gen,
                      kernel_function=kernel_function, 
                      kernel_bandwidth=kernel_bandwidth,
                      kernel_median_coef=kernel_median_coef)
        aov_obj.formula = formula
        aov_obj._factor_info = exog.design_info.term_name_slices
        # Simplify the names for OneHot:
        if 'OneHot' in formula:
            aov_obj._factor_info = {_name.replace(s1, '').replace(s2, '') : _slice
                                       for _name, _slice in aov_obj._factor_info.items()}
        return aov_obj
    
    def compute_diagnostics(self, t_max=100):
        """
        Calculates diagnostics associated with the model and saves them in the
        `diagnostics` attribute. The latter is a dictionary containing the 
        following quantities:
        - Embeddings: projections of the embeddings on the first 
        eigenfunctions of the residual covariance operator.
        - Predictions: projections of the predictions of the embeddings on 
        the first eigenfunctions of the residual covariance operator.
        - Residuals: projections of the residuals on the first
        eigenfunctions of the residual covariance operator.

        Parameters
        ----------
        t_max : int, optional
            Maximal truncation for projections calculation, the default is 100.

        """
        K = self.kernel(self.data.endog)
        sp, U_norm = self.data._diagonalize_residual_covariance(K)
        t_max = min(t_max, self.data.nobs)
        U_norm_T = U_norm[:, : t_max]
        
        columns = list(range(1, t_max + 1))
        U_norm_p12 = sp[:t_max] ** (1/2) * U_norm_T
        embeddings = pd.DataFrame(multi_dot([K, U_norm_p12]), 
                                  index=self.data.index, columns=columns)
        predictions = pd.DataFrame(multi_dot([self.data._ProjImX, K, U_norm_p12]),
                                   index=self.data.index, columns=columns)
        residuals = pd.DataFrame(multi_dot([self.data._ProjImXorthogonal, K, U_norm_p12]),
                                 index=self.data.index, columns=columns)
        self.diagnostics = {'Embeddings' : embeddings,
                            'Predictions' : predictions,
                            'Residuals' : residuals}
    
    def plot_diagnostics(self, t=100, diagnostic='residuals', t_max=100, 
                         colormap='viridis', alpha=.75, legend_fontsize=12, 
                         font_family='serif', figsize=None):
        """
        Plots diagnostics associated with the model. The graph contains sublots
        for each factor, with the projections of either the residuals 
        (`diagnostic='residuals'`) or the embeddings (`diagnostic='embeddings'`)
        on the first eigenfunctions of the residual covariance operator,
        plotted against the projections of the predictions on these eigenfunctions.

        Parameters
        ----------
        t : int, optional
            Axis to plot, by default equal to the maximal truncation.
        diagnostic : str, optional
            The type of diagnostic to plot against the predictions. 
            The default is 'residuals', the alternative is 'embeddings'.
        t_max : int, optional
            Maximal truncation for projections calculation, the default is 100.
        colormap : str, optional
            The name of a matplotlib colormap to be used for different factor
            levels. The default is 'viridis'.
        alpha : float, optional
           The alpha blending value, between 0 (transparent) and 1 (opaque).
           The default is 0.5.
        legend_fontsize : int, optional
            Legend font size. The default is 15.
        font_family : str, optional
             Legend and labels' font family name accepted by matplotlib 
             (e.g., 'serif', 'sans-serif', 'monospace', 'fantasy' or 'cursive'),
             the default is 'serif'.
        figsize : tuple, optional
            The size of the figure. If not specified, is set to
            (8 * nb_factors, 6).
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            A Figure object of the plot.
        axs : numpy.ndarray of matplotlib.axes._axes.Axes
            An Axes object of the plot.

        """
        if not self.diagnostics or t not in self.diagnostics['Predictions']:
            self.compute_diagnostics(t_max=max(t, t_max))
            
        factors = self.data.meta.columns
        nb_factors = len(factors)
        T_max = len(self.diagnostics['Predictions'].columns)
        t = min(t, T_max)
        pred = self.diagnostics['Predictions']
        a, b = pred[t].min(), pred[t].max()
        x = np.arange(a - (b - a) / 10, b + (b - a) / 10, (b - a) / 12)
        
        combs = list(itertools.product(*[self.data.meta[c].unique() for c in self.data.meta.columns]))
        
        if diagnostic == 'residuals':
            diagn = self.diagnostics['Residuals']
            figtitle = 'Residual plot'
            y = [0] * len(x)
        elif diagnostic == 'embeddings':
            diagn = self.diagnostics['Embeddings']
            figtitle = 'Response plot'
            y = x.copy()
        figtitle += f' (t={t})'
        rc('font',**{'family': font_family})
        figsize = figsize if figsize is not None else (8 * nb_factors, 6)
        fig, axs = plt.subplots(figsize=figsize, ncols=nb_factors)
        medianprops = dict(linewidth=0.75, color='black', alpha=alpha)
        whiskerprops = dict(linewidth=0.75, alpha=alpha)
        for f, factor in enumerate(factors):
            factor_lvls = self.data.meta[factor].unique()
            cmap = colormaps[colormap]
            colors = cmap(np.linspace(0.1, 0.9, len(factor_lvls)))
            ax = axs if nb_factors == 1 else axs[f]
            ax.plot(x, y, color='black', lw=.8, linestyle='--')
            ax.set_xlim(a - (b - a) / 10, b + (b - a) / 10)
            for c in combs:
                c_obs = (self.data.meta == c).all(axis=1)
                if c_obs.sum() > 0:
                    color_ind = np.intersect1d(factor_lvls, c, return_indices=True)[1]
                    bp = ax.boxplot(diagn[t][c_obs], positions=pred.round(5)[t][c_obs].unique(),
                                    widths=(b - a) / len(combs), patch_artist=True,
                                    manage_ticks=False, medianprops=medianprops,
                                    whiskerprops=whiskerprops)
                    bp['boxes'][0].set_alpha(alpha)
                    bp['boxes'][0].set_facecolor(colors[color_ind])
            markers = [plt.Line2D([0,0],[0,0], color=color, marker='o', linestyle='') 
                       for color in colors]
            ax.legend(markers, factor_lvls, numpoints=1, fontsize=legend_fontsize,
                      bbox_to_anchor=(1.01, 0.5), loc='center left')
            ax.set_title(factor, fontsize=18)
            ax.set_xlabel('Predictions', fontsize=14)
            ax.set_ylabel('Residuals', fontsize=14)
        plt.tight_layout()
        fig.suptitle(figtitle, fontsize=25, y=1.05)
        plt.show()
        return fig, axs
        
    def set_hypotheses(self, hypotheses='pairwise', by_level=False, 
                       test_intercept=False, true_proportions=False):
        """
        Set hypotheses to be tested.

        Parameters
        ----------
        hypotheses : str or None or list[tuple]
            Hypotheses to be tested.
            - if str: either 'pairwise' (default) or 'one-vs-all'. Recommended 
            options in combination with OneHot encoding. If 'pairwise', the 
            levels of a factor are compared one to another in the pairwise way.
            If 'one-vs-all', each level is compared to the factor mean.
            - if None: produces an identity contrast matrix for each factor.
            Intended for other coding schemes (e.g. Treatment, Sum, etc).
            - if list[tuple]: custom hypothesis option. Each element of the 
            list should be a tuple of size 2: `(name, contrast_L)`, where `name`
            is a string and contrast_L is a contrast matrix in the form of a
            torch.tensor (`dtype=torch.float64`).
        by_level : bool, optional
            If False (default), computes the global test. If True, computes the 
            test by level or by a pair of levels.
        test_intercept : bool, optional
            If True, adds a test for the intercept, which is set as the 
            grand mean (or the actual mean if `true_proportions=True`) of all the
            level effects. The default is False. It is unnecessary to add an 
            intercept test manually eith this option if an intercept is present
            in the design matrix.
        true_proportions : bool, optional
            Relevant for the calculation of the factor mean, i.e. if 
            `hypotheses='one-vs-all'` or `test_intercept=True`. If False (default),
            the factor mean is the grand mean (mean of means). If True, the
            true level proportions are taken into account, so the factor mean 
            is the actual global mean of the factor.

        Returns
        -------
        hypotheses : list[tuple]
            List of hypotheses to be tested. Each element of the list is a 
            tuple of size 2: `(name, contrast_L)`, where `name` is a string and 
            contrast_L is a contrast matrix in the form of a torch.tensor.

        """
        if isinstance(hypotheses, str):
            if not hasattr(self, 'formula') or 'OneHot' not in self.formula:
                warnings.warn("'pairwise' and 'one-vs-all' options for `hypotheses` "
                              "are intended for OneHot encoding. Otherwise, "
                              "the test might be difficult to interpret.")
            hypotheses_type = hypotheses
            hypotheses = []
            if not hasattr(self, '_factor_info'):
                warnings.warn("The absence of `_factor_info` attribute was "
                              "caused by not using the `from_formula` interface. "
                              "With multiple factors, using `from_formula` "
                              "is highly recommended.")
                self._factor_info = {'Factor' : slice(0, None, None)}
            for _name, _slice in self._factor_info.items():
                lvls = self.data.exog_names[_slice]
                DI = DesignInfo(self.data.exog_names)
                if hypotheses_type == 'one-vs-all' or (test_intercept and _name != 'Intercept'):
                    if true_proportions:
                        proportions = (np.array(self.data.exog.sum(axis=0, 
                                                              dtype=int)[_slice])
                                       .astype(str))
                        lvls_p =  [proportions[i] + ' * ' + lvls[i] for i in range(len(lvls))]
                        factor_mean = '(' + ' + '.join(lvls_p) + f') / {self.data.nobs}'
                    else:
                        factor_mean = '(' + ' + '.join(lvls) + f') / {len(lvls)}'
                if not by_level:
                    if hypotheses_type == 'pairwise':
                        formula = ' = '.join(lvls)
                    elif hypotheses_type == 'one-vs-all':
                        formula = [f + ' = ' + factor_mean for f in lvls[:-1]]
                    L = DI.linear_constraint(formula).coefs
                    hypotheses.append([_name, convert_to_torch(L)])
                else:
                    if hypotheses_type == 'pairwise':
                        combs = list(itertools.combinations(lvls, 2))
                        formula = [' = '.join(comb) for comb in combs]
                        hyp_name = formula
                    elif hypotheses_type == 'one-vs-all':
                        formula = [l + ' = ' + factor_mean for l in lvls]
                        hyp_name = [l + ' = ' + _name + ' Grand Mean' for l in lvls]
                    L = DI.linear_constraint(formula).coefs
                    hypotheses.extend(list(zip(hyp_name, convert_to_torch(L))))
                if test_intercept and _name != 'Intercept':
                    if len(self._factor_info.items()) == 1:
                        L = DI.linear_constraint(factor_mean).coefs
                        hypotheses.insert(0, ['Intercept', convert_to_torch(L)])
                    else:
                        raise ValueError("`test_intercept=True` is only "
                                         "accepted in the 1-factor setting.")
        elif hypotheses is None:
            if not by_level:
                if not hasattr(self, '_factor_info'):
                    warnings.warn("If `hypotheses` is not specified, "
                                  "and `from_formula` interface not used to define the model, "
                                  "all `exog_names` are considered as levels of one factor.")
                    self._factor_info = {'Factor' : slice(0, None, None)}
                hypotheses = []
                for _name, _slice in self._factor_info.items():
                    L = eye(self.data.nlvl, dtype=float64)[_slice, :]
                    if 'Intercept' not in self._factor_info:
                        L = L[1:]
                    hypotheses.append([_name, L])
                if test_intercept and 'Intercept' not in self._factor_info:
                    raise ValueError("Incompatible combination: "
                                     "`test_intercept=True` and `hypotheses=None`.")
            else:
                hypotheses = list(zip(self.data.exog_names, eye(self.data.nlvl, dtype=float64)))
                if test_intercept:
                    raise ValueError("Incompatible combination: "
                                     "`test_intercept=True` and `hypotheses=None`.")
        return hypotheses

    def _compute_LXXLinv(self, L):
        """
        Intermediate quantity used in statistics calculations, based on a 
        transformation of the design matrix X and the contrast matrix L.
        
        """
        LXXL = multi_dot([L, self.data._XXinv, L.T])
        LXXLinv = tensor([[LXXL**-1]]) if len(LXXL.shape) == 0 else LXXL.inverse()
        return LXXLinv
    
    def _compute_D(self, L):
        """
        Intermediate quantity used in statistics calculations, based on a 
        transformation of the design matrix X and the contrast matrix L.
        
        """
        LXXLinv = self._compute_LXXLinv(L)
        A = multi_dot([L.T, LXXLinv, L])
        D = multi_dot([self.data.exog, self.data._XXinv, A, self.data._XXinv, self.data.exog.T])
        return D       
    
    def _diagonalize_covariance_of_anchor_projected_residuals(self, anchors):
        """
        Computes the kernel trick version of the covariance operator associated
        with the residuals projected onto the Nystrom anchors. Returns its
        eigendecomposition.

        """
        K_YZ = self.kernel(self.data.endog, self.data_nystrom.endog)
        K_anchor_projected = multi_dot([anchors.T, K_YZ.T,
                                        self.data._ProjImXorthogonal,
                                        K_YZ, anchors])
        K_anchor_projected *= (1/self.data.nobs)
        return ordered_eigsy(K_anchor_projected)
    
    def _compute_K_T(self, t_max=100, n_anchors=None):
        """
        Intermediate quantity used in statistics calculations, based on a 
        transformation of the gram matrix K.
        
        """
        data = self.data_nystrom if self.data_nystrom is not None else self.data
        K = self.kernel(data.endog)
        _, U_norm = data._diagonalize_residual_covariance(K)
        if self.data_nystrom is not None:
            norm_anchors = U_norm[:, : n_anchors]
            sp_a, U_a = self._diagonalize_covariance_of_anchor_projected_residuals(norm_anchors)
            U_norm = multi_dot([norm_anchors, (sp_a ** (-1/2) * U_a)])
            K = self.kernel(self.data_nystrom.endog, self.data.endog)
        t_max = min(t_max, self.data.nobs)
        U_norm_T = U_norm[:, : t_max]
        K_T = multi_dot([U_norm_T.T, K])
        return K_T

    def _compute_kernel_HL_test_statistic(self, K_T, D, d, t_max=100):
        """
        Computes the truncated kernel Hotelling-Lawley test statistic based on
        some intermediate quantities.

        """
        K_T_D_K_T = multi_dot([K_T, D, K_T.T])
        stats = []
        pvals = []
        for t in range(1,t_max):
            stat = trace(K_T_D_K_T[:t,:t]).item()
            stats += [stat]
            pvals += [chi2.sf(stat,t * d)]
        return stats, pvals
    
    def project_on_discriminant(self, K, K_T, D, center=True):
        """
        Computes of embeddings of each observation on the discriminant axes 
        associated with the test. The latter are correspond to the 
        eigenfunctions of the test statistic operator.

        Parameters
        ----------
        K : torch.tensor
            The gram matrix.
        K_T : torch.tensor
            A transformed gram matrix, obtained with `_compute_K_T`.
        D : torch.tensor
            A quantity containing the information on the contrast matrix,
            obtained with `_compute_D`.
        center : bool, optional
            If True (default), the projections are centered with respect to
            the factor mean.

        Returns
        -------
        proj : pandas.DataFrame 
            Contains projection values, with rows corresponding to
            observations, and columns to truncations of the residual covariance 
            operator.

        """
        K_T_D_K_T = multi_dot([K_T, D, K_T.T])
        sp, ev = ordered_eigsy(K_T_D_K_T)
        # Cut off the spectrum since the matrix is not full rank:
        cutoff = np.linalg.matrix_rank(K_T_D_K_T)
        sp = sp[: cutoff]
        ev = ev[:, : cutoff]
        norm = diag(multi_dot([ev.T, K_T, D, K, D, K_T.T, ev]))
        norm = pow(norm, -1/2)
        centering_mat = ones((self.data.nobs, self.data.nobs), dtype=float64) / self.data.nobs
        K_ = K - matmul(K, centering_mat) if center else K
        proj = norm * multi_dot([ev.T, K_T, D, K_]).T
        proj = pd.DataFrame(proj, index=self.data.index)
        proj.columns += 1
        return proj
    
    def compute_cook_distances(self, L, t_max=100):
        """
        Computes influence measures in the form of Cook's distances for each
        observation.

        Parameters
        ----------
        L : torch.tensor
            Contrast matrix, defining a statistical test of interest.
        t_max : int, optional
            Maximal truncation of the residual covariance operator, 
            the default is 100.
            
        Returns
        -------
        proj : pandas.DataFrame 
            Contains influence measure values, with rows corresponding to
            observations, and columns to truncations of the residual covariance 
            operator.
            
        """
        W = multi_dot([self.data.exog, self.data._XXinv, L.T])
        LXXLinv = self._compute_LXXLinv(L)
        coef_D = diag(multi_dot([W, LXXLinv, W.T]))
        one_minus_hii = (1 - diag(self.data._ProjImX))
        coef_D /= (one_minus_hii ** 2 * L.shape[0])
        K_T = self._compute_K_T(t_max=t_max)

        cook_distances = {}
        for t in range(1,t_max + 1):
            cook_traces = diag(multi_dot([self.data._ProjImXorthogonal, K_T[:t].T, 
                                          K_T[:t], self.data._ProjImXorthogonal]))
            cook_distances[t] = (cook_traces * coef_D).numpy()
            
        cook_distances = pd.DataFrame(cook_distances, index=self.data.index)
        return(cook_distances)
    
    def correct_pvalues(self, pvalues, correction='bonferroni', by_level=False):
        """
        Corrects the p-values according to the chosen correction strategy.

        Parameters
        ----------
        pvalues : pandas.DataFrame
            Data frame with p-values to correct. Columns indicate hypotheses 
            that are tested and lines indicate truncation levels.
        correction : str, optional
            Relevent for multiple test comparisons, in particular when 
            `by_level=True`. If 'bonferroni' (default), permorms the Bonferroni 
            correction of the p-values. If 'BH', perfoms the Benjamini–Hochberg 
            correction.
        by_level : by_level : bool, optional
            If False (default), computes the global test. If True, computes the 
            test by level or by a pair of levels.

        """
        if hasattr(self, '_factor_info') and by_level:
             factor_cols = [pvalues.columns.str.contains(k) 
                            for k in self._factor_info.keys()]
        else:
            factor_cols = [[True,] * len(pvalues.columns),]
        for f in factor_cols:
            pvalues.loc[:, f] *= np.sum(f)
            if correction == 'BH':
                pval_ranks = pvalues.loc[:, f].rank(axis=1, method='dense')
                pvalues.loc[:, f] /= pval_ranks
                pvalues.clip(upper=1, inplace=True)
        
    def test(self, hypotheses='pairwise', hypotheses_subset=None,
             by_level=False, t_max=100, correction=None, test_intercept=False, 
             true_proportions=False, center_projections=True, verbose=0, n_anchors=None):
        """
        Performs kernel hypothesis tests for the given model. Simultaneously 
        calculates projections on the associated discriminant axes as well as
        influences of observations with respect to the test.

        Parameters
        ----------
        hypotheses : str or None or list[tuple]
            Hypotheses to be tested.
            - if str: either 'pairwise' (default) or 'one-vs-all'. Recommended 
            options in combination with OneHot encoding.
            - if None: produces an identity contrast matrix for each factor.
            Intended for other coding schemes (e.g. Treatment, Sum, etc).
            - if list[tuple]: custom hypothesis option. Each element of the 
            list should be a tuple of size 2: `(name, contrast_L)`, where `name`
            is a string and contrast_L is a contrast matrix in the form of a
            torch.tensor (`dtype=torch.float64`).
        hypotheses_subset : list of strings
            Names of tests to perform, subset of all the tests in the
            `hypotheses` variable (particularly useful with the by_level 
            testing option, when the total number of hypotheses is high and 
            the interest lies in the subset). The default is None, i.e. all 
            hypotheses are tested.
        by_level : bool, optional
            If False (default), computes the global test. If True, computes the 
            test by level or by a pair of levels.
        t_max : int, optional
            Maximal truncation for statistics calculation, the default is 100.
        correction : str or None, optional
            Relevent for multiple test comparisons, in particular when 
            `by_level=True`. If 'bonferroni', permorms the Bonferroni correction 
            of the p-values. If 'BH', perfoms the Benjamini–Hochberg 
            correction. If None (default), the p-values remain uncorrected.
        test_intercept : bool, optional
            If True, adds a test for the intercept, which is set as the 
            grand mean (or the actual mean if `true_proportions=True`) of all the
            level effects. The default is False. It is unnecessary to add an 
            intercept test manually eith this option if an intercept is present
            in the design matrix.
        true_proportions : bool, optional
            Relevant for the calculation of the factor mean, i.e. if 
            `hypotheses='one-vs-all'` or `test_intercept=True`. If False (default),
            the factor mean is the grand mean (mean of means). If True, the
            true level proportions are taken into account, so the factor mean 
            is the actual global mean of the factor.
        center_projections : bool, optional
            If True (default), the projections are centered with respect to
            the factor mean.
        n_anchors : int, optional
            Number of anchors used in the Nystrom method. If None, the value is
            set at `t_max`.
        verbose : int, optional
            The higher the verbosity, the more messages keeping track of 
            computations. The default is 0.
            - < 1: no messages,
            - 1: progress bar with computation time,
            - 2: print tested hypothesis' name.

        Returns
        -------
        KernelAOVResults object
           See the documentation for KernelAOVResults.

        """
        if verbose > 0:
            print('-Computing the Gram matrix...')
        K = self.kernel(self.data.endog)
        if n_anchors is None:
            n_anchors = t_max
        K_T = self._compute_K_T(t_max=t_max, n_anchors=n_anchors)
        if verbose > 0:
            print('-Testing hypotheses:')
        hyps = self.set_hypotheses(hypotheses=hypotheses, by_level=by_level, 
                                   test_intercept=test_intercept,
                                   true_proportions=true_proportions)
        if hypotheses_subset is not None:
            hyps = [h for h in hyps if h[0] in hypotheses_subset]
        results = {}
        projections = {}
        cook_distances = {}
        if correction is not None:
            corrected_pvals_dict = {}
        # Get factor dummies to add the factor information to the data frames:
        factor_dummies = pd.DataFrame(self.data.exog.numpy().astype(int),
                                      columns=self.data.exog_names, index=self.data.index)
        it = tqdm(range(len(hyps))) if verbose > 0 else range(len(hyps))
        for i in it:
            name, L = hyps[i]
            if verbose > 1:
                print('\n')
                print(f'-Testing {name}...')
            if any(isinstance(l, str) for l in L):
                L = DesignInfo(self.data.exog_names).linear_constraint(L).coefs
                L = convert_to_torch(L)
            L = L.unsqueeze(0) if L.dim() == 1 else L
            D = self._compute_D(L)
            stats, pvals = self._compute_kernel_HL_test_statistic(K_T, D, len(L), 
                                                                 t_max=t_max)
            results_dict = {'TKHL' : stats,
                               'P-value' : pvals}
            results[name] = pd.DataFrame(results_dict)
            results[name].index.name = 'Truncation'
            results[name].index += 1
            if correction is not None:
                corrected_pvals_dict[name] = pvals
                
            # Projections on discriminant axes and Cook's distances:
            projections[name] = self.project_on_discriminant(K, K_T, D,
                                                             center=center_projections)
            cook_distances[name] = self.compute_cook_distances(L, t_max=t_max)
            if not hasattr(self, 'formula'):
                projections[name][name] = np.nan
                cook_distances[name][name] = np.nan
            elif name == 'Intercept' or hypotheses not in [None, 'pairwise', 'one-vs-all']:
                pass
            else:
                # Adding factor information:
                pd.options.mode.chained_assignment = None  # default='warn'
                if not by_level: # specify all levels for all factors
                    _slice = self._factor_info[name]
                    dummies_factor_i = factor_dummies.iloc[:, _slice]
                else: # specify only those levels that are relevant for a given test
                    if hypotheses == 'one-vs-all':
                        _slice = [s for f, s in self._factor_info.items() if f in name][0]
                    else:
                        _slice = [i for i, en in enumerate(self.data.exog_names) if en in name]
                    dummies_factor_i = factor_dummies.iloc[:, _slice]
                    # Case of interaction effects:
                    if ':' in name:
                        interaction_cols = dummies_factor_i.columns.str.contains(':')
                        dummies_factor_i = dummies_factor_i.loc[:, interaction_cols]
                    # Put NaN for the irrelevant levels (needed for visualizations)
                    nan_obs = (dummies_factor_i.max(axis=1) == 0)
                    dummies_factor_i['NaN'] = 0
                    dummies_factor_i.loc[nan_obs, 'NaN'] = 2
                projections[name][name] = dummies_factor_i.idxmax(axis=1)
                cook_distances[name][name] = dummies_factor_i.idxmax(axis=1)
        # Correct p-values:
        if correction is not None:
            if verbose > 0:
                print('-Correcting p-values for multiple tests...')
            corrected_pvals_df = pd.DataFrame(corrected_pvals_dict)
            corrected_pvals_df.index += 1
            self.correct_pvalues(corrected_pvals_df, correction=correction, by_level=by_level)
            for hyp in hyps:
                name, L = hyp
                results[name]['P-value'] = corrected_pvals_df[name]
        return KernelAOVResults(hyps, results, projections, cook_distances)
    
class KernelAOVResults():
    """
    Class implementing Kernel Analysis Of Variance.
    
    Parameters
    ----------
    hypotheses : list
        All the consideres hypotheses. Each element of the list represents a 
        hypothesis in the form of a list with two elements: a name and a 
        congtrast matrix associated with the test.
    stats : dict
        A dictionary with keys corresponding to hypothesis names, and values 
        to instances of pandas.DataFrame with the results of the corresponding
        tests. Each data frame contains two columns, the first containing the
        truncated kernel Hotelling-Lawley test statistic (TKHL) values, and the
        second containing the associated p-values, indexed by truncations of the 
        residual covariance operator used in the calculations.
    projections : dict
        A dictionary with keys corresponding to hypothesis names, and values 
        to instances of pandas.DataFrame with the projections obtained with
        `AOV.project_on_discriminant`. In each data frame, rows correspond to
        observations, and columns to truncations of the residual covariance 
        operator.
    cook_distances : dict
        A dictionary with keys corresponding to hypothesis names, and values 
        to instances of pandas.DataFrame with the influence measure values
        obtained with `AOV.compute_cook_distances`. In each data frame, rows 
        correspond to observations, and columns to truncations of the 
        residual covariance operator.

    Attributes:
    ----------
    hypotheses : list
        See Parameters.
    stats : dict
        See Parameters.
    projections : dict
        See Parameters.
    cook_distances : dict
        See Parameters.
        
    """
    
    def __init__(self, hypotheses, stats, projections, cook_distances):
        self.hypotheses = hypotheses
        self.stats = stats
        self.projections = projections
        self.cook_distances = cook_distances
        
    def _summary_obj(self, t_max=5):
        """
        Creates a summary object to display a summary of the test.

        Parameters
        ----------
        t_max : int, optional
            Maximal truncation to display. The default is 5.

        Returns
        -------
        An instance of statsmodels.iolib.summary2

        """
        summ = summary2.Summary()
        summ.add_title('Kernel Analysis of Variance')
        for i, key in enumerate(self.stats):
            summ.add_dict({'': ''})
            df = self.stats[key].iloc[:t_max].transpose()
            df.columns = [f'T={t}' for t in df.columns]
            df.index = ['| ' + ' ' * (7 - len(ind)) + ind for ind in df.index]
            df = df.reset_index()
            c = list(df.columns)
            c[0] = key + ' |  Trunc.'
            df.columns = c
            df.index = ['', '']
            float_format = '%.3f'
            summ.add_df(df, float_format=float_format)
        return summ
    
    def summary(self, t_max=5):
        """
        Prints a summary of the test.

        Parameters
        ----------
        t_max : int, optional
            Maximal truncation to display. The default is 5.

        """
        _sum_obj = self._summary_obj(t_max=t_max)
        print(_sum_obj)
        
    def __str__(self):
        return self._summary_obj().__str__()
    
    def plot_density(self, t=100, tests=None, colormap='viridis', alpha=.5,
                     legend_fontsize=12, font_family='serif', figsize=None):
        """
        Plots kernel-densities of projections of the embeddings on the chosen
        discriminant axis, associated with the tests underlying the
        KernelAOVResults object. Produces separate subplots for each test.

        Parameters
        ----------
        t : int, optional
            Axis to plot, i.e. the embeddings are projected on the t-th 
            eigenfunction.
        tests : list of strings or None
            List containing names of tests to plot, out of all the tests in 
            the KernelAOVResults object (particularly useful with the by_level 
            testing option). The default is None, i.e. all tests are plotted.
        colormap : str, optional
            The name of a matplotlib colormap to be used for different factor
            levels. The default is 'viridis'.
        alpha : float, optional
           The alpha blending value, between 0 (transparent) and 1 (opaque).
           The default is 0.5.
        legend_fontsize : int, optional
            Legend font size. The default is 15.
        font_family : str, optional
             Legend and labels' font family name accepted by matplotlib 
             (e.g., 'serif', 'sans-serif', 'monospace', 'fantasy' or 'cursive'),
             the default is 'serif'.
        figsize : tuple, optional
            The size of the figure. If not specified, is set to
            (8 * nb_factors, 6).
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            A Figure object of the plot.
        axs : numpy.ndarray of matplotlib.axes._axes.Axes
            An Axes object of the plot.

        """
        tests = self.projections.keys() if tests is None else tests
        nb_tests = len(tests)
        
        rc('font',**{'family': font_family})
        
        figsize = figsize if figsize is not None else (8 * nb_tests, 6)
        fig, axs = plt.subplots(ncols=nb_tests, figsize=figsize)
        for f, test in enumerate(tests):
            ax = axs if nb_tests == 1 else axs[f]
            T_max = len(self.projections[test].columns) - 1
            t = min(t, T_max)
            proj_f = self.projections[test]
            test_lvls = proj_f[test].unique()
            test_lvls = test_lvls[test_lvls != 'NaN'] # extract relevant observations
            cmap = colormaps[colormap]
            colors = cmap(np.linspace(0.1, 0.9, len(test_lvls)))
            no_lvl_info = proj_f[test].isnull().all()
            for i, test_lvl in enumerate(test_lvls):
                if no_lvl_info:
                    lvl_proj = proj_f[t]
                else:
                    lvl_proj = proj_f[proj_f[test] == test_lvl][t]
                min_proj, max_proj = lvl_proj.min(), lvl_proj.max()
                min_scaled = min_proj - 0.1 * (max_proj - min_proj)
                max_scaled = max_proj + 0.1 * (max_proj - min_proj)
                x = np.linspace(min_scaled, max_scaled, 200)
                try:
                    density = gaussian_kde(lvl_proj, bw_method=.2)
                    y = density(x)                
                    ax.plot(x, y, color=colors[i], lw=2)
                    ax.fill_between(x, y, y2=0, color=colors[i], label=test_lvl, alpha=alpha)
                except ValueError:
                    pass
            if not no_lvl_info:
                ax.legend(bbox_to_anchor=(1.01, 0.5), loc='center left',
                          fontsize=legend_fontsize)
            ax.set_title(test, fontsize=18)
            ax.set_xlabel('Discriminant axis', fontsize=14)
            ax.set_ylabel('Density', fontsize=14)
        plt.tight_layout()
        fig.suptitle(f'Discriminant axis projection density (t={t})', 
                     fontsize=25, y=1.05)
        plt.show()
        return fig, axs
        
    def plot_influence(self, t1=100, t2=100, tests=None, 
                       colormap='viridis', alpha=.5, legend_fontsize=12, 
                       font_family='serif', figsize=None):
        """
        Plots influences (Cook's distances) of the embeddings, associated with 
        the tests underlying the KernelAOVResults object, against their 
        projections on the chosen discriminant axis. Produces separate 
        subplots for each test.

        Parameters
        ----------
        t1 : int, optional
            Truncation of the resdual covariance operator used for the Cook's
            distance calculation.
        t2 : int, optional
            Axis of the projections, i.e. the embeddings are projected on the t-th 
            eigenfunction.
        tests : list of strings
            List containing a list of tests to plot, out of all the tests in 
            the KernelAOVResults object (particularly useful with the by_level 
            testing option).
        colormap : str, optional
            The name of a matplotlib colormap to be used for different factor
            levels. The default is 'viridis'.
        alpha : float, optional
           The alpha blending value, between 0 (transparent) and 1 (opaque).
           The default is 0.5.
        legend_fontsize : int, optional
            Legend font size. The default is 15.
        font_family : str, optional
             Legend and labels' font family name accepted by matplotlib 
             (e.g., 'serif', 'sans-serif', 'monospace', 'fantasy' or 'cursive'),
             the default is 'serif'.
        figsize : tuple, optional
            The size of the figure. If not specified, is set to
            (8 * nb_factors, 6).
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            A Figure object of the plot.
        axs : numpy.ndarray of matplotlib.axes._axes.Axes
            An Axes object of the plot.

        """
        tests = self.cook_distances.keys() if tests is None else tests
        nb_tests = len(tests)
        
        rc('font',**{'family': font_family})
        
        figsize = figsize if figsize is not None else (8 * nb_tests, 6)
        fig, axs = plt.subplots(ncols=nb_tests, figsize=figsize)
        for f, test in enumerate(tests):
            ax = axs if nb_tests == 1 else axs[f]
            T_max = len(self.cook_distances[test].columns) - 1
            t1 = min(t1, T_max)
            t2 = min(t2, T_max)
            cook_f = self.cook_distances[test]
            proj_f = self.projections[test]
            test_lvls = cook_f[test].unique()
            test_lvls = test_lvls[test_lvls != 'NaN'] # extract relevant observations
            cmap = colormaps[colormap]
            colors = cmap(np.linspace(0.1, 0.9, len(test_lvls)))
            no_lvl_info = cook_f[test].isnull().all()
            for i, test_lvl in enumerate(test_lvls):
                if no_lvl_info:
                    lvl_cook = cook_f[t1]
                    lvl_proj = proj_f[t2]
                else:
                    lvl_cook = cook_f[cook_f[test] == test_lvl][t1]
                    lvl_proj = proj_f[proj_f[test] == test_lvl][t2]        
                ax.scatter(lvl_proj, lvl_cook, color=colors[i], 
                           alpha=alpha, label=test_lvl)
            if not no_lvl_info:
                ax.legend(bbox_to_anchor=(1.01, 0.5), loc='center left',
                          fontsize=legend_fontsize)
            ax.set_title(test, fontsize=18)
        plt.tight_layout()
        fig.suptitle(f"Cook's distances (t={t1}) against projections (t={t2})", 
                     fontsize=25, y=1.05)
        plt.show()
        return fig, axs
          
