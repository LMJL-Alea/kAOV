# kAOV: Kernel Analysis of Variance
A Python library for general kernel hypothesis testing, accompanying [[1]](#1).

## Installation
Install from GitHub:
```console
$ pip install kaov@git+https://github.com/LMJL-Alea/kAOV@main
```
## How to use
Load a dataset:
```python
>>> import pandas as pd
>>> url = "https://raw.githubusercontent.com/LMJL-Alea/kAOV/refs/heads/main/Data/reversion_kAOV.csv"
>>> data = pd.read_csv(url, index_col=0)
```
Write a formula using OneHot encoding:
```python
>>> formula = ' + '.join(data.columns[:-2]) + ' ~ C(Medium, OneHot) + C(Batch, OneHot)'
>>> print(formula)
AACS + ACSL6 + ACSS1 + ALAS1 + AMDHD2 + ARHGEF2 + BATF + BCL11A + betaglobin + BPI + CD151 + CD44 + CREG1 + CRIP2 + CTCF + CTSA + CYP51A1 + DCP1A + DCTD + DHCR24 + DHCR7 + DPP7 + EGFR + EMB + FAM208B + FHL3 + FNIP1F1R1 + GLRX5 + GPT2 + GSN + HMGCS1 + HRAS1 + HSD17B7 + HSP90AA1 + HYAL1 + LCP1 + LDHA + MAPK12 + MFSD2B + MID2 + MKNK2 + MTFR1 + MVD + MYO1G + NCOA4 + NSDHL + PDLIM7 + PIK3CG + PLAG1 + PLS1 + PLS3 + PPP1R15B + PTPRC + RBM38 + REXO2 + RFFL + RPL22L1 + RSFR + RUNX2 + sca2 + SCD + SERPINI1 + SLC25A37 + SLC6A9 + SLC9A3R2 + SMPD1 + SNX22 + SNX27 + SQLE + SQSTM1 + STARD4 + STX12 + SULF2 + SULT1E1 + TADA2L + TBC1D7 + TNFRSF21 + TPP1 + TTYH2 + UCK1 + VDAC3 + WDR91 + XPNPEP1 ~ C(Medium, OneHot) + C(Batch, OneHot)
```
Create a model and run a test:
```python
>>> from kaov import AOV
>>> kfit = AOV.from_formula(formula, data=data)
>>> kfit.test().summary()
                  Kernel Analysis of Variance
===============================================================
                                                               
---------------------------------------------------------------
 Medium |  Trunc.   T=1      T=2      T=3      T=4       T=5   
---------------------------------------------------------------
        |    TKHL 204.6946 423.7650 528.7334 552.0367 1263.0267
        | P-value   0.0000   0.0000   0.0000   0.0000    0.0000
---------------------------------------------------------------
                                                               
---------------------------------------------------------------
    Batch |  Trunc.   T=1     T=2      T=3      T=4      T=5   
---------------------------------------------------------------
          |    TKHL 85.9009 209.6519 315.4264 401.4933 465.7612
          | P-value  0.0000   0.0000   0.0000   0.0000   0.0000
===============================================================
```

For more details see the [tutorial](https://github.com/LMJL-Alea/kAOV/blob/main/tutorial_kaov.ipynb).

## References

<a id="1">[1]</a> 
Anthony Ozier-Lafontaine, Polina Arsenteva, Franck Picard, Bertrand Michel. Extending Kernel Testing To General Designs.  2024. Preprint, [	arXiv:2405.13799](https://arxiv.org/abs/2405.13799)
