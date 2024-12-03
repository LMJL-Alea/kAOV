# kAOV: Kernel Analysis of Variance

## Installation
Install from GitHub:
```console
$ pip install kaov@git+https://github.com/LMJL-Alea/kAOV@main
```
## How to use
Load dataset:
```python
>>> import pandas as pd
>>> url = "https://raw.githubusercontent.com/LMJL-Alea/kAOV/refs/heads/main/Data/reversion_kAOV.csv"
>>> data = pd.read_csv(url, index_col=0)
```
Write the formula using OneHot encoding:
```python
>>> formula = ' + '.join(data.columns[:-2]) + ' ~ C(Medium, OneHot) + C(Batch, OneHot)'
>>> print(formula)
AACS + ACSL6 + ACSS1 + ALAS1 + AMDHD2 + ARHGEF2 + BATF + BCL11A + betaglobin + BPI + CD151 + CD44 + CREG1 + CRIP2 + CTCF + CTSA + CYP51A1 + DCP1A + DCTD + DHCR24 + DHCR7 + DPP7 + EGFR + EMB + FAM208B + FHL3 + FNIP1F1R1 + GLRX5 + GPT2 + GSN + HMGCS1 + HRAS1 + HSD17B7 + HSP90AA1 + HYAL1 + LCP1 + LDHA + MAPK12 + MFSD2B + MID2 + MKNK2 + MTFR1 + MVD + MYO1G + NCOA4 + NSDHL + PDLIM7 + PIK3CG + PLAG1 + PLS1 + PLS3 + PPP1R15B + PTPRC + RBM38 + REXO2 + RFFL + RPL22L1 + RSFR + RUNX2 + sca2 + SCD + SERPINI1 + SLC25A37 + SLC6A9 + SLC9A3R2 + SMPD1 + SNX22 + SNX27 + SQLE + SQSTM1 + STARD4 + STX12 + SULF2 + SULT1E1 + TADA2L + TBC1D7 + TNFRSF21 + TPP1 + TTYH2 + UCK1 + VDAC3 + WDR91 + XPNPEP1 ~ C(Medium, OneHot) + C(Batch, OneHot)
```
Create a model and run a test:
```python
>>> from kaov import AOV
>>> kfit = AOV.from_formula(formula, data=data)
>>> print(kfit.test())
            Kernel Analysis of Variance
====================================================
                                                    
----------------------------------------------------
  Batch    T=1     T=2      T=3      T=4      T=5   
----------------------------------------------------
    TKHL 23.5792 217.6872 223.0625 280.5633 357.7317
 P-value  0.0014   0.0000   0.0000   0.0000   0.0000
====================================================
```