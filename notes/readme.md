effect on imputation on predicted proability of unclassified sources
with mode imputation, validation accuracies are comparable, in fact mode accuracy is a bit better than rf imputation. 
But when compared to the ecdf of predicted probabilities of unidentified sources, mode performs very poor. 
with following statistics 
for no prob th --
| class | num  0| 0.5   | 0.8 | 0.9 |
|-----|-----    | ---   | --- |--- |
| STAR |   10680|7174   |2541   
| AGN   |  10049|5761   |293
| YSO   |   9691|5718   |217
| HMXB  |   7246|2630   |495
| ULX   |   1114|144    |3
| CV    |    928|79     |11
| LMXB   |   185|94     |37


with Gradient boosting classifier
prob th : 0	prob th:0.5	prob th:0.75	prob th:0.8	prob th:0.9	prob th:0.95
YSO	9572	8777	6540	5871	3843	2033.0
STAR	8637	7273	5258	4801	3639	2727.0
AGN	6913	5459	3477	3043	1993	1214.0
HMXB	4823	2295	897	758	520	327.0
ULX	3103	1525	344	179	34	NaN
PULSAR	3007	1008	132	57	2	NaN
CV	2967	950	207	141	42	13.0
LMXB	871	230	114	102	80	67.0

with 0.5 --
STAR    7174
AGN     5761
YSO     5718
HMXB    2630
ULX      144
LMXB      94
CV        79

with 0.8 -- 
STAR    2541
HMXB     495
AGN      293
YSO      217
LMXB      37
CV        11
ULX        3

with 0.9 --
STAR    987
HMXB    248
YSO      13
LMXB     11
CV        8
AGN       1




## Improvement reason
* Conf flag , extended source flag
* After dropping GAIA features (or any other MW feature) , not much effect 
*   * But dropping all mw features at once acc drops 90 > 82-83 
* Dropping color features no effect.

## Imputation method comparison 
* Unresolved

## MUWCLASS training set comparison
* Source confusion due to close cross-match
* We have kept closest match source > What they have done ?
* No clear information about their object identification catalog
* In their training set, pulsars in ATNF are classified as LMXB ,
* * But in ATNF doccumantation, its given that there is no accretion-powerd pulsars



sky coverage comparison plot 


---
Updates on Apr 5 2022
# PULSAR catalogue verification
> **Issue** - For pulsar catalogue, we have ~30 cross matches from SIMBAD and similar from ATNF, but we are sure that ATNF pulsars are not accreting pulsars, but what about SIMBAD, If they are accreting then they may come in LMXB category. 

>Verify the origin of sourcecs of SIMBAD Pulsars

# Imputation verification