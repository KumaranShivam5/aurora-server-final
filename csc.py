import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt 
import seaborn as sns
from astropy.io import fits
import astropy 
import pyvo as vo
import numpy as np

tap = vo.dal.TAPService('http://cda.cfa.harvard.edu/csc2tap');

qry = """
SELECT m.name,m.ra,m.dec,m.var_flag,m.significance,m.acis_num,m.acis_hetg_num,m.acis_letg_num,m.hrc_num,m.hrc_hetg_num,m.hrc_letg_num,m.acis_time,m.acis_hetg_time,m.acis_letg_time,m.hrc_time,m.hrc_hetg_time,m.hrc_letg_time

FROM csc2.master_source m
WHERE m.name NOT LIKE '%X'
"""


cat = tap.search(qry)

tbl = cat.to_table()
tbl['name'] = [str(n) for n in tbl['name']]
df = tbl.to_pandas()
df = df.set_index('name')
df.to_csv('flags/chandra_all_time_flags.csv')