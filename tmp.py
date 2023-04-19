from astropy.coordinates import SkyCoord
from astropy import units as unit 
import pandas as pd
from matplotlib import pyplot as plt 
df_plot = pd.read_csv('compiled_data_v3/id_frame.csv' , index_col='name')
df_plot = df_plot[df_plot['class']=='HMXB']
eq = SkyCoord(df_plot['ra'] , df_plot['dec'] , unit = unit.deg)
gal = eq.galactic
plt.figure(figsize=(14,14))
plt.subplot(projection='aitoff', )
plt.scatter(gal.l.wrap_at('180d').radian, gal.b.radian , s=10, marker='o',alpha=0.4 , label='HMXB' , color='k')
plt.show()
# ax.set_title(classes[i-1])