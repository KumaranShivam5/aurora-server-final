from IPython.display import display
from tqdm import tqdm_notebook
import tqdm

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import  IterativeImputer 
from sklearn.impute import SimpleImputer , KNNImputer
from sklearn.preprocessing import Normalizer

from joblib import dump , load


from sklearn.ensemble import RandomForestRegressor


import pandas as pd 
import numpy as np 

RN =  199033
from utilities import deets

def hr():
    print('-------------------------------------------')
x = pd.read_csv('compiled_data_v3/x_phot_minmax.csv')
u = pd.read_csv('compiled_data_v3/x_phot_minmax-unid-var-src.csv')
#u = pd.read_csv('compiled_data_v2/final/x_phot_ui_minmax.csv')
x = x.set_index('name')
u = u.set_index('name')
hard_var_col = ['var_inter_hard_prob_hs', 'ks_intra_prob_b', 'var_inter_hard_sigma_hm', 'var_inter_hard_prob_ms', 'var_inter_hard_prob_hm',]
model_fit_col =         ['powlaw_gamma',
        'powlaw_nh',
        'powlaw_ampl',
        'powlaw_stat',
        'bb_kt',
        'bb_nh',
        'bb_ampl',
        'bb_stat',
        'brems_kt',
        'brems_nh',
        'brems_norm',
        'brems_stat',
        ]
sparse_col = [
    '0p5_2csc' , '2-10 keV (XMM)' , '1_2_csc' , '0p5_8_csc'
]
x = x.drop(columns=hard_var_col+sparse_col)
u = u.drop(columns=hard_var_col+sparse_col)
deets(x)
hr()
print('GOING for imputation')

imp_type = ['1iter_rfimp' , '10iter_rfimp' , 'mode' , 'mean' , 'const' ,'knn']


#print(x.columns.to_list())

estim = RandomForestRegressor(n_jobs=-1 , random_state=RN)

from missingpy import MissForest

imp_dict ={
    '1iter_rfimp' : IterativeImputer(
        estimator = estim , 
        verbose=2 ,
        max_iter = 1 ,
        random_state= RN , 
    ) , 
    '10iter_rfimp' : IterativeImputer(
        estimator = estim , 
        verbose=2 ,
        max_iter = 10 ,
        random_state= RN , 
    ) , 
    'mode' : SimpleImputer(strategy='most_frequent' , ) , 
    'mean' : SimpleImputer(strategy='mean' , ) , 
    'const' : SimpleImputer(strategy='constant' , fill_value=-100) , 
    'knn' : KNNImputer(weights='distance') , 
    'forest' : MissForest(),
}

imp_type = str(input("ENTER Imputation type :\n  ['1iter_rfimp' , '10iter_rfimp' , 'mode' , 'mean' , 'const' ,'knn'] \n : " ))
imp = imp_dict[imp_type]



#imp = SimpleImputer(strategy='most_frequent' , )
#imp = KNNImputer(weights='distance')

cols = x.columns.to_list()
#ind = u.index.to_list()
ind_x = x.index.to_list()
ind_u = u.index.to_list()
imp.fit(x)

print(x)
print(x.shape)
#imp = load('filename.joblib') 
x = imp.transform(x)
x = pd.DataFrame(x, columns = cols)
x.insert(0 , 'name' , ind_x)
x.to_csv(f'compiled_data_v3/imputed_data_v2/x_phot_minmax_{imp_type}imp.csv')

u = imp.transform(u)
u = pd.DataFrame(u, columns = cols)
u.insert(0 , 'name' , ind_u)
u.to_csv(f'compiled_data_v3/imputed_data_v2/unid_phot_minmax__{imp_type}_imp.csv')

#dump(imp, 'models/imputer_v2.joblib') 

# u = imp.transform(u)
# u = pd.DataFrame(u, columns = cols)
# u.insert(0 , 'name' , ind)
# u.to_csv('compiled_data/final/x_phot_ui_minmax_rfimp.csv')

