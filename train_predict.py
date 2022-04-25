import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt 
from IPython.display import display
from tqdm import tqdm_notebook
import tqdm
import seaborn as sns
from choices import param_dict
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import  IterativeImputer 
from sklearn.impute import SimpleImputer , KNNImputer
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import GradientBoostingClassifier
sns.set(font_scale=1.3, rc={'axes.facecolor':'white', 'figure.facecolor':'white' , 'axes.grid':True} , style="whitegrid")
# 
feat_to_drop = param_dict['hardness']+param_dict['IRAC']

from utilities import deets
from choices import get_train_data , param_dict
classes = ['AGN' ,'STAR' , 'YSO' ,  'CV' , 'LMXB' , 'HMXB' ,'ULX','PULSAR']
flag = {
    'conf_flag' : 0 , 
    'streak_src_flag' : 0 , 
    'extent_flag' : 0 , 
    'pileup_flag' : 0 , 
    }
ret_dict= {
    'clf': True,
 'prob_table': True,
 'acc': True,
 'pr_score': True,
 'precision': True,
 'recall': True , 
 'roc_auc_score' : True
 }


gb = GradientBoostingClassifier()


d = '1iter_rfimp'
model_name = 'GB'
model = gb 

file = f'compiled_data_v3/imputed_data_v2/x_phot_minmax_{d}imp.csv'
data = get_train_data(flags = flag, classes= classes , offset = 1, file=file)
data = data.drop(columns = feat_to_drop)
deets(data,0)

from utilities import cv
from sklearn.ensemble import GradientBoostingClassifier , RandomForestClassifier

# {'max_depth': 30, 'n_estimators': 400} for MODE imputation
# {'max_depth': 180, 'n_estimators': 400} for RF imputation
rf = RandomForestClassifier(n_estimators=400 , max_depth=30 , n_jobs=-1)
res_final  = cv(
    {'data' : data , 'name' : f'data_10iter_rfimp_tuned'},   
    {'model' : model , 'name' :'GB'} , ## CHANGE HERE########################################################
    k=10 , return_dict = ret_dict, save_df= f'temp_res_comp/train_prob/{d}_{model_name}.csv' , multiprocessing = True)


u = pd.read_csv(f'compiled_data_v3/imputed_data_v2/unid_phot_minmax__{d}_imp.csv' , index_col='name').iloc[:,1:]
u = u.drop(columns=feat_to_drop)
deets(u)


from utilities import softmax , norm_prob
clf = res_final['clf']
pred_prob = (clf.predict_proba(u))
pred_prob_df = pd.DataFrame(pred_prob , columns=[f'prob_{el}' for el in clf.classes_] , index = u.index.to_list())
pred_prob_df


u_df = pd.DataFrame({
    'name' : u.index.to_list() , 
    'class' : clf.predict(u) , 
    'prob' : [np.amax(el) for el in pred_prob] ,
    #'prob_margin' : [el[-1]-el[-2] for el in np.sort(pred_prob , axis=1 ,)]
}).set_index('name')
u_df = pd.merge(u_df , pred_prob_df , left_index=True , right_index=True)
u_df.index.name = 'name'
u_df 


##### CHANGE HERE ######
u_df.to_csv(f'temp_res_comp/unid_prob/{model_name}_{d}.csv')