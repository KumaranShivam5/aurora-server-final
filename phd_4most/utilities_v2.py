from IPython.display import display 
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split , LeaveOneOut , KFold ,StratifiedKFold
from tqdm import tqdm
def deets(df ,class_info = 0 ,dsp=0):
    print('_____________________________________________________')
    if(dsp):
        display(df)
        print('_____________________________________________________')
    print('------------------------------')
    print(f'Number of Objects : {df.shape[0]}')
    print(f'Number of Columns : {df.shape[1]}')
    if(class_info):
        print('------------------------------')
        display(df['class'].value_counts())
    print('_____________________________________________________')
#df_deets(data)


from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
import pandas as pd 



def oversampling(method, X_train, y_train):
    # oversampling training dataset to mitigate for the imbalanced TD
    # default = SMOTE
    exnum = 9282948832
    X_train.replace(np.nan, exnum, inplace=True)
    X_res, y_res = method.fit_resample(X_train, y_train)
    res = X_res.values

    X_train[X_train == exnum] = np.nan
    X_train_min = np.nanmin(X_train, axis=0)

    for i in np.arange(len(res[:,0])):
        for j in np.arange(len(res[0,:])):
            if res[i,j] <= X_train_min[j]:
                res[i,j] = np.nan
                
    X_res[:] = res
    return X_res, y_res

from imblearn.over_sampling import SMOTE
model_dict = {
    'RF' : RandomForestClassifier() , 
    'RF_mod' : RandomForestClassifier(
        n_estimators = 500 , 
    ) , 
    'GB' : GradientBoostingClassifier() , 
}



flag = {
    'conf_flag' : 0 , 
    'streak_src_flag' : 0 , 
    'extent_flag' : 0 , 
    'pileup_flag' : 0 , 
    }
classes =['AGN' ,'YSO' ,'STAR' ,'HMXB' ,'LMXB' ,'ULX' ,'CV' ,'PULSAR']

def get_train_data(flags=flag , classes=classes , offset = -1 , sig = 0, deets=0 , file = None ,ret_id_cols = ['class']):
    #print(flags)
    data_id = pd.read_csv('compiled_data_v3/id_frame.csv' , index_col='name')
    id_col = data_id.columns.to_list()
    default_data = 'compiled_data_v3/imputed_data_v2/x_phot_minmax_modeimp.csv'
    if(file):
        x_data = pd.read_csv(file , index_col = 'name').iloc[:,  1:]
    else:
        x_data = pd.read_csv( default_data, index_col = 'name').iloc[:,  1:]
    x_col = x_data.columns.to_list()
    #print(x_col)
    data = pd.concat([data_id, x_data] , axis=1).sort_values(by='offset')
    data = pd.concat([data_id, x_data] , axis=1).sort_values(by='offset')
    data = pd.merge(data_id , x_data , left_index=True , right_index= True ,  how ='left')
    #data = data.drop_duplicates('name')
    data= data.drop_duplicates(['ra','dec'])
    #data = data.set_index('name')
    if(deets):
        deets(data , 1)
    for flag , val in zip(flags.keys() , flags.values()):
        data = data[data[flag]==val]
    eps = 0.01 # offset epsilon
    if(offset>0):
        data = data[data['offset']<=offset+eps]
    max = data['offset'].max()
    print(f"offset:  \t{data['offset'].min() :.3f}|{data['offset'].max():.3f}")
    data = data[data['significance']> sig]
    print(f"singinficance:  {data['significance'].min():.3f}|{data['significance'].max():.3f}")
    data = data[data['class'].isin(classes)]
    src_class = data['class'].to_list()
    x = data[x_col]
    #print(id_col)
    for l in ret_id_cols:
        #print((l in(id_col)))
        if(not l in(id_col)):
            raise KeyError(f'Entered variable "{l}" is not in the database. ')
        else:
            x.insert(0 , l , data[l].to_list())
    return x

data_dict = {
    'no_filter' : 
        get_train_data(
            flags = {
            'conf_flag' : 0 , 
            'streak_src_flag' : 0 , 
            'extent_flag' : 0 , 
            'pileup_flag' : 0 , 
            } , 
            classes = ['AGN' ,'STAR' , 'YSO' , 'PULSAR'  , 'CV' , 'LMXB' , 'HMXB' ,'ULX'] , 
            offset = -1 , 
            sig = 0
        ) ,

    'off_2_sig_3' : 
        get_train_data(
            flags = {
            'conf_flag' : 0 , 
            'streak_src_flag' : 0 , 
            'extent_flag' : 0 , 
            'pileup_flag' : 0 , 
            } , 
            classes = ['AGN' ,'STAR' , 'YSO' , 'PULSAR'  , 'CV' , 'LMXB' , 'HMXB' ,'ULX'] , 
            offset = 2 , 
            sig = 3
        ) 

}


from choices import model_dict , data_dict
def train_model_loo(arr):
    model , x ,y , index = arr
    train_ix , test_ix = index[0] , index[1]
    x_train , x_test = x.loc[train_ix , : ] , x.loc[test_ix, :]
    y_train , y_test = y.loc[train_ix] , y.loc[test_ix]
    #display(x_train.head(10) , y_train.head(10))
    oversampler  = SMOTE(k_neighbors=4)
    x_train_up , y_train_up = oversampling(oversampler , x_train, y_train)
    x_train_up = x_train_up.replace(np.nan , -100)
    #x_test_up = x_test_up.replace(np.nan , -100)
    if(type(model)==str):
        clf = model_dict[model]
    else:
        clf = model
    clf.fit(x_train_up , y_train_up)
    return [clf.predict(x_test)[0], y_test , clf.predict_proba(x_test)]

def train_model_kfold(arr):
    model , x ,y , index , names = arr
    train_ix , test_ix  , = index[0] , index[1]
    x_train , x_test = x.loc[train_ix , : ] , x.loc[test_ix, :]
    test_names = names.loc[test_ix]
    y_train , y_test = y.loc[train_ix] , y.loc[test_ix]
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    test_names = test_names.reset_index(drop=True)
    oversampler  = SMOTE(k_neighbors=4)
    x_train_up , y_train_up = oversampling(oversampler , x_train, y_train)
    x_train_up = x_train_up.replace(np.nan , -100)
    if(type(model['model'])==str):
        clf = model_dict[model['model']]
    else:
        clf = model['model']
    clf.fit(x_train , y_train)
    df = pd.DataFrame({
        'name' : test_names , 
        'true_class' : y_test , 
        'pred_class' : clf.predict(x_test) , 
        'pred_prob' : [np.amax(el) for el in clf.predict_proba(x_test)]
    }).set_index('name')
    mem_table = pd.DataFrame(clf.predict_proba(x_test) , columns=[f'prob_{el}' for el in clf.classes_])
    mem_table.insert(0 , 'name' , test_names)
    mem_table = mem_table.set_index('name')
    df = pd.merge(df , mem_table , left_index=True , right_index=True)
    return df


ret_dict =  {
    'clf' : False , 
    'train_prob_table' : True , 
    'acc' : True , 
    'pr_score' : True , 
    'precision' : True , 
    'recall' : True ,
    #'recall' : True ,
}

def cv(data , model , k=-1 , return_dict  = ret_dict , save_df = '' , multiprocessing= 1 ):
    d_name = data['name']
    m_name = model['name']
    if(type(data['data'])==str):
        x = data_dict[data['data']]
    else : x = data['data']
    x = x.sample(frac=1)
    x = x.reset_index()
    x_name = x['name']
    y = x['class']
    classes = y.unique()
    x= x.drop(columns=['class' , 'name'])

    if k==-1:
        print('[INFO] Doing LeaveOneOut cross-validation')
        cv = LeaveOneOut()
    else:
        print(f'Doing {k} fold cross-validation')
        cv = StratifiedKFold(k)# KFold(k) 
    model = model
    index = [(t,i) for t,i in cv.split(x,y)]
    arr = list(zip([model]*len(index) , [x]*len(index) , [y]*len(index) , index , [x_name]*len(index)))
    if(multiprocessing):
        import multiprocessing as mp 
        num_cores = mp.cpu_count()
        with mp.Pool(int(num_cores)) as pool:
            if(k==-1):
                res = pool.map(train_model_loo , arr) 
            else: 
                res = pool.map(train_model_kfold , arr) 
    else:
        res = []
        for a in tqdm(arr):
            res.append(train_model_kfold(a))

    if k==-1 :
        res_df  = pd.DataFrame({
            'true_class' : [el[1].iloc[0] for el in res] , 
            'pred_class' : [el[0] for el in res], 
            'pred_prob' : [np.amax(el[2] )for el in res]
        })
        if(save_df):
            res_df.to_csv(f'validation_res/{d_name}_{m_name}_loo.csv')
    else:
        res_df = pd.concat(res, axis=0)
        #display(res_df)
        if(save_df):
            res_df.to_csv(f'{save_df}')
    acc_sc = accuracy_score(res_df['true_class'] , res_df['pred_class'])

    print(f'[RESULT] Overall Accuracy : {acc_sc :.2f}')
    ret = {}
    if(return_dict['prob_table']):
        print("[INFO] Validation probability table is available as ['prob_table'] in response")
        ret['prob_table'] = res_df
    if(return_dict['clf']):
        if(type(model['model'])==str):
            clf = model_dict[model['model']]
        else: 
            clf = model['model']
        print(f'>>> Training the final classifier {clf}')
        #print('>>> Classifier is : ' , clf)
        clf.fit(x,y)
        ret['clf'] = clf
        print("[DONE] Classifier is trained | acces via ['clf'] in the response")

    if(return_dict['acc']): ret['acc'] = accuracy_score(res_df['true_class'] , res_df['pred_class'])
    if(return_dict['precision']): ret['precision'] = precision_score(res_df['true_class'] , res_df['pred_class'] , average='macro')
    if(return_dict['recall']): ret['recall'] = recall_score(res_df['true_class'] , res_df['pred_class'],  average='macro')
    
    if(return_dict['pr_score']):
        pres = np.diag(confusion_matrix(res_df['true_class'] , res_df['pred_class'], normalize='true' , labels=y.unique()))
        recall = np.diag(confusion_matrix(res_df['true_class'] , res_df['pred_class'], normalize='pred' , labels=y.unique()))
        pr = pd.DataFrame({
            'class' : y.unique() , 
            'precision' : pres , 
            'recall' : recall
        })
        ret['pr_score'] = pr
    # mem_table = pd.concat([el[1] for el in res_a])#.reset_index(drop=True)
    # display(mem_table)
    # #mem_table.insert(0 , 'name' , x_index)
    # mem_table = mem_table.set_index('name')
    mem_columns = [f'prob_{el}' for el in classes]
    if(return_dict['roc_auc_score']):
        ra_score = roc_auc_score(res_df['true_class'] , res_df[mem_columns] , multi_class='ovr' , average = 'weighted') 
        ret['roc-auc'] = ra_score
    return ret

def get_score(arr , k=-1,confidance=0 , sc_average = 'weighted'):
    if(len(arr)==1):
        rdata = arr[0]
    else:
        data , model = arr[0] , arr[1]
        if(k==-1):
            rdata = pd.read_csv(f'validation_res/{data}_{model}_loo.csv')
        else:
            rdata = pd.read_csv(f'validation_res/{data}_{model}_{k}_fold.csv')
    #display(rdata)
    
    #rdata = rdata[rdata['class'].isin(classes)]
    y_total = rdata['true_class'].value_counts().to_dict()
    #print(y_total)
    pred_min = rdata['pred_prob'].min()
    pred_prob = rdata['pred_prob']
    rdata= rdata[rdata['pred_prob']>confidance]
    y_true = rdata['true_class']
    y_pred = rdata['pred_class']
    y_true_count = y_true.value_counts().to_dict()
    y_pred_count = y_pred.value_counts().to_dict()
    #print(y_pred_count)
    #labels = y_true.unique()
    #rint(labels)
    labels = np.sort(y_true.unique())
    xticks , yticks = [] , []
    for l in labels:
        try:
            yticks.append(f'{l}\n{y_true_count[l]}/{y_total[l]}')
            #xticks.append(f'{l}\n{y_pred_count[l]}')
        except : 
            yticks.append(f'{l}\n{0}/{y_total[l]}')
            #xticks.append(f'{l}\n{0}')
    #print(labels)
    from sklearn.metrics import accuracy_score , balanced_accuracy_score , precision_score , f1_score , recall_score , roc_auc_score , matthews_corrcoef 
    cm =  confusion_matrix(y_true , y_pred , normalize='true' , labels = labels)
    #f1 = recall_score(y_true , y_pred , average=None , )
    num_src = y_pred.value_counts().to_frame()
    score_dict = {
        'classes' : labels ,
        'num_src' : num_src , 
        'avg_scores': {
            'balanced_accuracy' : balanced_accuracy_score(y_true , y_pred ) , 
            'accuracy' : accuracy_score(y_true , y_pred , ) , 
            'precision' : precision_score(y_true , y_pred , average=sc_average) , 
            'recall' : recall_score(y_true , y_pred , average=sc_average) , 
            'f1' : f1_score(y_true , y_pred , average=sc_average)
        } , 
        #'roc_auc' : roc_auc_score(y_true , pred_prob , average = 'micro' , multi_class='ovr') ,
        'mcc' : matthews_corrcoef(y_true , y_pred),
        'class_scores' : pd.DataFrame({
            'class' : labels , 
            'recall_score' : recall_score(y_true , y_pred , average=None , ) , 
            'precision_score' : precision_score(y_true , y_pred , average=None , ) ,
            'f1_score' : f1_score(y_true , y_pred , average=None , )
        }).sort_values(by='class').set_index('class') , 
    }
    return score_dict

def norm_prob(arr):
    norm_arr = [el/sum(el) for el in arr]
    return norm_arr

def softmax(arr):
    print(arr.shape)
    norm_arr = []
    for el in arr:
        exp = np.power(el ,1)
        exp = exp / sum(exp)
        norm_arr.append(exp)
    #print(exp)
    norm_arr = np.asarray(norm_arr)
    return norm_arr
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold





from sklearn.metrics import f1_score, recall_score , confusion_matrix , precision_score , accuracy_score , balanced_accuracy_score , roc_auc_score
from matplotlib import pyplot as plt
import seaborn as sns
plot_dict_def = {
    'title' : True , 
    'font_scale' : 1.0 , 
    'cbar' : False ,
    'plot_num' : False,
}
all_cl = ['AGN' ,'STAR' , 'YSO' , 'PULSAR'  , 'CV' , 'LMXB' , 'HMXB' ,'ULX'] 







cl = [ 'STAR' , 'AGN', 'YSO' , 'HMXB' ,'LMXB' , 'CV' ,'ULX' , 'PULSAR' ]


class make_model():
    def __init__(self , name , clf , gamma ,x ,y):
        self.name = name 
        self.clf = clf 
        self.gamma = gamma 
        self.x = x 
        self.y = y 
        
    def validate(self , fname= '' , k=10 , normalize_prob=0 , score_average = 'macro'):
        from utilities import simple_cv
        #self.weight = self.calc_weight(self.gamma ,self.y)
        res = simple_cv(self.x,self.y , model=self.clf , k=k , normalize_prob=normalize_prob , score_average = score_average)
        res['gamma'] = self.gamma 
        #res['class_weight'] = calc_weight(slef)
        print(res['class_scores'].to_markdown())
        self.result = res
        if(fname):
            import joblib
            joblib.dump(res , fname)
        return self
    
    def train(self):
        clf = self.clf
        clf.fit(self.x , self.y)
        return self
    def save(self , fname):
        import joblib
        joblib.dump(self , fname)










def calc_score(y_true  , y_pred):
    from sklearn.metrics import precision_score , recall_score , f1_score , roc_auc_score 
    labels = np.sort(np.unique(y_true))
    #print(labels)
    pre = precision_score(y_true , y_pred , average=None , labels=labels)
    rec = recall_score(y_true , y_pred , average=None , labels=labels)
    f1 = f1_score(y_true , y_pred , average=None , labels=labels)
    df = pd.DataFrame({
        'class' : labels , 
        'precision' : pre , 
        'recall' : rec , 
        'f1_score' : f1 , 
    }).set_index('class')
    return(df)


def permute_feature(arr):
    model , x ,y , index = arr
    train_ix , test_ix = index[0] , index[1]
    x_train , x_test = x.loc[train_ix , : ] , x.loc[test_ix, :]
    y_train , y_test = y.loc[train_ix] , y.loc[test_ix]
    #display(x_train.head(10) , y_train.head(10))
    oversampler  = SMOTE(k_neighbors=4)
    x_train_up , y_train_up = oversampling(oversampler , x_train, y_train)
    x_train_up = x_train_up.replace(np.nan , -100)
    #x_test_up = x_test_up.replace(np.nan , -100)
    if(type(model['model'])==str):
        clf = model_dict[model['model']]
    else:
        clf = model['model']
    clf.fit(x_train_up , y_train_up)
    #clf.fit(x_train , y_train)
    df = pd.DataFrame({
        'true_class' : y_test , 
        'no_permute' : clf.predict(x_test)
    })
    i = 2
    for f in x_test.columns.to_list()[:]:
        x_test_temp = x_test.copy()
        x_test_temp[f] = np.random.permutation(x_test_temp[f])
        df.insert(i , f'{f}' , clf.predict(x_test_temp))
        i+=1
    return df



def feature_imp(data , model , k=-1 , return_score = 'recall' ,save_df = 0 ):
    d_name = data['name']
    m_name = model['name']
    if(type(data['data'])==str):
        x = data_dict[data['data']]
    else : x = data['data']
    x = x.sample(frac=1)
    x_index = x.index.to_list()
    x = x.reset_index(drop=True)
    y = x['class']
    x= x.drop(columns=['class'])

    if k==-1:
        print('Doing LeaveOneOut cross-validation')
        cv = LeaveOneOut()
    else:
        print(f'Doing {k} fold cross-validation.')
        cv = StratifiedKFold(k)# KFold(k) 
    model = model
    index = [(t,i) for t,i in cv.split(x,y)]
    arr = list(zip([model]*len(index) , [x]*len(index) , [y]*len(index) , index))
    import multiprocessing as mp 
    num_cores = mp.cpu_count()
    with mp.Pool(int(num_cores)) as pool:
        if(k==-1):
            res_a = pool.map(permute_feature, arr) 
        else: 
            res_a = pool.map(permute_feature , arr) 
    #res = [el[0] for el in res_a]
    res = pd.concat(res_a)
    if(save_df):
        res.to_csv(f'app_data/permutation_df/{d_name}-{m_name}.csv')
    true_class = res['true_class']
    res = res.drop(columns=['true_class'])
    def score(score_n):
        df = []
        for c in res.columns.to_list():
            df.append(
                calc_score(true_class, res[c])[[score_n]]
                .rename(columns={score_n:f'{c}'})
                )
        #calc_score()
        df = pd.concat(df , axis=1)
        no_permute = df['no_permute']
        for f in df.columns.to_list()[1:]:
            df[f] = no_permute - df[f]
        return df 
    ret = {}
    ret['f1_score'] = score('f1_score')
    ret['precision'] = score('precision')
    ret['recall'] = score('recall')
    return ret


