from cProfile import label
from random import randrange
from sre_compile import isstring
from turtle import Turtle
#from turtle import title
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


from sklearn.ensemble import RandomForestClassifier 
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
    #display(x_test)
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    test_names = test_names.reset_index(drop=True)
    #display(test_names)
    #display(y_test)
    #display(x_test)
    #display(x_train.head(10) , y_train.head(10))
    oversampler  = SMOTE(k_neighbors=4)
    x_train_up , y_train_up = oversampling(oversampler , x_train, y_train)
    x_train_up = x_train_up.replace(np.nan , -100)
    #x_test_up = x_test_up.replace(np.nan , -100)
    #print(f"model dict {model_dict['model']}")
    if(type(model['model'])==str):
        clf = model_dict[model['model']]
    else:
        clf = model['model']
    clf.fit(x_train_up , y_train_up)
    #clf.fit(x_train , y_train)
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
    #display(mem_table)
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
        print('Doing LeaveOneOut cross-validation')
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
                res_a = pool.map(train_model_loo , arr) 
            else: 
                res_a = pool.map(train_model_kfold , arr) 
        res = [el[0] for el in res_a]
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
        #display(res_df)
        if(save_df):
            res_df.to_csv(f'validation_res/{d_name}_{m_name}_loo.csv')
    else:
        res_df = pd.concat(res, axis=0)
        #display(res_df)
        if(save_df):
            res_df.to_csv(f'{save_df}')
    acc_sc = accuracy_score(res_df['true_class'] , res_df['pred_class'])

    print(f'Overall Accuracy : {acc_sc}')
    ret = {}
    if(return_dict['clf']):
        if(type(model['model'])==str):
            clf = model_dict[model['model']]
        else: 
            clf = model['model']
        print('>>> Training the final classifier')
        print('>>> Classifier is : ' , clf)
        clf.fit(x,y)
        ret['clf'] = clf
    if(return_dict['prob_table']):
        ret['prob_table'] = res_df
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
    ra_score = roc_auc_score(res_df['true_class'] , res_df[mem_columns] , multi_class='ovr' , average = 'weighted') 
    if(return_dict['roc_auc_score']):
        ret['roc-auc'] = ra_score
    return ret

def get_score(arr , k=-1,confidance=0):
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
    from sklearn.metrics import accuracy_score , balanced_accuracy_score , precision_score , f1_score , recall_score
    cm =  confusion_matrix(y_true , y_pred , normalize='true' , labels = labels)
    #f1 = recall_score(y_true , y_pred , average=None , )
    num_src = y_pred.value_counts().to_frame()
    score_dict = {
        'classes' : labels ,
        'num_src' : num_src , 
        'balanced_accuracy' : balanced_accuracy_score(y_true , y_pred ) , 
        'accuracy' : accuracy_score(y_true , y_pred , ) , 
        'precision' : precision_score(y_true , y_pred , average='weighted') , 
        'recall' : recall_score(y_true , y_pred , average='weighted') , 
        'f1' : f1_score(y_true , y_pred , average='weighted') , 
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
def simple_cv(x,y,model , k=10 , normalize_prob=0):
    x_col = x.columns.to_list()
    #display(x)
    #display(y.to_frame())
    df = pd.merge(x,y.to_frame() , left_index=True , right_index=True)
    df = df.sample(frac=1)
    x = df[x_col] 
    y = df['class']
    x = x.reset_index(drop=True) 
    y = y.reset_index(drop=True)
    cv_split = StratifiedKFold(k)
    i=0
    df_all = []
    for train,test in (cv_split.split(x,y)):
        i+=1
        print('----------------------------------------------------------')
        print(f'GOING for {i} / {k} Iteration FOLD')
        print('___________________________________________________________')

        x_train , x_test = x.loc[train , :] , x.loc[test , :]
        y_train , y_test = y.loc[train] , y.loc[test]
        model_temp = model
        model_temp.fit(x_train , y_train)
        
        if(normalize_prob):
            prob = norm_prob(model_temp.predict_proba(x_test))
        else:
            prob = model_temp.predict_proba(x_test)
        df = pd.DataFrame({
            'true_class' : y_test , 
            'pred_class' : model_temp.predict(x_test) , 
            'pred_prob' : [np.amax(el) for el in prob]
            })  
        df_all.append(df)
    #score_dict = get_score([df])
    #score_dict['res_table'] = df  
    df = pd.concat(df_all)  
    score = get_score([df])
    score['res_table'] = df 
    model_temp = model 
    model_temp.fit(x,y)
    score['clf'] = model
    return score 





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





def plot_cf(dataset , classes = all_cl , k=-1 , ax='self' ,confidance=0 ,save=False, label  = '',plot_dict = plot_dict_def):
    #sns.set(font_scale = plot_dict['font_scale'])
    
    sns.set(font_scale=plot_dict['font_scale'], rc={'axes.facecolor':'white', 'figure.facecolor':'white' , 'axes.grid':True} , style="ticks")
    if(len(dataset)==1):
        rdata  = dataset[0]
        data , model = '' , ''
    else:
        data , model = dataset[0] , dataset[1]
        if(k==-1):
            rdata = pd.read_csv(f'validation_res/{data}_{model}_loo.csv')
        else:
            rdata = pd.read_csv(f'validation_res/{data}_{model}_{k}_fold.csv')
    #rdata = rdata[rdata['class'].isin(classes)]
    #display(rdata)
    y_total = rdata['true_class'].value_counts().to_dict()
    #print(y_total)
    pred_min = rdata['pred_prob'].min()
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
    
    #print(labels)
    cm =  confusion_matrix(y_true , y_pred , normalize='true' , labels = labels)
    temp_df = pd.DataFrame({
        'class': labels,
        'diag' : np.diag(cm)
    })
    temp_df = temp_df.sort_values(by='diag' , ascending=False)
    labels = temp_df['class']
    labels = ['AGN' ,'STAR' ,'YSO', 'HMXB' , 'LMXB','ULX','CV', 'PULSAR']
    cm =  confusion_matrix(y_true , y_pred , normalize='true' , labels = labels)
    for l in labels:
        try:
            yticks.append(f'{l}\n{y_true_count[l]}/{y_total[l]}')
            #xticks.append(f'{l}\n{y_pred_count[l]}')
        except : 
            yticks.append(f'{l}\n{0}/{y_total[l]}')
            #xticks.append(f'{l}\n{0}')
    if(ax=='self'):
        fig , ax = plt.subplots(nrows=1 , ncols=1 , figsize = (7,6))
    acc = accuracy_score(y_true , y_pred)
    cmap = 'rocket_r'
    if(plot_dict['plot_num']):
        sns.heatmap(cm*100 , annot=True , ax=ax ,xticklabels=labels , yticklabels=yticks , fmt='.1f' , cbar = plot_dict['cbar'] ,cmap=cmap )
    else:
        sns.heatmap(cm*100 , annot=True , ax=ax ,xticklabels=labels , yticklabels=labels , fmt='.1f' , cbar = plot_dict['cbar'] ,cmap= cmap )
    #print(confidance , rdata['pred_prob'].min())
    if(plot_dict['title']):
        if(confidance < pred_min):
            ax.set_title(f'Data : {data} | Model : {model} \n Prob Threshold : Max prob | Accuracy : {acc*100:.1f}')
            ax.set_title(f'{label} | Prob Threshold : Max prob | Accuracy : {acc*100:.1f}')
        else :
            #print('in else')
            
            ax.set_title(label+f'Data : {data} | Model : {model} \n Prob Threshold {confidance} | Accuracy : {acc*100:.1f}')
            #print(label)
            ax.set_title(f'{label} | Prob Threshold {confidance} | Accuracy : {acc*100:.1f}')
            #ax.set_title(f'Prob Threshold {confidance} | Accuracy : {acc*100:.1f}')
    #ax.set_title(f'Accuracy : {acc*100:.1f}')
    ax.set_ylabel('True class')
    ax.set_xlabel('Predicted class')
    #ax.set_yticks(rotation=0)
    ax.tick_params(axis='y', rotation=0)
    if save:
        plt.savefig(f'{save}/{data}_{model}.jpg')




def tune_res_plot(tune_res , tune_param):

    acc = [el[0] for el in tune_res]
    pres = pd.concat([el[1][['class' , 'precision']].set_index('class').T for el in tune_res]).reset_index(drop=True)
    pres.insert(0 , tune_param['name'], tune_param['val'])
    pres.insert(1 , 'Overall', acc)
    pres_melt = pres.melt(id_vars=tune_param['name'] , value_name='accuracy' , var_name = 'class')
    sns.set(font_scale=1.1, rc={'axes.facecolor':'white', 'figure.facecolor':'white' , 'axes.grid':True} , style="ticks")
    sns.relplot(
        data = pres_melt, 
        hue = 'class' ,
        style='class' ,
        x = tune_param['name'] , 
        y = 'accuracy' ,
        kind='line' , 
        palette = 'icefire' 
    )
    plt.grid(False)
    #plt.show()
    return pres



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



def take_df_mean(arr):
    arr_ret = np.asmatrix(arr[0])
    col_names = arr[0].columns.to_list()
    index_names = arr[0].index.to_list()
    l = len(arr)
    df_i , df_j = arr[0].shape
    mean_mat , std_mat = [] , []
    for i in range(df_i):
        temp_mean_arr = []
        temp_std_arr = []
        for j in range(df_j):
            temp_mean_arr.append(np.mean([el.iloc[i][j] for el in arr]))
            temp_std_arr.append(np.std([el.iloc[i][j] for el in arr]))
        mean_mat.append(temp_mean_arr)
        std_mat.append(temp_std_arr)
    mean_df = pd.DataFrame(mean_mat , index = index_names , columns=col_names)
    std_df = pd.DataFrame(std_mat , index = index_names , columns=col_names)
    return mean_df , std_df

