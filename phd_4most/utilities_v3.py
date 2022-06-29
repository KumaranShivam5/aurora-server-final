from tqdm import tqdm 
import numpy as np 
import pandas as pd 

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier



model_dict = {
    'RF' : RandomForestClassifier() , 
    'RF_mod' : RandomForestClassifier(
        n_estimators = 500 , 
    ) , 
    'GB' : GradientBoostingClassifier() , 
}
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

def cumulative_cross_validation(x ,y , model , k=-1 , return_dict  = ret_dict , save_df = '' , multiprocessing = 1 ):
    classes = y.unique()
    x = x.drop(columns=['class'])

    if k==-1:
        print('[INFO] Doing LeaveOneOut cross-validation')
        cv = LeaveOneOut()
    else:
        print(f'Doing {k} fold cross-validation')
        cv = StratifiedKFold(k)# KFold(k) 
    model = model
    index = [(t,i) for t,i in cv.split(x,y)]
    arr = list(zip([model]*len(index) , [x]*len(index) , [y]*len(index) , index ))
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




class make_model():
    def __init__(self , name , clf , gamma ,x ,y):
        self.name = name 
        self.clf = clf 
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