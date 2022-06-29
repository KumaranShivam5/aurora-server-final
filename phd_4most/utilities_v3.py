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
    # X_train.replace(np.nan, exnum, inplace=True)
    X_res, y_res = method.fit_resample(X_train, y_train)
    # res = X_res.values
    # X_res[:] = res
    return X_res, y_res

from imblearn.over_sampling import SMOTE



def train_model_leave_one_out(arr):
    model , x ,y , index = arr
    train_ix , test_ix = index[0] , index[1]
    x_train , x_test = x.loc[train_ix , : ] , x.loc[test_ix, :]
    y_train , y_test = y.loc[train_ix] , y.loc[test_ix]
    #display(x_train.head(10) , y_train.head(10))
    oversampler  = SMOTE(k_neighbors=4)
    x_train_up , y_train_up = oversampling(oversampler , x_train, y_train)
    x_train_up = x_train_up.replace(np.nan , -100)
    #x_test_up = x_test_up.replace(np.nan , -100)
    clf = model
    clf.fit(x_train_up , y_train_up)
    return [clf.predict(x_test)[0], y_test , clf.predict_proba(x_test)]

def train_model_k_fold(arr):
    model , x ,y , index = arr
    train_ix , test_ix  , = index[0] , index[1]
    x_train , x_test = x.iloc[train_ix , : ] , x.iloc[test_ix, :]
    # test_names = names.loc[test_ix]
    y_train , y_test = y.iloc[train_ix] , y.iloc[test_ix]
    # x_test = x_test.reset_index(drop=True)
    # y_test = y_test.reset_index(drop=True)
    # test_names = test_names.reset_index(drop=True)
    oversampler  = SMOTE(k_neighbors=4)
    x_train_up , y_train_up = oversampling(oversampler , x_train, y_train)
    x_train_up = x_train_up.replace(np.nan , -100)
    clf = model
    clf.fit(x_train_up , y_train_up)
    df = pd.DataFrame({
        'name' : x_test.index.to_list() , 
        'true_class' : y_test , 
        'pred_class' : clf.predict(x_test) , 
        'pred_prob' : [np.amax(el) for el in clf.predict_proba(x_test)]
    }).set_index('name')
    # mem_table = pd.DataFrame(clf.predict_proba(x_test) , columns=[f'prob_{el}' for el in clf.classes_])
    # mem_table.insert(0 , 'name' , test_names)
    # mem_table = mem_table.set_index('name')
    # df = pd.merge(df , mem_table , left_index=True , right_index=True)
    return df



from sklearn.model_selection import LeaveOneOut , StratifiedKFold 
def cumulative_cross_validation(x ,y , model , k_fold=-1 , save_result_filename = '' , multiprocessing = 1 ):

    if k_fold==-1:
        print('[INFO] >>> Doing LeaveOneOut cross-validation')
        cv = LeaveOneOut()
    else:
        print(f'[INFO] >>> Doing {k_fold} fold cross-validation')
        cv = StratifiedKFold(k_fold)# KFold(k) 
    model = model
    index = [(t,i) for t,i in cv.split(x,y)]
    arr = list(zip([model]*len(index) , [x]*len(index) , [y]*len(index) , index ))
    if(multiprocessing):
        import multiprocessing as mp 
        num_cores = mp.cpu_count()
        with mp.Pool(int(num_cores)) as pool:
            if(k_fold==-1):
                res = pool.map(train_model_leave_one_out , arr) 
            else: 
                res = pool.map(train_model_k_fold , arr) 
    else:
        res = []
        for a in tqdm(arr):
            res.append(train_model_k_fold(a))

    if k_fold==-1 :
        res_df  = pd.DataFrame({
            'name' : res.index.to_list() , 
            'true_class' : [el[1].iloc[0] for el in res] , 
            'pred_class' : [el[0] for el in res], 
            'pred_prob' : [np.amax(el[2] )for el in res]
        }).set_index('name')
        if(save_result_filename):
            res_df.to_csv(f'{save_result_filename}')
    else:
        print('doing k fold in else')
        res_df = pd.concat(res, axis=0)
        #display(res_df)
        if(save_result_filename):
            res_df.to_csv(f'{save_result_filename}')
    return res_df


def get_score(y_true , y_pred , pred_prob , confidance=0 , sc_average = 'weighted'):
    y_true = y_true[pred_prob>confidance]
    y_pred = y_pred[pred_prob > confidance]
    pred_prob = pred_prob[pred_prob>confidance]
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
    from sklearn.metrics import accuracy_score , balanced_accuracy_score , precision_score , f1_score , recall_score , roc_auc_score , matthews_corrcoef , confusion_matrix
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


class make_model():
    def __init__(self , name , clf , x ,y):
        self.name = name 
        self.clf = clf 
        self.x = x 
        self.y = y 
        self.validation_prediction = 'validation predictions are not stored'
        
    def validate(self , fname= '' , k=10 , normalize_prob=0 , score_average = 'macro' , save_predictions = '' , multiprocessing = True):
        #from utilities import simple_cv
        #self.weight = self.calc_weight(self.gamma ,self.y)
        validation_predictions = cumulative_cross_validation(self.x,self.y ,k_fold=k , model=self.clf , multiprocessing=True)
        if(save_predictions):
            self.validation_prediction = validation_predictions
        # if(fname):
        #     import joblib
        #     joblib.dump(res , fname)
        return self
    
    def train(self):
        clf = self.clf
        clf.fit(self.x , self.y)
        return self
    def save(self , fname):
        import joblib
        joblib.dump(self , fname)