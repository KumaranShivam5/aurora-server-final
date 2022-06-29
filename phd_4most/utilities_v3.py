from tqdm import tqdm 
import numpy as np 
import pandas as pd 

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

from sklearn.model_selection import LeaveOneOut , StratifiedKFold 
from imblearn.over_sampling import SMOTE



def oversampling(method, X_train, y_train):
    X_res, y_res = method.fit_resample(X_train, y_train)
    return X_res, y_res





def train_model_leave_one_out(arr):
    """
    For a sample size of N, Performs Training on N-1 samples and returns prediction on the the test sample

    Parameters
    ----------
    arr : array
        Should contain : [model, data, label, index]
            model : sklearn classifier model which implements fit, predict and predict_proba methods
        index : array of length 2 : [training_indices , test_index]

    Returns
    -------
    [predicted_class , true_class , predicted_probability]
    """
    clf , x ,y , index = arr
    train_index , test_index = index[0] , index[1]
    x_train , x_test = x.loc[train_index , : ] , x.loc[test_index, :]
    y_train , y_test = y.loc[train_index] , y.loc[test_index]

    # Oversample the training set
    oversampler  = SMOTE(k_neighbors=4)
    x_train_up , y_train_up = oversampling(oversampler , x_train, y_train)
    
    #training on the upsampled data
    clf.fit(x_train_up , y_train_up)
    return [clf.predict(x_test)[0], y_test , clf.predict_proba(x_test)]

def train_model_k_fold(arr):
    """
    For a sample size of N, and given indices of train and validation data, performs training on the train-data and does predictions on the validation data

    Parameters
    ----------
    arr : array
        Should contain : [model, data, label, index]
            model : sklearn classifier model which implements fit, predict and predict_proba methods
        index : array of length 2 : [training_indices , test_indices]

    Returns
    -------
    df : dataframe
        columns : 
            true_class : true class
            pred_class : predicted class
            pred_prob : membership probability for the predicted class
            prob_<class> : membership probability of all classes
    """
    model , x ,y , index = arr
    train_index , test_index  , = index[0] , index[1]
    x_train , x_test = x.iloc[train_index , : ] , x.iloc[test_index, :]
    y_train , y_test = y.iloc[train_index] , y.iloc[test_index]
    oversampler  = SMOTE(k_neighbors=4)
    x_train_up , y_train_up = oversampling(oversampler , x_train, y_train)
    clf = model
    clf.fit(x_train_up , y_train_up)
    df = pd.DataFrame({
        'name' : x_test.index.to_list() , 
        'true_class' : y_test , 
        'pred_class' : clf.predict(x_test) , 
        'pred_prob' : [np.amax(el) for el in clf.predict_proba(x_test)]
    }).set_index('name')
    membership_table = pd.DataFrame(clf.predict_proba(x_test) , columns=[f'prob_{el}' for el in clf.classes_])
    membership_table.insert(0 , 'name' , x_test.index.to_list())
    membership_table = membership_table.set_index('name')
    df = pd.merge(df , membership_table , left_index=True , right_index=True)
    return df



def cumulative_cross_validation(x ,y , model , k_fold=-1 , multiprocessing = True ):
    """
    Performs a cumulative cross validation 
    In standard K-fold of Leave one out cross validation, 
    model scores are calculated on each fold and then the average of scores are reported. 
    In this custom version of validation, we accumulate the predictions from each folds, and then calculate the model scores.

    Parameters
    ----------
    x : Pandas Dataframe
        training data of size (N,M), N is number of samples, M is number of features
    y : Pandas Series
        Training Labels of size N
    k_fold : int , default=-1
        Number of folds for cross validation. -1 for leave one out cross vlidation
    multiprocessing : Boolean, default=True
        Select if cross validation is performed with multiprocessing or not.
    """

    # Selection of Cross validation method, based on k_fold value
    if k_fold==-1:
        print('[INFO] >>> Doing LeaveOneOut cross-validation')
        cv = LeaveOneOut()
    else:
        print(f'[INFO] >>> Doing {k_fold} fold cross-validation')
        cv = StratifiedKFold(k_fold)# KFold(k) 

    # Using CV , split indices for training and validation set
    index = [(t,i) for t,i in cv.split(x,y)]
    zipped_arr = list(zip([model]*len(index) , [x]*len(index) , [y]*len(index) , index ))
    
    if(multiprocessing):
        import multiprocessing as mp 
        num_cores = mp.cpu_count()
        with mp.Pool(int(num_cores)) as pool:
            if(k_fold==-1):
                result = pool.map(train_model_leave_one_out , zipped_arr) 
            else: 
                result = pool.map(train_model_k_fold , zipped_arr) 
    else:
        result = []
        for a in tqdm(zipped_arr):
            result.append(train_model_k_fold(a))

    if k_fold==-1 :
        result_df  = pd.DataFrame({
            'name' : result.index.to_list() , 
            'true_class' : [el[1].iloc[0] for el in result] , 
            'pred_class' : [el[0] for el in result], 
            'pred_prob' : [np.amax(el[2] )for el in result]
        }).set_index('name')
    else:
        print('doing k fold in else')
        result_df = pd.concat(result, axis=0)
    return result_df


def get_score(pred_table  , confidance=0 , score_average_type = 'weighted'):
    pred_table = pred_table[pred_table['pred_prob']>confidance]
    y_true = pred_table['true_class']
    y_pred = pred_table['pred_class']
    labels = np.sort(y_true.unique())
    
    from sklearn.metrics import accuracy_score , balanced_accuracy_score , precision_score , f1_score , recall_score , roc_auc_score , matthews_corrcoef , confusion_matrix
    cm = confusion_matrix(y_true , y_pred , labels=labels )

    score_dict = {
        'class_labels' : list(labels) ,
        'confusion_matrix' : cm ,
        'overall_scores': {
            'balanced_accuracy' : balanced_accuracy_score(y_true , y_pred ) , 
            'accuracy' : accuracy_score(y_true , y_pred , ) , 
            'precision' : precision_score(y_true , y_pred , average=score_average_type) , 
            'recall' : recall_score(y_true , y_pred , average=score_average_type) , 
            'f1' : f1_score(y_true , y_pred , average=score_average_type) , 
            'mcc' : matthews_corrcoef(y_true , y_pred),
        } , 
        'class_wise_scores' : pd.DataFrame({
            'class' : labels , 
            'recall_score' : recall_score(y_true , y_pred , average=None , ) , 
            'precision_score' : precision_score(y_true , y_pred , average=None , ) ,
            'f1_score' : f1_score(y_true , y_pred , average=None , )
        }).sort_values(by='class').set_index('class') , 
    }
    return score_dict


class make_model():
    def __init__(self , name , clf , train_data ,label):
        self.name = name 
        self.clf = clf 
        self.train_data = train_data
        self.label = label
        self.validation_prediction = 'validation predictions are not stored'
        
    def validate(self , fname= '' , k=10 , normalize_prob=0 , score_average = 'macro' , save_predictions = '' , multiprocessing = True):
        validation_predictions = cumulative_cross_validation(self.train_data,self.label ,k_fold=k , model=self.clf , multiprocessing=multiprocessing)
        if(save_predictions):
            self.validation_prediction = validation_predictions
        self.validation_score = get_score(validation_predictions)
        return self

    def train(self):
        clf = self.clf
        clf.fit(self.train_data , self.label)
        return self
    def save(self , fname):
        import joblib
        joblib.dump(self , fname)