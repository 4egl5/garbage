import numpy as np
import pandas as pd 
from collections import Counter
import math
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier


def preprocess_dataset(dataset_path):
    numcols = ['Overall', 'Crossing','Finishing',  'ShortPassing',  'Dribbling','LongPassing', 'BallControl', 'Acceleration','SprintSpeed', 'Agility',  'Stamina','Volleys','FKAccuracy','Reactions','Balance','ShotPower','Strength','LongShots','Aggression','Interceptions']
    catcols = ['Preferred Foot','Position','Body Type','Nationality','Weak Foot']
    
    player_df = pd.read_csv(dataset_path)[numcols+catcols]
    traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])],axis=1).dropna()
    features = traindf.columns
    traindf = pd.DataFrame(traindf,columns=features)
    y = traindf['Overall']>=87
    X = traindf.copy()
    del X['Overall']
    return X, y

def cor_selector(X, y,num_feats):
    # Calculate Pearson correlation between X and y
    # np.corrcoef: return Pearson correlation
    cor_list = [np.corrcoef(X[i],y)[0,1] for i in X.columns.tolist()]
    
    # replace nan with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    
    # sort the list by the pearson correlation value in ascending order, and choose num_features from the largest
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    
    # Create True False Table to show if the feature_name is in cor_feature
    cor_support = [True if i in cor_feature else False for i in list(X.columns)]
    
    # Return results
    return cor_support, cor_feature


def chi_squared_selector(X, y, num_feats):
    # SelectKBest(): Select features by highest score
    # MinMaxScaler(): normalize feature by scaling each feature to a given range
    # fit_transform(): fit the data
    # get_support(): return selected features in True False array
    chi_support = SelectKBest(chi2, k = num_feats).fit(MinMaxScaler().fit_transform(X), y).get_support()

    # add the columns name to chi_feature if it is True in chi_support 
    chi_feature = X.loc[:,chi_support].columns.tolist()
    return chi_support, chi_feature


def rfe_selector(X, y, num_feats):
    rfe_support = RFE(
        estimator=LogisticRegression(),
        n_features_to_select=num_feats,
        step=10,
        # verbose = 0
    ).fit(MinMaxScaler().fit_transform(X),y).get_support()
    
    rfe_feature = X.loc[:,rfe_support].columns.tolist()
    return rfe_support, rfe_feature


def embedded_log_reg_selector(X, y, num_feats):
    embedded_lr_support = SelectFromModel(
        LogisticRegression(penalty='l1',solver = 'liblinear',max_iter = 1000),
        max_features = num_feats
    ).fit(MinMaxScaler().fit_transform(X), y).get_support()
    
    embedded_lr_feature = X.loc[:,embedded_lr_support].columns.tolist()
    return embedded_lr_support, embedded_lr_feature


def embedded_rf_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    embedded_rf_support = SelectFromModel(
        RandomForestClassifier(	n_estimators=100),
        max_features = num_feats
    ).fit(MinMaxScaler().fit_transform(X), y).get_support()
    
    embedded_rf_feature = X.loc[:,embedded_rf_support].columns.tolist()
    
    # Your code ends here
    return embedded_rf_support, embedded_rf_feature


def embedded_lgbm_selector(X, y, num_feats):
    embedded_lgbm_support = SelectFromModel(
        LGBMClassifier(
            n_estimators=500,
            learning_rate = 0.05,
            num_leaves=32,
            colsample_bytree=0.2,
            reg_alpha=3,
            reg_lambda=1,
            min_split_gain=0.01,
            min_child_weight=40,
            verbose = -1
        ),max_features=num_feats
    ).fit(X,y).get_support()
    embedded_lgbm_feature = X.loc[:,embedded_lgbm_support].columns.tolist()
    return embedded_lgbm_support, embedded_lgbm_feature



def autoFeatureSelector(dataset_path, num_feats, methods=[]):
    # Parameters
    # data - dataset to be analyzed (csv file)
    # methods - various feature selection methods we outlined before, use them all here (list)
    
    # preprocessing
    X, y = preprocess_dataset(dataset_path)
    feature_name = list(X.columns)
    
    # Run every method we outlined above from the methods list and collect returned best features from every method
    if 'pearson' in methods:
        cor_support, cor_feature = cor_selector(X, y,num_feats)
    if 'chi-square' in methods:
        chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
    if 'rfe' in methods:
        rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
    if 'log-reg' in methods:
        embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
    if 'rf' in methods:
        embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
    if 'lgbm' in methods:
        embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
    
    # Combine all the above feature list and count the maximum set of features that got selected by all methods
    #### Your Code starts here (Multiple lines)
    feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embedded_lr_support,
                                    'Random Forest':embedded_rf_support, 'LightGBM':embedded_lgbm_support})
    feature_selection_df['Total'] = np.sum(feature_selection_df == True, axis=1)
    feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
    #### Your Code ends here
    return feature_selection_df.Feature[:num_feats].to_list()

best_features = autoFeatureSelector(dataset_path="data/fifa19.csv", methods=['pearson', 'chi-square', 'rfe', 'log-reg', 'rf', 'lgbm'],num_feats = 30)
print(best_features)
