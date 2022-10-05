import pickle
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score
import os
# from SHAP_age_new23 import SHAP_Age
from SHAP_age_exponential import SHAP_Age
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import GridSearchCV
import xgboost

parser = argparse.ArgumentParser()
parser.add_argument('--path', dest="path")
parser.add_argument('--dataset', dest='dataset', help='choose from NHANES, biobank, NHANES_small, or biobank_small')
parser.add_argument('--transfer', dest="transfer", default = None, type=int, help='whether transfer to another datset')
parser.add_argument('--target_data_path', dest="target_data_path", default = None, help='the path of target dataset')
parser.add_argument('--save_folder', dest="save_folder", help='the path of saved SHAP_Age object')
parser.add_argument('--task', dest="task", help='mortality cause')
args = parser.parse_args()

def label(alive_year, mortstat, year):
    if alive_year > year:
        return 0
    else:
        if mortstat == 1:
            return 1
        else:
            return 2
        
model_path = args.path+'/model.pickle.dat'
model_train = pickle.load(open(model_path, "rb"))
if (args.dataset == 'NHANES') or (args.dataset == 'NHANES_small') or (args.dataset == 'biobank_small'):
    age_feature_name = 'Demographics_Age'
    gender_feature_name = 'Demographics_Gender_2.0'
elif args.dataset == 'biobank':
    age_feature_name = 'Age'
    # age_feature_name = '21003-0.0'
    gender_feature_name = '31-0.0'
    
else:
    print('Unsupported dataset')
    exit()


##### load model and data #####********************************************************************************************************
model_path = args.path+'/model.pickle.dat'
model_train = pickle.load(open(model_path, "rb"))

if args.dataset == 'biobank':
    features = pd.read_csv('/projects/leelab/nobackup/wqiu/UK_Biobank_genetic_data/pheno_data/features_initial_preprocessing_missforest_imputed_no_missing_lancet_and_meaningful_adjusted_assays_remove20002and20004_AgeAdjusted_CancerAdjusted_geo.csv')
#     columns = pd.read_csv('../uk_biobank/features_initial_preprocessing_feature_selection_missforest_imputed_no_missing_lancet_and_meaningful_adjusted_assays_remove20002and20004.csv', nrows=1).columns
#     features = features[columns]
    label_df = pd.read_csv('/projects/leelab/nobackup/wqiu/UK_Biobank_genetic_data/pheno_data/death_label_new.csv')
    data_mortality = pd.merge(features, label_df[['eid', 'alive_year', args.task, 'external']], how='left', on='eid')
    X = data_mortality
    X = X[(X[age_feature_name]>=39.5) & (X[age_feature_name]<=70.5)].reset_index(drop=True)  
#     X = X[(X['external']!=1)]
    mortstat = X[args.task]
    permth_int = X['alive_year']
    y = permth_int * (mortstat - .5)*2
    eid = X['eid']
    X = X.drop([args.task, 'external', 'alive_year', 'eid', '21003-0.0'], axis=1)
    print(X.shape)
elif args.dataset == 'biobank_small':
    X = pd.read_csv('../UK_Biobank_51_features.csv')    
    X = X[(X[age_feature_name]>=39.5) & (X[age_feature_name]<=70.5)].reset_index(drop=True)
    X[str(year_num)+'_year_label'] = X.apply(lambda x: label(x['alive_year'], x[args.task], int(year_num)), axis=1)    
    X = X[(X['external']!=1) | (X['alive_year'] > year_num)]
    X = X[X[str(year_num)+'_year_label']!=2]
    y = X[str(year_num)+'_year_label']
    print(y.value_counts())
    mortstat = X[args.task]
    permth_int = X['alive_year']
    X = X.drop([str(year_num)+'_year_label', 'eid', 'flag', 'alive_year', 'all-cause', 'neoplasms', 'circulatory', 'respiratory', 'digestive', 'external', 'other'], axis=1)
    X = X[model_train.get_booster().feature_names]
    print(X.shape)
elif (args.dataset == 'NHANES_small') and args.transfer:
    X = pd.read_csv('/projects/leelab2/wqiu/NHANES/data/data_460_classification_imputed_missforest_feature_selection.csv')
    mortality = pd.read_csv('/projects/leelab2/wqiu/NHANES/data/mortality_label.csv')
    fea_list = pd.read_csv('../NHANES_feature_list.csv')
    nominal_fea = fea_list[fea_list['Nominal']==1]['Type_Short_Name'].tolist()
    nominal_fea = list(set(nominal_fea) & set(X.columns))
    X = pd.get_dummies(X, columns=nominal_fea, drop_first=True)
    X[args.task] = mortality[args.task]
    X[str(year_num)+'_year_label'] = X.apply(lambda x: label(x['permth_int']/12, x[args.task], int(year_num)), axis=1)    
    X = X[(mortality['external']!=1) | (X['permth_int'] > year_num)]
    X = X[X[str(year_num)+'_year_label']!=2]
    X = X[(X['Demographics_Age']>=40) & (X['Demographics_Age']<=70)]
    y = X[str(year_num)+'_year_label']
    print(y.value_counts())
    X = X[model_train.get_booster().feature_names]
    print(X.shape)
elif (args.dataset == 'NHANES'):
    X = pd.read_csv('/projects/leelab2/wqiu/NHANES/data/data_460_classification_imputed_missforest_feature_selection.csv')
    mortality = pd.read_csv('/projects/leelab2/wqiu/NHANES/data/mortality_label_causes.csv')
    fea_list = pd.read_csv('./NHANES_feature_list.csv')
    nominal_fea = fea_list[fea_list['Nominal']==1]['Type_Short_Name'].tolist()
    nominal_fea = list(set(nominal_fea) & set(X.columns))
    X = pd.get_dummies(X, columns=nominal_fea, drop_first=True)
    X['cause'] = mortality['ucod_leading']
    SEQN = mortality['SEQN']
#     X = X[(X['cause']!='4.0')]
#     X = X[(X['mortstat']==0) | (X['cause']==str(args.cause)+'.0')]
    y_mort_status = (X['cause'] == str(args.task)+'.0').astype('int')
    print(y_mort_status.value_counts())
    y_years = X["permth_int"]
    y = y_years * (y_mort_status - .5)*2
    X = X[model_train.get_booster().feature_names]
    print(X.shape)
else:
    print('Please check the dataset and transfer parameters')
    exit()
    
print('# samples: ', X.shape[0])
print('# features: ', X.shape[1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

shap_age_obj_female = pickle.load(open(args.path+'/different_age_background/'+'female'+'/SHAP_age_exponential/shap_age_object.pkl', "rb"))
shap_age_obj_male = pickle.load(open(args.path+'/different_age_background/'+'male'+'/SHAP_age_exponential/shap_age_object.pkl', "rb"))

age_index = None
if args.transfer:
    print('Transfer to '+args.dataset)
    print('\n')
    print('Performance of models using original features on test set:')
    print('# test samples: ', len(X_test))

    print('Samples aged '+ str(min(shap_age_obj_female.age_list))+'-'+str(max(shap_age_obj_female.age_list))+':')
    train_age_index = X_train[(X_train[age_feature_name]>=min(shap_age_obj_female.age_list)) & (X_train[age_feature_name]<=max(shap_age_obj_female.age_list))].index
    test_age_index = X_test[(X_test[age_feature_name]>=min(shap_age_obj_female.age_list)) & (X_test[age_feature_name]<=max(shap_age_obj_female.age_list))].index
   
#     print('# samples: ', len(train_age_index)+len(test_age_index))
#     print('# positive samples: ', sum(y_train[train_age_index]==1)+sum(y_test[test_age_index]==1))
#     print('# negative samples: ', sum(y_train[train_age_index]==0)+sum(y_test[test_age_index]==0))
    
    pre = model_train.predict_proba(X_test.loc[test_age_index, :])[:, 1]
    print('# test samples: ', len(pre))
    print('AUC: ', roc_auc_score(y_test[test_age_index], pre))
    
#     print('Samples aged '+ str(min(X_train[age_feature_name]))+'-'+str(min(shap_age_obj_female.age_list))+':')
#     train_age_index = X_train[(X_train[age_feature_name]<min(shap_age_obj_female.age_list))].index
#     test_age_index = X_test[(X_test[age_feature_name]<min(shap_age_obj_female.age_list))].index
#     pre = model_train.predict_proba(X_test.loc[test_age_index, :])[:, 1]
#     print('# test samples: ', len(pre))
#     print('AUC: ', roc_auc_score(y_test[test_age_index], pre))
    
    print('Samples aged '+ str(max(shap_age_obj_female.age_list))+'-'+str(max(X_train[age_feature_name]))+':')
    train_age_index = X_train[(X_train[age_feature_name]>max(shap_age_obj_female.age_list))].index
    test_age_index = X_test[(X_test[age_feature_name]>min(shap_age_obj_female.age_list))].index
    pre = model_train.predict_proba(X_test.loc[test_age_index, :])[:, 1]
    print('# test samples: ', len(pre))
    print('AUC: ', roc_auc_score(y_test[test_age_index], pre))
    
    print('All samples')
    pre = model_train.predict_proba(X_test)[:, 1]
    print('AUC: ', roc_auc_score(y_test, pre))
    print('\n')

##### get shap age #####********************************************************************************************************
X_train['shap_age'] = -1
X_test['shap_age'] = -1
if args.dataset == 'biobank':
    X_train.loc[X_train[gender_feature_name] == 0, 'shap_age'] = shap_age_obj_female.get_shap_age(model_train.predict(X_train.loc[X_train[gender_feature_name] == 0, model_train.get_booster().feature_names], output_margin=True))
    X_train.loc[X_train[gender_feature_name] == 1, 'shap_age'] = shap_age_obj_male.get_shap_age(model_train.predict(X_train.loc[X_train[gender_feature_name] == 1, model_train.get_booster().feature_names], output_margin=True))
    X_test.loc[X_test[gender_feature_name] == 0, 'shap_age'] = shap_age_obj_female.get_shap_age(model_train.predict(X_test.loc[X_test[gender_feature_name] == 0, model_train.get_booster().feature_names], output_margin=True))
    X_test.loc[X_test[gender_feature_name] == 1, 'shap_age'] = shap_age_obj_male.get_shap_age(model_train.predict(X_test.loc[X_test[gender_feature_name] == 1, model_train.get_booster().feature_names], output_margin=True))
else:
    X_train.loc[X_train[gender_feature_name] == 1, 'shap_age'] = shap_age_obj_female.get_shap_age(model_train.predict(X_train.loc[X_train[gender_feature_name] == 1, model_train.get_booster().feature_names], output_margin=True))
    X_train.loc[X_train[gender_feature_name] == 0, 'shap_age'] = shap_age_obj_male.get_shap_age(model_train.predict(X_train.loc[X_train[gender_feature_name] == 0, model_train.get_booster().feature_names], output_margin=True))
    X_test.loc[X_test[gender_feature_name] == 1, 'shap_age'] = shap_age_obj_female.get_shap_age(model_train.predict(X_test.loc[X_test[gender_feature_name] == 1, model_train.get_booster().feature_names], output_margin=True))
    X_test.loc[X_test[gender_feature_name] == 0, 'shap_age'] = shap_age_obj_male.get_shap_age(model_train.predict(X_test.loc[X_test[gender_feature_name] == 0, model_train.get_booster().feature_names], output_margin=True))

if (args.dataset == 'biobank') or (args.dataset == 'biobank_small'):
    X.loc[X[gender_feature_name] == 0, 'shap_age'] = shap_age_obj_female.get_shap_age(model_train.predict(X.loc[X[gender_feature_name] == 0, model_train.get_booster().feature_names], output_margin=True))
    X.loc[X[gender_feature_name] == 1, 'shap_age'] = shap_age_obj_male.get_shap_age(model_train.predict(X.loc[X[gender_feature_name] == 1, model_train.get_booster().feature_names], output_margin=True))
    X['eid'] = eid
    X[['eid', age_feature_name, gender_feature_name, 'shap_age']].to_csv(args.path+'/different_age_background/'+args.task+'_SHAP_age.csv', index=False)
    
if args.dataset == 'NHANES':
    X.loc[X[gender_feature_name] == 0, 'shap_age'] = shap_age_obj_female.get_shap_age(model_train.predict(X.loc[X[gender_feature_name] == 0, model_train.get_booster().feature_names], output_margin=True))
    X.loc[X[gender_feature_name] == 1, 'shap_age'] = shap_age_obj_male.get_shap_age(model_train.predict(X.loc[X[gender_feature_name] == 1, model_train.get_booster().feature_names], output_margin=True))
    X['SEQN'] = SEQN
    X[['SEQN', age_feature_name, gender_feature_name, 'shap_age']].to_csv(args.path+'/different_age_background/'+args.task+'_SHAP_age.csv', index=False)