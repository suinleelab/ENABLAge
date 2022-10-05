import xgboost
import sys; sys.path.insert(0,"/projects/leelab3/hughchen/shap")
import shap
import matplotlib.pylab as pl
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
from sklearn.metrics import roc_auc_score, average_precision_score
import argparse
# from feature_selector import FeatureSelector
from sklearn.model_selection import GridSearchCV
import pickle
# from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from lifelines.utils.sklearn_adapter import sklearn_adapter

def c_statistic_harrell(pred, labels):
    """
    The C-statistic measures how well we can order people by 
    their survival time (1.0 is a perfect ordering).
    """
    total = 0
    matches = 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            if labels[j] > 0 and abs(labels[i]) > labels[j]:
                total += 1
                if pred[j] > pred[i]:
                    matches += 1
    return matches/total

parser = argparse.ArgumentParser()
parser.add_argument('--age', dest="age", help="age range", default='')
parser.add_argument('--mortality', dest='mortality', help='mortality type, choose from all-cause, neoplasms, circulatory, respiratory, digestive, external, other')
parser.add_argument('--feature_selection', dest='feature_selection', help='whether use feature selection')
args = parser.parse_args()
path = './result/IMPACT_Age_missforest/Linear_Cox'+'_'+args.mortality
if args.age != '':
    path += '_age_'+args.age
if args.feature_selection == '1':
    path += '_feature_selection'
path += '_remove20002and20004_small_imputed_AgeAdjusted_CancerAdjusted_test_NewLabel_FloatAge_test_GeoValidation/'
if not os.path.isdir(path):
    os.mkdir(path)
C_file = open(path+'score.txt', 'w')

random_state = 528
age_feature = 'Age'

features = pd.read_csv('/projects/leelab/nobackup/wqiu/UK_Biobank_genetic_data/pheno_data/features_initial_preprocessing_missforest_imputed_no_missing_lancet_and_meaningful_adjusted_assays_remove20002and20004_AgeAdjusted_CancerAdjusted_geo.csv')
# print(data_path, file=C_file)
label_df = pd.read_csv('/projects/leelab/nobackup/wqiu/UK_Biobank_genetic_data/pheno_data/death_label_new.csv')
if args.mortality != 'all-cause':
    data_mortality = pd.merge(features, label_df[['eid', 'alive_year', 'all-cause', args.mortality, 'external']], how='left', on='eid')
else:
    data_mortality = pd.merge(features, label_df[['eid', 'alive_year', 'all-cause', 'external']], how='left', on='eid')

X = data_mortality
X = X[(X[age_feature]>=39.5) & (X[age_feature]<=70.5)].reset_index(drop=True)  
X = X[(X['external']!=1)]
X = X[(X['all-cause']==0) | (X[args.mortality]==1)]


X_geo = X.loc[(X['54-0.0']==11004) | (X['54-0.0']==11005)].reset_index(drop=True)
X = X.loc[(X['54-0.0']!=11004) & (X['54-0.0']!=11005)].reset_index(drop=True)
mortstat = X[args.mortality]
permth_int = X['alive_year']
X = X.drop(['all-cause', args.mortality, 'external', 'eid', '54-0.0', '21003-0.0'], axis=1)
mortstat_geo = X_geo[args.mortality]
permth_int_geo = X_geo['alive_year']
X_geo = X_geo.drop(['all-cause', args.mortality, 'external', 'eid', '54-0.0', '21003-0.0'], axis=1)
X[args.mortality] = mortstat
X_geo[args.mortality] = mortstat_geo

print(X.shape)

if args.age != '':
    age_range = args.age.split('_')
    print(age_range)
    y = y[(X['Age']>=int(age_range[0])) & (X['Age']<int(age_range[1]))]
    X = X[(X['Age']>=int(age_range[0])) & (X['Age']<int(age_range[1]))]

print(X.columns)
print(X.shape)
print('# samples: ', X.shape[0])
print('# positive samples: ', sum(mortstat==1))
print('# negative samples: ', sum(mortstat==0))
print('# features: ', X.shape[1])
print('# samples: ', X.shape[0], file=C_file)
print('# positive samples: ', sum(mortstat==1), file=C_file)
print('# negative samples: ', sum(mortstat==0), file=C_file)
print('# features: ', X.shape[1], file=C_file)     

X = X.drop(['5088-0.9', '5089-0.9', '5111-0.5', '5116-0.5', '5119-0.5', 
                  # '5112-0.5', '5115-0.5', '5118-0.5', '5117-0.5',
#                 '5113-0.5', '5114-0.5', '5086-0.5', '5087-0.5', '5084-0.5', '5085-0.5',
#                 '5104-0.5', '5107-0.5', '5103-0.5', '5100-0.5', '5105-0.5', '5106-0.5',
#                 '5101-0.5', '5102-0.5', '5089-0.5', '5088-0.5', 
               ], axis=1)
X_geo = X_geo.drop(['5088-0.9', '5089-0.9', '5111-0.5', '5116-0.5', '5119-0.5', 
                  # '5112-0.5', '5115-0.5', '5118-0.5', '5117-0.5',
#                 '5113-0.5', '5114-0.5', '5086-0.5', '5087-0.5', '5084-0.5', '5085-0.5',
#                 '5104-0.5', '5107-0.5', '5103-0.5', '5100-0.5', '5105-0.5', '5106-0.5',
#                 '5101-0.5', '5102-0.5', '5089-0.5', '5088-0.5', 
               ], axis=1)
print(X.shape)
X = X.drop(['21000-0.0_3', '21000-0.0_4', '42034-0.0', '5326-0.0_2', '5327-0.0_3', '5327-0.0_2', '20001-0.0_1031', '20001-0.0_1046', '20001-0.0_1082', '20001-0.0_1033', '20001-0.0_1010', '20001-0.0_1086', '20001-0.0_1009', '20001-0.0_1027', '20001-0.0_1066', '20001-0.0_1076', '20001-0.0_1025', '20001-0.0_1067', '20001-0.0_1080', '20001-0.0_1007', '20001-0.0_1079', '20001-0.0_1037', '20001-0.0_1064', '20001-0.0_1051', '20001-0.0_1036', '20001-0.0_1075', '20001-0.0_1074', '20001-0.0_1085', '20001-0.0_1038', '20001-0.0_1016', '20001-0.0_1005', '20001-0.0_1029', '20001-0.0_1077', '20001-0.0_1081', '20001-0.0_1008', '20001-0.0_1088', '20001-0.0_1042', '20001-0.0_1012', '20001-0.0_1078', '20001-0.0_1071', '20001-0.0_1087', '20118-0.0_17', '20118-0.0_14', '20118-0.0_2', '20118-0.0_1', '20118-0.0_9', '20118-0.0_18', '20118-0.0_4', '20118-0.0_3', '40012-0.0_9', '21000-0.0_2', '5326-0.0_3', '5328-0.0_4', '20001-0.0_1058', '42024-0.0', '42036-0.0', '20118-0.0_12', '20118-0.0_16', '20118-0.0_13', '20118-0.0_11'], axis=1)

X_geo = X_geo.drop(['21000-0.0_3', '21000-0.0_4', '42034-0.0', '5326-0.0_2', '5327-0.0_3', '5327-0.0_2', '20001-0.0_1031', '20001-0.0_1046', '20001-0.0_1082', '20001-0.0_1033', '20001-0.0_1010', '20001-0.0_1086', '20001-0.0_1009', '20001-0.0_1027', '20001-0.0_1066', '20001-0.0_1076', '20001-0.0_1025', '20001-0.0_1067', '20001-0.0_1080', '20001-0.0_1007', '20001-0.0_1079', '20001-0.0_1037', '20001-0.0_1064', '20001-0.0_1051', '20001-0.0_1036', '20001-0.0_1075', '20001-0.0_1074', '20001-0.0_1085', '20001-0.0_1038', '20001-0.0_1016', '20001-0.0_1005', '20001-0.0_1029', '20001-0.0_1077', '20001-0.0_1081', '20001-0.0_1008', '20001-0.0_1088', '20001-0.0_1042', '20001-0.0_1012', '20001-0.0_1078', '20001-0.0_1071', '20001-0.0_1087', '20118-0.0_17', '20118-0.0_14', '20118-0.0_2', '20118-0.0_1', '20118-0.0_9', '20118-0.0_18', '20118-0.0_4', '20118-0.0_3', '40012-0.0_9', '21000-0.0_2', '5326-0.0_3', '5328-0.0_4', '20001-0.0_1058', '42024-0.0', '42036-0.0', '20118-0.0_12', '20118-0.0_16', '20118-0.0_13', '20118-0.0_11'], axis=1)

X_train, X_test, _, _ = train_test_split(X, X, test_size=0.2, random_state=random_state)


print('start training')
###################
### Train Model ###
###################
# CoxRegression = sklearn_adapter(CoxPHFitter, event_col=args.mortality)
# tuned_parameters=[{'penalizer':[0, 0.01, 0.1, 1], 'l1_ratio':[0, 0.01, 0.1, 1]}]
# cph_cv = GridSearchCV(CoxRegression(), param_grid=tuned_parameters, cv=3, verbose=4)
# cph_cv.fit(X_train.drop(['alive_year'], axis=1), X_train['alive_year'])
# cph = CoxPHFitter(**cph_cv.best_params_)
# print(cph_cv.cv_results_, file=C_file)
# print('best parameters:', cph_cv.best_params_, file=C_file)

cph = CoxPHFitter(penalizer=1, l1_ratio=0.1)
cph.fit(X_train, duration_col='alive_year', event_col=args.mortality)
pickle.dump(cph, open(path+"model.pickle.dat", "wb"))
pickle.dump(X_train.columns, open(path+"model_columns.pickle.dat", "wb"))

######################
### Evaluate Model ###
######################
# see how well we can order people by survival
C_package = concordance_index(X_test['alive_year'], -cph.predict_partial_hazard(X_test), X_test[args.mortality])
print('testing C-index: ', C_package)
# C = c_statistic_harrell(model_train.predict(X_test), y_test)
C_file.write('C-statistic package: '+str(C_package)+'\n')
# C_file.write('C-statistic: '+str(C))

C_package = concordance_index(X_geo['alive_year'], -cph.predict_partial_hazard(X_geo), X_geo[args.mortality])
print('GeoValidatioin C-index: ', C_package)
# C = c_statistic_harrell(model_train.predict(X_test), y_test)
C_file.write('GeoValidatioin C-statistic package: '+str(C_package)+'\n')
# C_file.write('C-statistic: '+str(C))
C_file.close()
