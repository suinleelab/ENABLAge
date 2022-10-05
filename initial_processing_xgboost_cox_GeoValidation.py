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
random_state = 528
parser = argparse.ArgumentParser()
parser.add_argument('--age', dest="age", help="age range", default='')
parser.add_argument('--mortality', dest='mortality', help='mortality type, choose from all-cause, neoplasms, circulatory, respiratory, digestive, external, other')
parser.add_argument('--feature_selection', dest='feature_selection', help='whether use feature selection')
args = parser.parse_args()
path = './result/IMPACT_Age_missforest/XGB_Cox'+'_'+args.mortality
# path = './result/IMPACT_Age_missforest/XGB_Cox_GradientBoostingSurvivalAnalysis'+'_'+args.mortality
if args.age != '':
    path += '_age_'+args.age
if args.feature_selection == '1':
    path += '_feature_selection'
# path += '_remove20002and20004_small_imputed_AgeAdjusted_CancerAdjusted_FloatAge_test_GeoValidation_WithoutVal/'
path += '_remove20002and20004_small_imputed_AgeAdjusted_CancerAdjusted_test_NewLabel_FloatAge_test_GeoValidation/'

if not os.path.isdir(path):
    os.mkdir(path)
C_file = open(path+'score.txt', 'a')
age_feature = 'Age'

### scottish code: 11004, 11005
# if args.feature_selection == '1':
#     features = pd.read_csv('../features_initial_preprocessing_feature_selection.csv')
# else:
#     features = pd.read_csv('../features_initial_preprocessing.csv')
# features = pd.read_csv('/projects/leelab/nobackup/wqiu/UK_Biobank_genetic_data/pheno_data/features_initial_preprocessing.csv')
# columns = pd.read_csv('../uk_biobank/features_initial_preprocessing_feature_selection_missforest_imputed_no_missing_lancet_and_meaningful_adjusted_assays_remove20002and20004.csv', nrows=1).columns
# features = features[columns]
features = pd.read_csv('/projects/leelab/nobackup/wqiu/UK_Biobank_genetic_data/pheno_data/features_initial_preprocessing_missforest_imputed_no_missing_lancet_and_meaningful_adjusted_assays_remove20002and20004_AgeAdjusted_CancerAdjusted_geo.csv')
# print(data_path, file=C_file)
# label_df = pd.read_csv('/projects/leelab/nobackup/wqiu/UK_Biobank_genetic_data/pheno_data/death_label.csv')
label_df = pd.read_csv('/projects/leelab/nobackup/wqiu/UK_Biobank_genetic_data/pheno_data/death_label_new.csv')
if args.mortality != 'all-cause':
    data_mortality = pd.merge(features, label_df[['eid', 'alive_year', 'all-cause', args.mortality, 'external']], how='left', on='eid')
else:
    data_mortality = pd.merge(features, label_df[['eid', 'alive_year', 'all-cause', 'external']], how='left', on='eid')

# feature_df = pd.read_csv('../feature_list_preprocessed_0_removed.csv')
# impute_sth_list = list(set(feature_df.loc[~feature_df['Impute_sth'].isnull(), 'UDI']) - set(feature_df.loc[~feature_df['Onehot'].isnull(), 'UDI']) - set(feature_df.loc[~feature_df['Merge_cat_onehot'].isnull(), 'UDI']))
# for fea in impute_sth_list:
#     if fea in data_mortality:
#         data_mortality.loc[data_mortality[fea] == feature_df.loc[feature_df['UDI']==fea, 'Impute_sth'].values[0], fea] = np.nan
#     else:
#         print(fea)
        
X = data_mortality
X = X[(X[age_feature]>=39.5) & (X[age_feature]<=70.5)].reset_index(drop=True)  
X = X[(X['external']!=1)]
X = X[(X['all-cause']==0) | (X[args.mortality]==1)]
X_geo = X.loc[(X['54-0.0']==11004) | (X['54-0.0']==11005)].reset_index(drop=True)
X = X.loc[(X['54-0.0']!=11004) & (X['54-0.0']!=11005)].reset_index(drop=True)
mortstat = X[args.mortality]
permth_int = X['alive_year']
y = permth_int * (mortstat - .5)*2
eid = X['eid']
X = X.drop([args.mortality, 'all-cause', 'external', 'alive_year', 'eid', '54-0.0', '21003-0.0'], axis=1)
mortstat_geo = X_geo[args.mortality]
permth_int_geo = X_geo['alive_year']
y_geo = permth_int_geo * (mortstat_geo - .5)*2
eid_geo = X_geo['eid']
X_geo = X_geo.drop([args.mortality, 'all-cause', 'external', 'alive_year', 'eid', '54-0.0', '21003-0.0'], axis=1)
# if args.feature_selection != '1':
#     X = X.drop(['Age'], axis=1)
# X = X.drop(['21022-0.0', '41250-0.0_3000', '41250-0.0_3001', '41250-0.0_3002', '41250-0.0_3004', '6156-0.0_-1', '20111-0.0_-1', '41231-0.0_-1', '41245-0.0_-1', '41246-0.0_-1', '20001-0.0_99999', '20001-0.0_-2', '20002-0.0_99999', '20002-0.0_-2', '20004-0.0_99999', '20004-0.0_-2', '41221-0.0_-1', '41222-0.0_-1', '41223-0.0_-1', '41224-0.0_-1', '41225-0.0_-1', '41226-0.0_-1', '41227-0.0_-1', '41228-0.0_-1', '6143-0.0_-1', '54-0.0_2', '54-0.0_3', '54-0.0_4', '54-0.0_5', '54-0.0_6', '54-0.0_7', '54-0.0_-7'], axis=1)

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=random_state)

eid_train, eid_test, _, _ = train_test_split(eid, eid, test_size=0.2, random_state=random_state)

pickle.dump(list(eid_test), open(path+'eid_test.pkl', 'wb'))
pickle.dump(list(eid_geo), open(path+'eid_geo.pkl', 'wb'))

exit()
mortstat_train, mortstat_test, permth_int_train, permth_int_test = train_test_split(mortstat, permth_int, test_size=0.2, random_state=random_state)
print(X_train.index)
# xgb_train = xgboost.DMatrix(X_train, label=y_train)
# xgb_val = xgboost.DMatrix(X_val, label=y_val)
# xgb_test = xgboost.DMatrix(X_test, label=y_test)
y_train = np.array(y_train); y_test = np.array(y_test);# y_val = np.array(y_val)

print('start training XGBoost')
###################
### Train Model ###
###################
### all-cause
# params = {'learning_rate': 0.01, 'max_depth': 7, 'subsample': 0.7, 'n_estimators': 10000, 'objective': 'survival:cox'}

### neoplasms
# params = {'learning_rate': 0.01, 'max_depth': 7, 'subsample': 0.7, 'n_estimators': 10000, 'objective': 'survival:cox'}

### circulatory
# params = {'learning_rate': 0.01, 'max_depth': 3, 'subsample': 0.5, 'n_estimators': 10000, 'objective': 'survival:cox'}

### other tasks
# params = {'learning_rate': 0.01, 'max_depth': 5, 'subsample': 0.5, 'n_estimators': 10000, 'objective': 'survival:cox'}

# print(params)
# print(params, file=C_file)
# xlf = xgboost.XGBRegressor(**params)
# xlf.fit(X_train, y_train, eval_set = [(X_val, y_val)], early_stopping_rounds=100, verbose=True)
# model_train = xlf
# pickle.dump(model_train, open(path+"model.pickle.dat", "wb"))
model_train = pickle.load(open(path+"model.pickle.dat", "rb"))
######################
### Evaluate Model ###
######################
# see how well we can order people by survival
C_package = concordance_index(permth_int_test, -model_train.predict(X_test), mortstat_test)
C_package_geo = concordance_index(permth_int_geo, -model_train.predict(X_geo), mortstat_geo)

print('testing C-index: ', C_package)
print('Geographical validation C-index: ', C_package_geo)
# C = c_statistic_harrell(model_train.predict(X_test), y_test)
C_file.write('C-statistic package: '+str(C_package)+'\n')
C_file.write('Geographical validation C-statistic package: '+str(C_package_geo)+'\n')
# C_file.write('C-statistic: '+str(C))
C_file.close()
# exit()
#######################################################################################
print('start training TreeExplainer')
if len(X_train)>=10000:
    back_data = X_train.sample(n=10000, random_state=random_state)
else:
    back_data = X_train
back_data.to_csv(path+'back_data.csv', index = False)
if len(X_test)>=5000:
    fore_data = X_test.sample(n=5000, random_state=random_state)
    fore_data_label = pd.DataFrame(y_test).sample(n=5000, random_state=random_state)
else:
    fore_data = X_test
    fore_data_label = pd.DataFrame(y_test)
fore_data.to_csv(path+'fore_data.csv', index = False)
fore_data_label.to_csv(path+'fore_data_label.csv', index = False)

file = open("./feature_names_dictionary_DateToAge.pkl", "rb")
col_dict = pickle.load(file)
file.close()
display_col=[]
for col in fore_data.columns:
    if col in col_dict:
        display_col.append(col_dict[col])
    else:
        display_col.append(col)

# ####################### SHAP values & SHAP interaction values ############################
# explainer = shap.TreeExplainer(model_train, data=back_data)
# shap_values = explainer.shap_values(fore_data)
# np.save(path+'shap_values.npy', shap_values)
# shap.summary_plot(shap_values, fore_data, feature_names=display_col, show=False)
# pl.savefig(path+'summary_plot.png', format='png', bbox_inches='tight')
# pl.close()

# plot_feature = X.columns[np.argsort(-np.sum(np.abs(shap_values), axis=0))][0:5]
# for f in plot_feature:
#     if f not in X.columns:
#         continue
#     feature_name = col_dict[f]
#     shap.dependence_plot(feature_name, shap_values, fore_data, feature_names=display_col, show=False)
#     if '/' in feature_name:
#         feature_name = feature_name.replace('/', '_')
#     pl.savefig(path+feature_name+'.png', format='png', bbox_inches='tight')
#     pl.close()

# print('start calculating SHAP interaction values')
# shap_inter_values = shap.TreeExplainer(model_train, data=back_data, feature_perturbation='tree_path_dependent').shap_interaction_values(fore_data)
# np.save(path+'shap_interaction_values.npy', shap_inter_values)


######################SHAP AGE########################

if not os.path.isdir(path+'different_age_background/'):
    os.mkdir(path+'different_age_background/')
mortstat_train, mortstat_test, permth_int_train, permth_int_test = train_test_split(mortstat, permth_int, test_size=0.2, random_state=random_state)
mortstat_train, mortstat_val, permth_int_train, permth_int_val = train_test_split(mortstat_train, permth_int_train, test_size=0.2, random_state=random_state)

print(mortstat_test.index)
print(mortstat_test.shape)
np.save(path+'different_age_background/mortstat_test.npy', np.array(mortstat_test))
np.save(path+'different_age_background/permth_int_test.npy', np.array(permth_int_test))

sample_rate = 1
if sample_rate == 1:
    back_data = X_train
    fore_data = X_test
    mortstat_fore = mortstat_test
    permth_int_fore = permth_int_test
else:
    back_data = X_train.sample(n=int(sample_rate*len(X_train)), random_state=random_state)
    fore_data = X_test.sample(n=int(sample_rate*len(X_test)), random_state=random_state)

    back_data_label = pd.DataFrame(y_train).sample(n=int(sample_rate*len(X_train)), random_state=random_state)
    fore_data_label = pd.DataFrame(y_test).sample(n=int(sample_rate*len(X_test)), random_state=random_state)

    mortstat_fore = mortstat_test.sample(n=int(sample_rate*len(X_test)), random_state=random_state)
    permth_int_fore = permth_int_test.sample(n=int(sample_rate*len(X_test)), random_state=random_state)

# back_data.to_csv(path+'different_age_background/back_data.csv', index = False)
# fore_data.to_csv(path+'different_age_background/fore_data.csv', index = False)
# back_data_label.to_csv(path+'different_age_background/back_data_label.csv', index = False)
# fore_data_label.to_csv(path+'different_age_background/fore_data_label.csv', index = False)

if not os.path.isdir(path+'different_age_background/male/'):
    os.mkdir(path+'different_age_background/male/')
if not os.path.isdir(path+'different_age_background/female/'):
    os.mkdir(path+'different_age_background/female/')


back_data_female = back_data[back_data['31-0.0']==0]
back_data_male = back_data[back_data['31-0.0']==1]
fore_data_female = fore_data[fore_data['31-0.0']==0]
fore_data_male = fore_data[fore_data['31-0.0']==1]
mortstat_fore_female = np.array(mortstat_fore)[fore_data['31-0.0']==0]
mortstat_fore_male = np.array(mortstat_fore)[fore_data['31-0.0']==1]
permth_int_fore_female = np.array(permth_int_fore)[fore_data['31-0.0']==0]
permth_int_fore_male = np.array(permth_int_fore)[fore_data['31-0.0']==1]

fore_data_female.to_csv(path+'different_age_background/female/fore_data.csv', index=False)
fore_data_male.to_csv(path+'different_age_background/male/fore_data.csv', index=False)
back_data_female.to_csv(path+'different_age_background/female/back_data.csv', index=False)
back_data_male.to_csv(path+'different_age_background/male/back_data.csv', index=False)
np.save(path+'different_age_background/female/mortstat_fore.npy', np.array(mortstat_fore_female))
np.save(path+'different_age_background/male/mortstat_fore.npy', np.array(mortstat_fore_male))
np.save(path+'different_age_background/female/permth_int_fore.npy', np.array(permth_int_fore_female))
np.save(path+'different_age_background/male/permth_int_fore.npy', np.array(permth_int_fore_male))

fore_age_round_female = fore_data_female[age_feature].apply(lambda x: np.round(x))
fore_age_round_male = fore_data_male[age_feature].apply(lambda x: np.round(x))
back_age_round_female = back_data_female[age_feature].apply(lambda x: np.round(x))
back_age_round_male = back_data_male[age_feature].apply(lambda x: np.round(x))

age_list_female = sorted(list(set(fore_age_round_female)))
age_list_male = sorted(list(set(fore_age_round_male)))

pickle.dump(age_list_female, open(path+'different_age_background/female/age_list.pkl', 'wb'))
pickle.dump(age_list_male, open(path+'different_age_background/male/age_list.pkl', 'wb'))




# ############ different age shap values
# back_female_max = 0
# fore_female_max = 0
# back_male_max = 0
# fore_male_max = 0
# for age in age_list_female:
#     print(age)
#     fore_data_temp = fore_data_female[fore_age_round_female==age]
#     back_data_temp = back_data_female[back_age_round_female==age]
#     back_female_max = max(back_female_max, len(back_data_temp))
#     fore_female_max = max(fore_female_max, len(fore_data_temp))
# #     fore_data_temp.to_csv(path+'different_age_background/female/fore_data_'+str(age)+'.csv', index=False)
# #     back_data_temp.to_csv(path+'different_age_background/female/back_data_'+str(age)+'.csv', index=False)
#     explainer = shap.TreeExplainer(model_train, data=back_data_temp)
#     shap_values_all = explainer.shap_values(fore_data_temp, per_reference=True) # Attributions per reference
#     np.save(path+'different_age_background/female/shap_values_all_'+str(age)+'.npy', shap_values_all)
    
# for age in age_list_male:
#     print(age)
#     fore_data_temp = fore_data_male[fore_age_round_male==age]
#     back_data_temp = back_data_male[back_age_round_male==age]
#     back_male_max = max(back_male_max, len(back_data_temp))
#     fore_male_max = max(fore_male_max, len(fore_data_temp))
# #     fore_data_temp.to_csv(path+'different_age_background/male/fore_data_'+str(age)+'.csv', index=False)
# #     back_data_temp.to_csv(path+'different_age_background/male/back_data_'+str(age)+'.csv', index=False)
#     explainer = shap.TreeExplainer(model_train, data=back_data_temp)
#     shap_values_all = explainer.shap_values(fore_data_temp, per_reference=True) # Attributions per reference
#     np.save(path+'different_age_background/male/shap_values_all_'+str(age)+'.npy', shap_values_all)
    
# print('back_female_max: ', back_female_max)
# print('fore_female_max: ', fore_female_max)
# print('back_male_max: ', back_male_max)
# print('fore_male_max: ', fore_male_max)
# ##### when using all samples
# # back_female_max:  11163
# # fore_female_max:  2720
# # back_male_max:  8960
# # fore_male_max:  2217