import pickle
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score
import os
import sys; sys.path.insert(0,"/projects/leelab3/hughchen/shap")
import shap
from SHAP_age_new23 import SHAP_Age

parser = argparse.ArgumentParser()
parser.add_argument('--path', dest="path")
parser.add_argument('--dataset', dest='dataset', help='choose from NHANES, biobank, NHANES_small, or biobank_small')
parser.add_argument('--gender', dest="gender", help='choose from female and male')
parser.add_argument('--transfer', dest="transfer", default = None, type=int, help='whether transfer to another datset')
parser.add_argument('--target_data_path', dest="target_data_path", default = None, help='the path of target dataset')
parser.add_argument('--save_folder', dest="save_folder", help='the path of saved SHAP_Age object')
parser.add_argument('--task', dest="task", help='mortality cause')
args = parser.parse_args()



save_path = args.path+'/different_age_background/'+args.gender+'/'+args.save_folder+'/'
print(save_path)
if not os.path.isdir(save_path):
    os.mkdir(save_path)
model_path = args.path+'/model.pickle.dat'
model_train = pickle.load(open(model_path, "rb"))
if (args.dataset == 'NHANES') or (args.dataset == 'NHANES_small') or (args.dataset == 'biobank_small'):
    age_feature_name = 'Demographics_Age'
elif args.dataset == 'biobank':
    age_feature_name = '21003-0.0'
else:
    print('Unsupported dataset')
    exit()


##### load data #####********************************************************************************************************
back_data_ori = pd.read_csv(args.path+'/different_age_background/'+args.gender+'/back_data.csv')
    
if (args.dataset == 'NHANES_small') and args.transfer:
    fore_data_ori = pd.read_csv('/projects/leelab2/wqiu/NHANES/data/data_460_classification_imputed_missforest_feature_selection.csv')
    mortality = pd.read_csv('/projects/leelab2/wqiu/NHANES/data/mortality_label.csv')
    not_external_index = mortality[mortality['external']==0].index
    fea_list = pd.read_csv('../NHANES_feature_list.csv')
    nominal_fea = fea_list[fea_list['Nominal']==1]['Type_Short_Name'].tolist()
    nominal_fea = list(set(nominal_fea) & set(fore_data_ori.columns))
    fore_data_ori = pd.get_dummies(fore_data_ori, columns=nominal_fea, drop_first=True)
    fore_data_ori = fore_data_ori[back_data_ori.columns]
    if args.gender == 'female':
        gender_index = fore_data_ori[fore_data_ori['Demographics_Gender_2.0']==1].index
    elif args.gender == 'male':
        gender_index = fore_data_ori[fore_data_ori['Demographics_Gender_2.0']==0].index
    else:
        print('Unsupported gender')
        exit()
    print('len(gender_index): ', len(gender_index))
    age_index = fore_data_ori[(fore_data_ori['Demographics_Age']>=40) & (fore_data_ori['Demographics_Age']<=70)].index
    leave_index = list(set(gender_index) & set(not_external_index) & set(age_index))
    print('len(leave_index): ', len(leave_index))
    fore_data_ori = fore_data_ori.loc[leave_index, :]
    print(len(fore_data_ori))
else:
    fore_data_ori = pd.read_csv(args.path+'/different_age_background/'+args.gender+'/fore_data.csv')
    
fore_data = fore_data_ori.copy()


if (args.dataset == 'NHANES') or (args.dataset == 'NHANES_small') or (args.dataset == 'biobank_small'):
    display_name = pd.read_csv('../NHANES_feature_list_Display_name.csv')
    display_col=[]
    for col in fore_data.columns:
        display_col.append(list(display_name.loc[display_name['Type_Short_Name']==col, 'Display_Name'])[0])
    col_dict = dict(zip(fore_data.columns, display_col))

    change_0_1 = list(display_name.loc[display_name['Change_0_1']==1, 'Type_Short_Name'])
    for fea in change_0_1:
        if fea in fore_data.columns:
            temp = fore_data[fea].copy()
            fore_data.loc[temp==1, fea] = 0
            fore_data.loc[temp==0, fea] = 1
    print(fore_data.shape)
elif args.dataset == 'biobank':
    file = open("./feature_names_dictionary_DateToAge.pkl", "rb")
    col_dict = pickle.load(file)
    file.close()
    display_col=[]
    for col in fore_data.columns:
        if col in col_dict:
            display_col.append(col_dict[col])
        else:
            display_col.append(col)

            
##### preprocessing data #####********************************************************************************************************
fore_age_round = fore_data_ori[age_feature_name].apply(lambda x: np.round(x))
back_age_round = back_data_ori[age_feature_name].apply(lambda x: np.round(x))
age_list = sorted(list(set(back_age_round)))
fore_age_list = sorted(list(set(fore_age_round)))
shap_values_all_dict = {}
expected_value = {}
for age in age_list:
    print(age)
    fore_data_temp = fore_data_ori[fore_age_round==age]
    back_data_temp = back_data_ori[back_age_round==age]
    pre = model_train.predict(back_data_temp, output_margin=True)
    expected_value[age] = np.median(pre)  # before used median
    if args.transfer:
        explainer = shap.TreeExplainer(model_train, data=back_data_temp)
        shap_values_all_dict[age] = explainer.shap_values(fore_data_temp, per_reference=True)
    else:
        shap_values_all_dict[age] = np.load(args.path+'/different_age_background/'+args.gender+'/shap_values_all_'+str(age)+'.npy')\
    
############# Calculate SHAP age #############*************************************************************************
back_prediction = model_train.predict(back_data_ori, output_margin=True)
fore_prediction = model_train.predict(fore_data_ori, output_margin=True)
model_predict_min = min(back_prediction.min(), fore_prediction.min())
model_predict_max = max(back_prediction.max(), fore_prediction.max())
expected_value_list = [expected_value[age] for age in age_list]

if args.transfer:
    shap_age = pickle.load(open(args.path+'/different_age_background/'+args.gender+'/SHAP_age/shap_age_object.pkl', "rb"))
else:
    shap_age = SHAP_Age(task=args.dataset+', '+args.gender)
    popt = shap_age.fit(model_predict_min, expected_value_list, age_list)
    pickle.dump(shap_age, open(save_path+'shap_age_object.pkl', 'wb'))
    shap_age.fitting_plot(fore_prediction, title=args.dataset+', '+args.gender+', '+args.task, show=True, save_path=save_path+'/fitting_plot.pdf')

    
############# SHAP Age v.s. Age plot #############***********************************************************************
if (args.dataset == 'biobank') or (args.dataset == 'biobank_small'):
    mortstat_fore = np.load(args.path+'/different_age_background/'+args.gender+'/mortstat_fore.npy')
    permth_int_fore = np.load(args.path+'/different_age_background/'+args.gender+'/permth_int_fore.npy')

elif (args.dataset == 'NHANES_small') and args.transfer:
    mortstat_fore = np.array(pd.read_csv('/projects/leelab2/wqiu/NHANES/data/mortality_label.csv')[args.task][leave_index])
    permth_int_fore = np.array(pd.read_csv('/projects/leelab2/wqiu/NHANES/data/mortality_label.csv')['permth_int'][leave_index])
    print('# number of deceased samples: ', sum(mortstat_fore))
else:
    mortstat_fore = np.load(args.path+'/different_age_background/'+args.gender+'/mortstat_test.npy')
    permth_int_fore = np.load(args.path+'/different_age_background/'+args.gender+'/permth_int_test.npy')

fore_shap_age_dict = {}
back_shap_age_dict = {}
for age in age_list:
    back_lodd_pred = back_prediction[back_age_round==age]
    back_func_pred = shap_age.get_shap_age(back_lodd_pred)
    back_shap_age_dict[age] = back_func_pred

for age in fore_age_list:
    fore_lodd_pred = fore_prediction[fore_age_round==age]    
    fore_func_pred = shap_age.get_shap_age(fore_lodd_pred)
    fore_shap_age_dict[age] = fore_func_pred
    
fore_shap_age = shap_age.shap_age_vs_age_plot(fore_age_list, fore_prediction, fore_shap_age_dict, back_shap_age_dict, fore_age_round, mortstat_fore, title= args.dataset+', '+args.gender+', '+args.task, show=False, save_path=save_path+'/SHAP_age_vs_age.pdf')

############# Survival probability #############***********************************************************************
if (args.dataset == 'biobank') or (args.dataset == 'biobank_small'):
    year_num = 13
elif (args.dataset == 'NHANES') or (args.dataset == 'NHANES_small'):
    year_num = 10
    permth_int_fore /= 12
shap_age.survival_probability(fore_data_ori[age_feature_name], fore_shap_age, year_num, permth_int_fore, mortstat_fore, title=args.dataset+', '+args.gender+', '+args.task, show=False, save_path=save_path+'/survival_probability.pdf')

# ############# Get per-reference attributions #############*************************************************************************
# fore_final_attr_dict, fore_shap_age_dict, back_shap_age_dict = shap_age.covert_shap_values(model_train, shap_values_all_dict, fore_prediction, back_prediction, fore_age_round, back_age_round)
# pickle.dump(fore_final_attr_dict, open(save_path+'fore_final_attr_dict.pkl', 'wb'))
# pickle.dump(back_shap_age_dict, open(save_path+'back_shap_age_dict.pkl', 'wb'))

# ############# Individualized decomposed age plot #############***********************************************************************
# age = 60
# idx = 1
# shap_age.individualized_plot(back_shap_age_dict[age].mean(), fore_final_attr_dict[age][idx,:], fore_data[fore_age_round==age].iloc[idx,:], display_col, show=False, save_path=save_path+'/individualized_plot_age_'+str(age)+'_index_'+str(idx)+'.pdf')
