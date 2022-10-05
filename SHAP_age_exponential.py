import matplotlib.pylab as plt
from scipy.optimize import curve_fit
import shap
import numpy as np


def lnxy(x,y):
    return np.log(y)/np.log(x)


class SHAP_Age:
    def __init__(self, task):
        self.popt = None
        self.pcov = None
        self.task = None

    def fit(self, prediction, age, expected_value_list, age_list):
        popt, pcov = curve_fit(lambda t,a,b,c: np.exp(a*t+b)+np.min(prediction)+c, np.array(age), prediction, maxfev=10000)
        if popt[2] >= 0:
            popt, pcov = curve_fit(lambda t,a,b: np.exp(a*t+b)+np.min(prediction)-0.1, np.array(age), prediction, maxfev=10000)
            popt = np.append(popt, -0.1)
        print(popt[2])
        self.popt = popt
        self.pcov = pcov
        self.model_predict_min = np.min(prediction)
        self.expected_value_list = expected_value_list
        self.age_list = age_list
        return popt, pcov
    
    def fitting_plot(self, fore_prediction, title='', fontsize=18, show=False, save_path=None):
        '''plot the fitted function and the data points'''
        expected_value_list_long = np.arange(fore_prediction.min(), fore_prediction.max(), 0.1)
        yvals = [(np.log(x - self.model_predict_min - self.popt[2])-self.popt[1])/self.popt[0] for x in expected_value_list_long]
        plot1 = plt.plot(self.expected_value_list, self.age_list, 's',label='Mean predicted values')
        plot2 = plt.plot(expected_value_list_long, yvals, 'r',label='Fitted values')
        plt.xlabel('Predicted value', fontsize=fontsize)
        plt.ylabel('Age', fontsize=fontsize)
        plt.tick_params(axis='both',which='major',labelsize=fontsize)
        plt.rc('axes', labelsize=fontsize)
        plt.rc('legend', fontsize=fontsize)
        plt.legend(loc=4)
        plt.title(title+': fitting plot', fontsize=fontsize)
        if save_path:
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
        if show:
            plt.show()
        return
    
    def exp_plot(self, prediction, age, title='', fontsize=18, show=False, save_path=None):
        fitted_pred = [np.exp(self.popt[0]*x+self.popt[1])+np.min(prediction)+self.popt[2] for x in self.age_list]
        plt.plot(age, prediction, 'C0.', markersize=12)
        plt.plot(self.age_list, fitted_pred, '-', color='orange', label='Log fit')
        plt.rc('axes', labelsize=fontsize)
        plt.rc('legend', fontsize=fontsize)
        plt.title(title+': fitting plot', fontsize=fontsize)
        plt.xlabel('Chronological age', fontsize=fontsize)
        plt.ylabel('Prediction (log-odds)', fontsize=fontsize)
        if save_path:
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
        if show:
            plt.show()
        plt.clf()
        return
    
    def get_shap_age(self, x):
        return (np.log(x - self.model_predict_min - self.popt[2])-self.popt[1])/self.popt[0]
#         return self.popt[0]*x+self.popt[1]
        
    def convert_shap_values(self, model_train, shap_values_all_dict, fore_prediction, back_prediction, fore_age_round, back_age_round):
        print('Starting converting SHAP values')
        fore_final_attr_dict = {}
        back_shap_age_dict = {}   # Foreground samples prediction
        fore_shap_age_dict = {}   # Background samples prediction
        for age in self.age_list:
            print(age)
#             fore_data_temp = fore_data_ori[fore_age_round==age]  # Foreground samples to explain
#             back_data_temp = back_data_ori[back_age_round==age]  # Background samples to compare to
#             fore_data_temp = fore_data_temp[model_train.get_booster().feature_names]
#             back_data_temp = back_data_temp[model_train.get_booster().feature_names]
            fore_attr_defa = shap_values_all_dict[age].mean(2)
            fore_attr_per_ref = shap_values_all_dict[age]
            fore_attr_per_ref = np.swapaxes(fore_attr_per_ref,1,2)                           # Re-order axes 
            fore_lodd_pred = fore_prediction[fore_age_round==age]
            back_lodd_pred = back_prediction[back_age_round==age]
#             assert(np.allclose(fore_attr_defa.sum(1), fore_lodd_pred-back_lodd_pred.mean(), atol=1e-3))
            # Function of the model prediction applied to fore_data_temp and back_data_temp (used for rescale rule)
            fore_func_pred = self.get_shap_age(fore_lodd_pred)
            back_func_pred = self.get_shap_age(back_lodd_pred)
            fore_shap_age_dict[age] = fore_func_pred
            back_shap_age_dict[age] = back_func_pred
            # Get the factor by which we rescale the per back_data_temp attributions
            denom    = fore_lodd_pred[:,None] - back_lodd_pred[None,:]
            numer    = fore_func_pred[:,None] - back_func_pred[None,:]
            safe_div = lambda a,b : np.divide(a, b, out=np.zeros_like(a), where=b!=0) # Divide by zero gives zero
            rescale  = safe_div(numer, denom)        # Rescale factor based on func_lodds
            # Do the final rescaling
            final_attr = fore_attr_per_ref * rescale[:,:,None] # The final, rescaled attribution
            fore_final_attr_dict[age] = final_attr.mean(1)
        return fore_final_attr_dict, fore_shap_age_dict, back_shap_age_dict
    
    def individualized_plot(self, mean_age, attributions, feature_values, feature_names, show=False, save_path=None):
        shap.force_plot(mean_age, attributions, feature_values, matplotlib=True, feature_names=feature_names, show=False, text_rotation=375)
        if save_path:
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
        if show:
            plt.show()
            
    def shap_age_vs_age_plot(self, age_list, prediction, shap_age_dict, back_shap_age_dict, age_ori, age_round, mortstat=None , title='', fontsize=18, show=False, save_path=None):
        shap_age = np.array([self.get_shap_age(prediction[i]) for i in range(len(prediction))])
        shap_age_dict = {}
        for age in age_list:
            shap_age_dict[age] = np.array(shap_age[age_round==age])

        mean_list = [np.mean(shap_age_dict[age]) for age in age_list]
        base_value_list = [np.mean(back_shap_age_dict[age]) for age in self.age_list]
        shap_age [shap_age < 0] = 0
        plt.figure(figsize=(10,8))
        if mortstat is not None:
            plt.scatter(np.array(age_ori)[mortstat == 0], np.array(shap_age)[mortstat == 0], alpha=0.3, color='#018ae7', label='Alive')
            plt.scatter(np.array(age_ori)[mortstat == 1], np.array(shap_age)[mortstat == 1], alpha=0.3, color='#ff0255', label='Deceased')
        else:
            plt.scatter(np.array(age_ori), np.array(shap_age), alpha=0.3, color='#018ae7', label='Alive')
        plot1 = plt.plot(list(range(10,90)), list(range(10,90)), 'gray', linestyle="-")
        plot2 = plt.plot(age_list, mean_list, 'orange', label='Mean values')
#         plot3 = plt.plot(self.age_list, base_value_list, 'black', label='Base values')
        plt.ylabel('EXPECT Age', fontsize=fontsize)
        plt.xlabel('Chronological Age', fontsize=fontsize)
        plt.tick_params(axis='both',which='major',labelsize=fontsize)
        plt.xlim(10, 90)
        plt.ylim(10, 90)
        plt.rc('axes', labelsize=fontsize)
        plt.rc('legend', fontsize=fontsize)
        plt.legend()
        plt.title(title+': EXPECT Age v.s. Chronological Age', fontsize=fontsize)
        if save_path:
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
        if show:
            plt.show()
        return shap_age
    
    def label(self, permth, mortstat, year):
        if permth > year:
            return 0
        else:
            if mortstat == 1:
                return 1
            else:
                return 2
    
    def survival_probability(self, age, shap_age, year_num, permth_int, mortstat, title='', fontsize=18, show=False, save_path=None):  # the unit of permth_int is year
        age = np.array(age)
        shap_age = np.array(shap_age)
        
        age_old = age.copy()
        age = age[(age_old>=min(self.age_list)) & (age_old<=max(self.age_list))]
        shap_age = shap_age[(age_old>=min(self.age_list)) & (age_old<=max(self.age_list))]
        permth_int = permth_int[(age_old>=min(self.age_list)) & (age_old<=max(self.age_list))]
        mortstat = mortstat[(age_old>=min(self.age_list)) & (age_old<=max(self.age_list))]
        
        unhealthy_index = [i for i in range(len(age)) if (shap_age[i] > age[i])]
        healthy_index = [i for i in range(len(age)) if (shap_age[i] <= age[i])]
        unhealthy_surv_prob = [1]
        for i in range(1, year_num+1):
            label_temp = np.array([self.label(permth_int[j], mortstat[j], i) for j in unhealthy_index])
            label_temp_pre = np.array([self.label(permth_int[j], mortstat[j], (i-1)) for j in unhealthy_index])
            num_samples = sum(label_temp==0)+sum(label_temp==1)-sum(label_temp_pre==1)
            unhealthy_surv_prob.append((sum(label_temp==0)/num_samples)*unhealthy_surv_prob[-1])
#             print((sum(label_temp==0)/num_samples))
        healthy_surv_prob = [1]
        for i in range(1, year_num+1):
            label_temp = np.array([self.label(permth_int[j], mortstat[j], i) for j in healthy_index])
            label_temp_pre = np.array([self.label(permth_int[j], mortstat[j], (i-1)) for j in healthy_index])
            num_samples = sum(label_temp==0)+sum(label_temp==1)-sum(label_temp_pre==1)
            healthy_surv_prob.append((sum(label_temp==0)/num_samples)*healthy_surv_prob[-1])
#             print((sum(label_temp==0)/num_samples))
        plt.figure(figsize=(10,8))
        plt.step([i for i in range(year_num+1)], healthy_surv_prob, 'blue',label='Healthy agers', where="post")
        plt.step([i for i in range(year_num+1)], unhealthy_surv_prob, 'r',label='Unhealthy agers', where="post")
        plt.xlabel('Years from screening time', fontsize=fontsize)
        plt.ylabel('Cumulative Survival Probability', fontsize=fontsize)
        plt.tick_params(axis='both',which='major',labelsize=fontsize)
        plt.rc('axes', labelsize=fontsize)
        plt.rc('legend', fontsize=fontsize)
        plt.legend()
        plt.title(title+': survival probability', fontsize=fontsize)
        if save_path:
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
        if show:
            plt.show()
        return healthy_surv_prob, unhealthy_surv_prob