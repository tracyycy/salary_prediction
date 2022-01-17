import re
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
import shap

from clean_data import gnp_dict, dict_country, salary_median, ed2int, feat4model
from explain_samples import all_summary_plot

all_feat = feat4model['cat'] + feat4model['multi']  + feat4model['num'] 

class Request:
    
    def __init__(self, job, country, edu, years, language, database, platform, webframe):
        
        self.job = job
        self.job_clean = re.sub("[\W]+", "_", job)
        self.country = country
        self.edu = edu
        self.language =  ['Language_' + i for i in language]
        self.database = ['Database_' + i for i in database]
        self.platform = ['Platform_' + i for i in platform]
        self.webframe = ['Webframe_' + i for i in webframe]
        self.country_code = gnp_dict['Country Code'].get(self.country)

        self.request = dict.fromkeys(all_feat, 0)
        self.request['Region'] = dict_country['Region'].get(self.country_code)
        self.request['YearsCodePro'] = years
        self.request['gnp'] = gnp_dict['2020'].get(self.country)
        self.request['role_adjustment'] = salary_median.get(self.job)
        self.request['education'] = ed2int.get(edu)
        
        for tech in [self.language, self.database, self.platform, self.webframe]:
            for i in tech:
                self.request[i] = 1
                
        self.model = CatBoostRegressor().load_model("./model/" + self.job_clean + ".cbm")
        #self.summary_plot = "./summary_plot/" + self.job_clean + ".png"
        self.summary_plot = all_summary_plot[self.job_clean]
        self.request_df = pd.DataFrame([self.request])[all_feat]

    def predict(self):
        predicted_ratio = self.model.predict(self.request_df)[0]
        return predicted_ratio
    
    def waterfall(self):
        explainer = shap.TreeExplainer(self.model)
        shap_exp = explainer(self.request_df)
        shap_values = explainer.shap_values(
            Pool(self.request_df,   
                 cat_features=feat4model['cat'] + feat4model['multi']))
        shap_exp.values = shap_values
        shap.plots.waterfall(shap_exp[0],  max_display=20)
        fig = plt.gcf()
        return fig