import shap
import matplotlib.pyplot as plt
from catboost import Pool, CatBoostRegressor, cv

from clean_data import data4model, feat4model

all_feat = feat4model['cat'] + feat4model['multi']  + feat4model['num'] 
all_summary_plot = {}

for job in data4model.keys():
    
    job_clean = job.replace("DevType_", "")
    samples = data4model[job].sample(min(1000, len(data4model[job])), random_state=42)[all_feat]
    
    model = CatBoostRegressor().load_model("./model/" + job_clean + ".cbm")
    explainer = shap.TreeExplainer(model)
    
    shap_values = explainer.shap_values(
            Pool(samples, cat_features=feat4model['cat'] + feat4model['multi'])
    )
    shap.summary_plot(shap_values, samples, show=True)
    fig = plt.gcf()
    
    all_summary_plot[job_clean] = fig
    
    plt.close()