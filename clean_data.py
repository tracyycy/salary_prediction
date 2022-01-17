#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from zipfile import ZipFile
import re

# In[2]:


pd.set_option("display.max_columns", 101)


# # Read data

# In[3]:


with ZipFile(f'./data/stack-overflow-developer-survey-2021.zip') as myzip:
    with myzip.open('survey_results_public.csv') as myfile:
        df = pd.read_csv(myfile)




# [GNI per capita, Atlas method](https://data.worldbank.org/indicator/NY.GNP.PCAP.CD)

# In[6]:


gnp = pd.read_csv('./data/gnp.csv', skiprows=4)
gnp.shape


# In[7]:



country = pd.read_csv('./data/country.csv')
country.shape


# In[9]:



# remove null-salary
df.dropna(subset=['ConvertedCompYearly'], inplace=True)


# In[13]:


cols = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', '2020']
gnp_dict = gnp[cols].set_index('Country Name').to_dict()


# In[14]:


set(df['Country']) - set(gnp['Country Name'])


# In[15]:


rename_country = {
    'Cape Verde': 'Cabo Verde',
    'Congo, Republic of the...': 'Congo, Rep.',
    "Côte d'Ivoire": "Cote d'Ivoire",
    'Democratic Republic of the Congo': 'Congo, Dem. Rep.',
    'Egypt': 'Egypt, Arab Rep.',
    'Gambia': 'Gambia, The',
    'Hong Kong (S.A.R.)': 'Hong Kong SAR, China',
    'Iran, Islamic Republic of...': 'Iran, Islamic Rep.',
    #'Kyrgyzstan',
    "Lao People's Democratic Republic": 'Lao PDR',
    'Libyan Arab Jamahiriya': 'Libya',
    #'Nomadic',
    #'North Korea',
    #'Palestine',
    'Republic of Korea': 'Korea, Rep.',
    'Republic of Moldova': 'Moldova',
    'Saint Kitts and Nevis': 'St. Kitts and Nevis',
    'Saint Lucia': 'St. Lucia',
    'Saint Vincent and the Grenadines': 'St. Vincent and the Grenadines',
    'Slovakia': 'Slovak Republic',
    'South Korea': 'Korea, Rep.',
    'Swaziland': 'Eswatini ',
    #'Taiwan',
    'The former Yugoslav Republic of Macedonia': 'North Macedonia',
    'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
    'United Republic of Tanzania': 'Tanzania',
    'United States of America': 'United States',
    'Venezuela, Bolivarian Republic of...': 'Venezuela, RB',
    'Viet Nam': 'Vietnam',
    'Yemen': 'Yemen, Rep.'
}


# In[16]:


df['Country'].replace(rename_country, inplace=True)


# In[17]:


df['gnp'] = df['Country'].map(gnp_dict['2020'].get)
df['country_code'] = df['Country'].map(gnp_dict['Country Code'].get)


# In[18]:


# remove null-salary
df.dropna(subset=['gnp'], inplace=True)
df.shape


# In[19]:


df['ratio'] = df['ConvertedCompYearly'] / df['gnp']


# In[20]:


df['YearsCodePro'].replace({
    'Less than 1 year': 0,
    'More than 50 years': 51
}, inplace=True)

df['YearsCodePro'].dtype


# In[21]:


df['YearsCodePro'] = df['YearsCodePro'].astype(float)


# In[25]:
edu_rename = {
    'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)': 'Secondary school'
}

ed2int = {
       'Primary/elementary school': 0,
       'Secondary school': 0,
       'Some college/university study without earning a degree': 1,
       'Associate degree (A.A., A.S., etc.)': 1, 
       'Bachelor’s degree (B.A., B.S., B.Eng., etc.)': 2,
       'Master’s degree (M.A., M.S., M.Eng., MBA, etc.)': 3,
       'Professional degree (JD, MD, etc.)': 4,
       'Other doctoral degree (Ph.D., Ed.D., etc.)': 4,
       'Something else': np.nan
}


# In[26]:


df['education'] = df['EdLevel'].replace(edu_rename).map(ed2int)
df['education'].value_counts()


# # Extract

# In[27]:


cond1 = df['ratio'] >= 0.5
cond2 = df['ratio'] <= 20
df2 = df[cond1 & cond2].copy()


df2['YearsCodePro2'] = df2['YearsCodePro'].clip(upper=20)


# In[31]:


cols = [
    'ResponseId', 'Country', 'US_State', 'DevType', 'EdLevel', 'education',
    'LanguageHaveWorkedWith', 'DatabaseHaveWorkedWith', 'PlatformHaveWorkedWith', 
    'WebframeHaveWorkedWith', 'MiscTechHaveWorkedWith', 'ToolsTechHaveWorkedWith', 
    'ConvertedCompYearly', 'gnp', 'country_code', 'ratio', 'YearsCodePro', 'YearsCodePro2'
]


# In[32]:


df2 = df2[cols]
df2.columns = [col.replace('HaveWorkedWith', '') for col in df2.columns]


# In[33]:


dict_country = country.set_index('Country Code').to_dict()


# In[34]:


df2['Region'] = df2['country_code'].map(dict_country['Region'])
df2['IncomeGroup'] = df2['country_code'].map(dict_country['IncomeGroup'])


# In[35]:


df2[['Region', 'IncomeGroup']].isnull().sum()



# In[41]:


def multi_choice2df(feat_col, data, sep, drop=None):
    tmp = data[feat_col].str.split(sep, expand=True).stack().reset_index(drop=True, level=1).rename('choice')
    tmp = pd.DataFrame(tmp).reset_index().rename(columns={'index': 'OriginalIndex'})
    tmp['present'] = 1
    tmp = tmp.pivot(index='OriginalIndex', columns='choice', values='present').fillna(0)
    
    if drop:
        keep = (tmp.mean() >= drop) & (tmp.mean() <= (1 - drop))
        tmp = tmp[keep[keep].index]
        
    return tmp


# In[42]:


dict_tech = {}
for tech in ['Language', 'Database', 'Platform', 'Webframe', 'MiscTech', 'ToolsTech', 'DevType']:
    dict_tech[tech] = multi_choice2df(tech, df2, ';')


# In[43]:


df2.columns


# In[56]:


cols = [ 'ratio', 'YearsCodePro', 'gnp', 'Country', 'Region', 'IncomeGroup', 'EdLevel', 'education']
df2[cols].isnull().sum()


# In[57]:


df3 = pd.DataFrame(index=df2.index)
keep_types = ['Language',  'Platform', 'Webframe', 'DevType', 'Database'] # 'MiscTech', 'ToolsTech',
for tech in  keep_types:
    tmp = dict_tech[tech]
    tmp.columns = [tech + '_' + col for col in tmp.columns]
    df3 = pd.concat([df3, tmp], axis=1)
    
df3 = pd.concat([df2[cols], df3.fillna(0).astype(int)], axis=1)


choices = ['AWS', 'Google Cloud Platform', 'Microsoft Azure']
choices = ['Platform_' + choice for choice in choices]
df3['Platform_AWS/GCP/Azure'] = df3[choices].max(axis=1)
df3.drop(choices, axis=1, inplace=True)


choices = ['Microsoft SQL Server', 'MySQL', 'PostgreSQL', 'SQLite']
choices = ['Database_' + choice for choice in choices]
df3['Database_SQL'] = df3[choices].max(axis=1)
df3.drop(choices, axis=1, inplace=True)


# salary median by role
dev_cols = [col for col in df3.columns if col.startswith('DevType')]

salary_median = {}
for dev in dev_cols:
    cond = df3[dev] == 1
    m = df3.loc[cond, 'ratio'].median()
    salary_median[dev.replace('DevType_', '')] = m
    

role = df['DevType'].fillna('Not provided').str.split(';', expand=True).stack()\
        .reset_index(drop=True, level=1).rename('DevType')
role = pd.DataFrame(role).reset_index().rename(columns={'index': 'ResponseId'})
role['role_adjustment'] = role['DevType'].map(salary_median)
role_max = role.groupby("ResponseId").max("multi_role")


df3 = df3.join(role_max)

df4model = df3[df3['IncomeGroup'].isin(['High income'])].copy() #, 'Upper middle income'
multi_choice = [col for col in df4model.columns if col.startswith(tuple(keep_types)) ]

thres = 1000
m = df4model[list(multi_choice)].sum()
keep = (m >= thres) 
keep_multi = list(keep[keep].index)

keep_dev = [col for col in keep_multi if col.startswith('DevType_')]
multi_feat = list(set(keep_multi) - set(keep_dev))
num_feat =  ['YearsCodePro', 'gnp', 'education', 'role_adjustment']
cat_feat = ['Region']

feat4model = {
    'multi': multi_feat,
    "num": num_feat,
    "cat": cat_feat
}

# group data by job
dev_availabel = []
data4model = {}
for job in keep_dev:
    if 'other' in job.lower():
        continue
    cond = df4model[job] == 1
    feats = ['ratio'] + cat_feat + num_feat + multi_feat
    tmp = df4model.loc[cond, feats].copy()
    job_renamed = re.sub("[\W]+", "_", job)
    data4model[job_renamed] = tmp
    dev_availabel.append(job)


# In[58]:

def group_others(x, thres=0.05):
    tmp = x.value_counts(normalize=True)
    keep = tmp[tmp > thres].index.tolist()
    x = np.where(x.isin(keep), x, 'Others')
    return x