from clean_data import df4model as df, dev_availabel


all_countries = df['Country'].unique().tolist()
all_countries.sort()

all_jobs = [i.replace("DevType_", "") for i in dev_availabel]
all_languages = [col.replace("Language_", "") for col in df.columns if col.startswith('Language_')]
all_webframe = [col.replace("Webframe_", "") for col in df.columns if col.startswith('Webframe_')]
all_platform = [col.replace("Platform_", "") for col in df.columns if col.startswith('Platform_')]
all_database= [col.replace("Database_", "") for col in df.columns if col.startswith('Database_')]


all_edu = [
         'Primary/elementary school',
         'Secondary school',
         'Some college/university study without earning a degree',
         'Associate degree (A.A., A.S., etc.)', 
         'Bachelor’s degree (B.A., B.S., B.Eng., etc.)',
         'Master’s degree (M.A., M.S., M.Eng., MBA, etc.)',
         'Professional degree (JD, MD, etc.)',
         'Other doctoral degree (Ph.D., Ed.D., etc.)'
]