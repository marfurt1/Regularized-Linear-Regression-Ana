### Lasso to select features

# Import libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_validate
import statsmodels.api as sm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# Read dataset
url = 'https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/dataset.csv'
df_raw = pd.read_csv(url)

# Make a copy of the raw datset
df = df_raw.copy()

# Split dataframe in features and target
# The variable chosen as target is 'Total Specialist Physicians (2019)'
X= df.drop(['CNTY_FIPS','fips','Active Physicians per 100000 Population 2018 (AAMC)','Total Active Patient Care Physicians per 100000 Population 2018 (AAMC)', 'Active Primary Care Physicians per 100000 Population 2018 (AAMC)', 'Active Patient Care Primary Care Physicians per 100000 Population 2018 (AAMC)','Active General Surgeons per 100000 Population 2018 (AAMC)','Active Patient Care General Surgeons per 100000 Population 2018 (AAMC)','Total nurse practitioners (2019)','Total physician assistants (2019)','Total physician assistants (2019)','Total Hospitals (2019)','Internal Medicine Primary Care (2019)','Family Medicine/General Practice Primary Care (2019)','STATE_NAME','COUNTY_NAME','ICU Beds_x','Total Specialist Physicians (2019)'], axis=1)
y=df['Total Specialist Physicians (2019)']


## Model 2: alpha selected by CV

pipe2 = make_pipeline(StandardScaler(), Lasso())

params={
    'lasso__fit_intercept':[True,False],
    'lasso__alpha':10.0**np.arange(-2, 6, 1)
}

#setting up the grid search
gs=GridSearchCV(pipe2,params,n_jobs=-1,cv=5)


# get coefficients
coef_list=model2[1].coef_

# Location of Lasso coefficients
loc2=[i for i, e in enumerate(coef_list) if e != 0]



#cargo los modelos
#loaded_model = pickle.load(open(filename, 'rb'))
filename='../models/final_ols_model.sav'
modelo_ols = pickle.load(open("../models/final_ols_model.sav", 'rb'))

filename='../models/final_lasso_model.sav'
modelo_ols = pickle.load(open("../models/final_lasso_model.sav", 'rb'))

#Predict using the model whith new data

# modelo.predict(modelo_datos.transform([[40,1,22,1,1,1,0,0]])))



