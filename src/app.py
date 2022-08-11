
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
df = pd.read_csv(url)

#cargo los modelos
filename1='../models/final_lasso_model.sav'
modelo_lasso = pickle.load(open(filename1, 'rb'))

filename='../models/final_ols_model.sav'
modelo_ols = pickle.load(open(filename, 'rb'))

#Predict using the model whith new data
datos=pd.read_csv('/workspace/Regularized-Linear-Regression-Ana/data/processed/datos_para_testeo', index_col=0)

print('El numero de camas predicho por el modelo lasso es: {}'.format(modelo_lasso.predict(datos)[0])) #lasso no necesita transformacion

X_ols = datos[['0-9',
 '0-9 y/o % of total pop',
 '20-29 y/o % of total pop',
 '40-49 y/o % of total pop',
 '50-59',
 '50-59 y/o % of total pop',
 '80+',
 'White-alone pop',
 '% White-alone',
 'Black-alone pop',
 '% Black-alone',
 'Asian-alone pop',
 'Hawaiian/Pacific Islander-alone pop',
 '% Hawaiian/PI-alone',
 'Two or more races pop',
 '% Two or more races',
 'POP_ESTIMATE_2018',
 'N_POP_CHG_2018',
 'GQ_ESTIMATES_2018',
 'Less than a high school diploma 2014-18',
 'High school diploma only 2014-18',
 "Some college or associate's degree 2014-18",
 "Bachelor's degree or higher 2014-18",
 'Percent of adults with less than a high school diploma 2014-18',
 'POVALL_2018',
 'PCTPOVALL_2018',
 'PCTPOV017_2018',
 'PCTPOV517_2018',
 'CI90LBINC_2018',
 'Unemployed_2018',
 'Unemployment_rate_2018',
 'Med_HH_Income_Percent_of_State_Total_2018',
 'anycondition_Upper 95% CI',
 'Obesity_number',
 'Heart disease_number',
 'COPD_Upper 95% CI',
 'diabetes_number']]

print('El numero de camas predicho por el modelo ols es: {}'.format(modelo_ols.predict(X_ols)[0]))



