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


filename='../models/final_ols_model.sav'
modelo_ols = pickle.load(open(filename, 'rb'))

filename1='../models/final_lasso_model.sav'
modelo_lasso = pickle.load(open(filename1, 'rb'))

#Predict using the model whith new data

datos=[1.38300000e+03, 1.87000000e+02, 1.35213304e+01, 1.47000000e+02,
       1.06290673e+01, 1.57000000e+02, 1.13521330e+01, 1.32000000e+02,
       9.54446855e+00, 1.34000000e+02, 9.68908171e+00, 1.97000000e+02,
       1.42443962e+01, 2.05000000e+02, 1.48228489e+01, 1.20000000e+02,
       8.67678959e+00, 1.04000000e+02, 7.51988431e+00, 1.33400000e+03,
       9.64569776e+01, 9.00000000e+00, 6.50759219e-01, 1.20000000e+01,
       8.67678959e-01, 3.00000000e+00, 2.16919740e-01, 5.00000000e+00,
       3.61532900e-01, 2.00000000e+01, 1.44613160e+00, 1.38300000e+03,
       1.60000000e+01, 1.40000000e+01, 1.31000000e+01, 1.31000000e+01,
       0.00000000e+00, 0.00000000e+00, 1.16000000e+01, 1.16000000e+01,
       3.90000000e+01, 3.61000000e+02, 3.47000000e+02, 2.04000000e+02,
       4.10000000e+00, 3.80000000e+01, 3.65000000e+01, 2.15000000e+01,
       1.88000000e+02, 1.38000000e+01, 1.92000000e+01, 1.98000000e+01,
       4.47040000e+04, 3.99280000e+04, 4.94800000e+04, 9.63000000e+02,
       9.46000000e+02, 1.70000000e+01, 1.80000000e+00, 4.47040000e+04,
       6.21000000e+01, 1.36800000e+03, 3.70000000e+02, 2.70000000e+01,
       8.00000000e+00, 1.07400000e+03, 3.74000000e+01, 3.51000000e+01,
       3.98000000e+01, 4.02000000e+02, 2.59000000e+01, 2.41000000e+01,
       2.74000000e+01, 2.78000000e+02, 8.60000000e+00, 7.80000000e+00,
       9.50000000e+00, 9.30000000e+01, 8.30000000e+00, 7.10000000e+00,
       9.50000000e+00, 8.90000000e+01, 1.09000000e+01, 1.00000000e+01,
       1.19000000e+01, 1.17000000e+02, 3.70000000e+00, 3.40000000e+00,
       4.00000000e+00, 3.90000000e+01, 6.00000000e+00]

X_para_ols= sm.add_constant(datos) 

print('resultado de modelo ols: {}'.format(modelo_ols.predict(X_para_ols))) 

print('resultado de modelo lasso: {}'.format(modelo_lasso.predict(datos))) #lasso no necesita transformacion


