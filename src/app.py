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

datos=[]
X_para_ols= sm.add_constant(datos) 

print('resultado de modelo ols: {}'.format(modelo_ols.predict(X_para_ols))) 

print('resultado de modelo lasso: {}'.format(modelo_lasso.predict(datos))) #lasso no necesita transformacion


