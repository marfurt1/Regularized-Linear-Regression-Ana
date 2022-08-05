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



#cargo los modelos
#loaded_model = pickle.load(open(filename, 'rb'))
filename='../models/final_ols_model.sav'
modelo_ols = pickle.load(open("../models/final_ols_model.sav", 'rb'))

filename='../models/final_lasso_model.sav'
modelo_lasso = pickle.load(open("../models/final_lasso_model.sav", 'rb'))

#Predict using the model whith new data

# modelo.predict(modelo_datos.transform([[40,1,22,1,1,1,0,0]])))



