import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

base = pd.read_csv('Datasets/strokeMod.csv', na_values=["Unknown", "N/A"])
imputer = SimpleImputer(strategy='mean')
base['bmi'] = imputer.fit_transform(base[['bmi']])

base = base.dropna(subset=['smoking_status'])

dados_selecionados = base[['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']]
print(np.unique(base['gender']))
print(np.unique(base['ever_married']))
print(np.unique(base['work_type']))
print(np.unique(base['Residence_type']))
print(np.unique(base['smoking_status']))

columns_to_encode = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

print(base)
# #['Female' 'Male' 'Other']
# # Inicialize o OneHotEncoder
# encoder = OneHotEncoder()

# # Ajuste e transforme os dados
# encoded_data = encoder.fit_transform(base[['smoking_status']])

# # Converta a matriz esparsa em uma matriz densa
# encoded_data_dense = encoded_data.toarray()
# base['smoking_status'] = encoded_data_dense


encoder = OneHotEncoder(sparse_output=False,handle_unknown='error')
retorno = encoder.fit_transform(base[['smoking_status']])
print(retorno)




# # Exiba o resultado
# print(base[0])

