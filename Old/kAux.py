from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import statistics
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import copy

df = pd.read_csv("Datasets/Presos_MissForestv2.csv", delimiter=';', skipinitialspace=True)
#Get the input features
columns = df.columns
class_name = 'WrittenUpOrFoundGuiltyOfBreakingAnyRules'
columns_tmp = list(columns)
columns_tmp.remove(class_name)


x, x_test, y, y_test = train_test_split(df[columns_tmp], df[class_name], test_size=0.1)
model = RandomForestClassifier(criterion = 'entropy', max_depth = 10, n_estimators = 100)
model.fit(x,y)
predicao = model.predict(x_test)
print(accuracy_score(y_test,predicao))

#DEFININDO QUANTIDADE DE FOLDS
from sklearn.model_selection import KFold
kf = KFold(n_splits=3)
melhor = 0


fold = 1
labels = ['0', '1']
labels2 = [0, 1]

precision_valid = [0,0]
recall_valid = [0,0]
fscore_valid = [0,0]

for train_index, valid_index in kf.split(x):
    x_train = x.iloc[train_index].loc[:]
    y_train = y.iloc[train_index]    
    
    x_valid = x.iloc[valid_index].loc[:]
    y_valid = y.iloc[valid_index]
    
    model.fit(x_train, y_train)
    predito = model.predict(x_valid)
    
    #Calculando as métricas
    precision,recall,fscore,support = precision_recall_fscore_support(y_valid, predito, average=None)   
    precision_valid = np.add(precision_valid,precision)
    recall_valid = np.add(recall_valid,recall)
    fscore_valid = np.add(fscore_valid,fscore)     

    acuracia = np.mean(y_valid == predito)
    
    if acuracia > melhor:          
        melhor = acuracia
        best_model = copy.deepcopy(model)
    
    print('FOLD: ' + str(fold))
    print(metrics.classification_report(y_valid,predito, target_names=labels))
    
    cm = metrics.confusion_matrix(y_valid,predito, labels=labels2)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels2)
    disp.plot()
    # plt.show()
    
    fold += 1
    print('---------------------------------------------------------')

# print("Precisão média na validação das classes Inferior e Superior: " , np.round(precision_valid/10,2))
# print("----------------------------------------------------------------")
# print("Recall médio na validação das classes Inferior e Superior: " ,  np.round(recall_valid/10,2))
# print("----------------------------------------------------------------")
# print("F1 score médio na validação das classes Inferior e Superior: " ,  np.round(fscore_valid/10,2))
#Teste
# p = best_model.predict(x_test)

# print(classification_report(y_test, p))