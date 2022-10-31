from http.client import HTTPException
from telnetlib import STATUS
from fastapi import APIRouter
from schemas.user import User
#import ML
import pandas as pd
import numpy as np  
import seaborn as sns
import sklearn.metrics 
from sklearn.metrics import accuracy_score 
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
import pygeohash as gh
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
import joblib

user = APIRouter()

posts = []

#CODIGO DEL API
@user.get("/prediction")
def get_data():
    return posts

@user.get("/prediction/{id_pre}")
def get_data_id(id_pre : int):
    for post in posts:
        if post["id"] == id_pre:
            return post

    return "Id not found"

@user.post("/data")
def create_data(user: User):
    mnb = joblib.load('routes/Naive-bayes-multiclasses.pkl')

    user.hash = gh.encode(user.latitude, user.longitude, precision=7)

    le_hash = preprocessing.LabelEncoder()

    hashpr = np.array([user.hash])
    hashpr = pd.Series(hashpr)
    le_hash.fit(hashpr)
    user.hash2 = le_hash.transform(hashpr)

    data_prueba = {
        'hash' : user.hash2,
        'year' : [user.year],
        'month': [user.month],
        'hour': [user.hour],
        'dayOfTheWeek' : [user.dayOfTheWeek],
        'dayOfTheMonth' : [user.dayOfTheMonth],
    }

    data_prueba = pd.DataFrame(data_prueba)

    #------ datos a predecir
    y_pred=mnb.predict(data_prueba)

    if y_pred[0] == 0:
        result1 = 'HOMICIDIO CALIFICADO - ASESINATO'
    elif y_pred[0] == 1:
        result1 = 'HURTO'
    elif y_pred[0] == 2:
        result1 = 'HURTO AGRAVADO'
    elif y_pred[0] == 3:
        result1 = 'MICROMERCIALIZACIÓN DE DROGAS'
    elif y_pred[0] == 4:
        result1 = 'ROBO'
    else:
        result1 = 'ROBO AGRAVADO'

    # ------ datos del proba
    aux = mnb.predict_proba(data_prueba)

    new_method = {
      'id' : user.id,
      'hash': user.hash,
      'year':user.year,
      'month': user.month,
      'hour':user.hour,
      'dayOfTheWeek': user.dayOfTheWeek,
      'dayOfTheMonth':user.dayOfTheMonth,
      'latitude': user.latitude,
      'longitude':user.longitude,
      'homicidio calificado - asesinato': str(round(aux[0][0], 3)),
      'hurto': str(round(aux[0][1], 3)),
      'hurto agravado': str(round(aux[0][2], 3)),
      'microcomercializacion de drogas': str(round(aux[0][3], 3)),
      'robo': str(round(aux[0][4], 3)),
      'robo agravado':str(round(aux[0][5], 3)), 
      'prediction':result1
      }

    posts.append(new_method)
    print(new_method)
    return "Data saved"
