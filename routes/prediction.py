from http.client import HTTPException
from telnetlib import STATUS
from fastapi import APIRouter
from schemas.prediction import Prediction
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

prediction = APIRouter()

posts = []

@prediction.get("/prediction")
def get_prediction():
    return posts

@prediction.get("/prediction/{id_pre}")
def get_prediction_id(id_pre : int):
    for post in posts:
        if post["id"] == id_pre:
            return post

    return "Id not found"

@prediction.post("/")
def create_data(prediction: Prediction):
    mnb = joblib.load('routes/Naive-bayes-multiclasses.pkl')

    prediction.hash = gh.encode(prediction.latitude, prediction.longitude, precision=7)

    le_hash = preprocessing.LabelEncoder()

    hashpr = np.array([prediction.hash])
    hashpr = pd.Series(hashpr)
    le_hash.fit(hashpr)
    prediction.hash2 = le_hash.transform(hashpr)

    data_prueba = {
        'hash' : prediction.hash2,
        'year' : [prediction.year],
        'month': [prediction.month],
        'hour': [prediction.hour],
        'dayOfTheWeek' : [prediction.dayOfTheWeek],
        'dayOfTheMonth' : [prediction.dayOfTheMonth],
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
      'id' : prediction.id,
      'hash': prediction.hash,
      'year':prediction.year,
      'month': prediction.month,
      'hour':prediction.hour,
      'dayOfTheWeek': prediction.dayOfTheWeek,
      'dayOfTheMonth':prediction.dayOfTheMonth,
      'latitude': prediction.latitude,
      'longitude':prediction.longitude,
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
    return new_method
