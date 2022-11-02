from fastapi import FastAPI
from routes.prediction import prediction

app = FastAPI()

app.include_router(prediction)
