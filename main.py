from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('delivery_prediction.joblib')
selector = joblib.load('selector.joblib')
scaler = joblib.load('scaler.joblib')

class Shipment(BaseModel):
    arrival: str
    process_start: str
    process_finished: str

origins = ["*"]  # This will allow all sites to access your backend
methods = ["POST"]  # This will only allow the POST method
app.add_middleware( 
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=methods,
)

@app.post("/predict/")
async def predict(shipment: Shipment):
    new_shipment = pd.DataFrame({'ARRIVAL': [shipment.arrival], 'PROCESS STARTED': [shipment.process_start],'PROCESS FINISHED \n(yyyy/mm/dd)': [shipment.process_finished]})

    new_shipment['ARRIVAL'] = pd.to_datetime(new_shipment['ARRIVAL'])
    new_shipment['PROCESS STARTED'] = pd.to_datetime(new_shipment['PROCESS STARTED'])
    new_shipment['PROCESS FINISHED \n(yyyy/mm/dd)'] = pd.to_datetime(new_shipment['PROCESS FINISHED \n(yyyy/mm/dd)'])

    new_shipment['ARRIVAL'] = new_shipment['ARRIVAL'].apply(lambda x: x.timestamp())
    new_shipment['PROCESS STARTED'] = new_shipment['PROCESS STARTED'].apply(lambda x: x.timestamp())
    new_shipment['PROCESS FINISHED \n(yyyy/mm/dd)'] = new_shipment['PROCESS FINISHED \n(yyyy/mm/dd)'].apply(lambda x: x.timestamp())
    X = new_shipment[['ARRIVAL', 'PROCESS STARTED', 'PROCESS FINISHED \n(yyyy/mm/dd)']]
    X_new = selector.transform(X)
    X_new = scaler.transform(X_new)
    y_pred = model.predict(X_new)

    return pd.to_datetime(y_pred[0], unit='s').date()
