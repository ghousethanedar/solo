#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 21:48:25 2020

@author: mit
"""

import json
import numpy as np
import pandas as pd

columns =['patient_nbr', 'number_diagnoses', 'admission_type_id',
       'discharge_disposition_id', 'admission_source_id', 'time_in_hospital',
       'encounter_id', 'race', 'gender', 'age', 'num_lab_procedures',
       'num_procedures', 'num_medications', 'number_outpatient',
       'number_emergency', 'number_inpatient', 'change', 'readmitted',
       'Total_drugs', 'diagnosis']
def init():
    try:
        # One-time initialization of predictive model and scaler
        from azureml.core.model import Model
        from sklearn.externals import joblib
        global model
        
        model_name = 'MODEL-NAME' # Placeholder model name
        print('Looking for model path for model: ', model_name)
        model_path = Model.get_model_path(model_name=model_name)
        print('Looking for model in: ', model_path)
        model = joblib.load(model_path)
        print('Model loaded...')

    except Exception as e:
        print('Exception during init: ', str(e))

def run(input_json):     
    try:
        inputs = json.loads(input_json)
        data_df = pd.DataFrame(np.array(inputs).reshape(-1, len(columns)), columns = columns)
        # Get the predictions...
        prediction = model.predict(data_df).tolist()
        prediction = json.dumps(prediction)
    except Exception as e:
        prediction = str(e)
    return prediction

