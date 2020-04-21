#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 21:52:47 2020

@author: mit
"""



import argparse
import os
import pandas as pd
import numpy as np
import pickle
import json

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn_pandas import DataFrameMapper
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import math
import azureml.core
from azureml.core import Run
from azureml.core.model import Model
from azure.storage.blob import BlockBlobService
from io import StringIO

print("In train.py")
print("As a data scientist, this is where I write my training code.")

parser = argparse.ArgumentParser("train")
  
parser.add_argument("--model_name", type=str, help="model name", dest="model_name", required=True)
  
args = parser.parse_args()
  
print("Argument 1: %s" % args.model_name)
  

block_blob_service = BlockBlobService(account_name = 'mlworkshop',
                              account_key = 'kNYne8MDwB5flQnDnW6x8aX4MTNRZP0eraAEIM/040jQdC4gwUgd1ZR23MGzR0+8qMb1xcApb/n0WUzK1vXOwg==' )
#get data from blob storage in the form of bytes
blob_byte_data = block_blob_service.get_blob_to_bytes('final','temp1.csv')
#convert to bytes data into pandas df to fit scaler transform
s=str(blob_byte_data.content,'utf-8')
bytedata = StringIO(s)
df=pd.read_csv(bytedata)


x_df = df.drop(['Solo_Insulin'], axis=1)
y_df = df['Solo_Insulin']

X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=0)

categorical = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id',
       'race', 'gender', 'age', 'change', 'readmitted']

numerical = ['patient_nbr', 'number_diagnoses', 'time_in_hospital', 'encounter_id',
       'num_lab_procedures', 'num_procedures', 'num_medications',
       'number_outpatient', 'number_emergency', 'number_inpatient',
       'Total_drugs', 'Solo_Insulin', 'diagnosis']

numeric_transformations = [([f], Pipeline(steps=[
    ('scaler', StandardScaler())])) for f in numerical]
    
categorical_transformations = [([f], OneHotEncoder(handle_unknown='ignore', sparse=False)) for f in categorical]

transformations = numeric_transformations + categorical_transformations

clf = Pipeline(steps=[('preprocessor', DataFrameMapper(transformations)),
                      ('classifier', RandomForestClassifier(max_depth=5))])

clf.fit(X_train, y_train)

os.makedirs('./outputs', exist_ok=True)
model_file_name = args.model_name + '.pkl'
with open(model_file_name, 'wb') as file:
    joblib.dump(value=clf, filename=os.path.join('./outputs',
                                                 model_file_name))

run = Run.get_context()


run = Run.get_context()

y_predict = clf.predict(X_test)
score = accuracy_score(y_test, y_predict)
run.log('Accuracy', score, 'The Accuracy score on test data for LogisticRegression')
print('The Accuracy score on test data for RandomForestClassifier: ', score)



os.chdir("./outputs")

model_description = 'This model was trained using GradientBoostingRegressor.'
model = Model.register(
    model_path=model_file_name,  # this points to a local file
    model_name=args.model_name ,  # this is the name the model is registered as
    tags={"type": "classifier", "Score": score, "run_id": run.id},
    description=model_description,
    workspace=run.experiment.workspace
)

os.chdir("..")

print("Model registered: {} \nModel Description: {} \nModel Version: {}".format(model.name, 
                                                                                model.description, model.version))



