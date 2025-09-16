# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle 
import sklearn

loaded_model = pickle.load(open('D:\ml deploy/trained_model.sav' , 'rb'))

input_data=(2,88,74,19,0,29.0,0.229,22)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=loaded_model.predict(input_data_reshaped)
print(prediction)

if prediction[0]==0:
  print("The patient does not have diabetics")
else:
  print("The patient has diabetics")
  
  
loaded_model = pickle.load(open('D:\ml deploy/Heart_trained_model.sav' , 'rb'))
input_data=(57,1,0,140,192,0,1,148,0,0.4,1,0,1)
input_data_as_numpy_array=np.asarray(input_data)
input_data_as_numpy_array_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=loaded_model.predict(input_data_as_numpy_array_reshaped)

if prediction[0]==0:
  print('The patient has no heart disease')
else:
  print('The patient has heart disease')
  