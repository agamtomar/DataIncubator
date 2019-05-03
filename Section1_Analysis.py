

# Importing modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('NYPD_Motor_Vehicle_Collisions.csv', low_memory=False)
data['DATE'] = pd.to_datetime(data['DATE'], format='%m/%d/%Y')


# What is the total number of persons injured in the dataset (up to December 31, 2018?)
print('Total number of persons injured in the dataset (up to December 31, 2018: %d'
      % data[data['DATE'] <= pd.to_datetime('12/31/2018', format='%m/%d/%Y')]['NUMBER OF PERSONS INJURED'].sum())


# What proportion of all collisions in 2016 occured in Brooklyn? Only consider entries with a non-null value for BOROUGH.
# TODO Complete this problem.
df1 = data[(data['DATE'] >= pd.to_datetime('01/01/2016', format='%m/%d/%Y')) &
     (data['DATE'] <= pd.to_datetime('12/31/2016', format='%m/%d/%Y')) &
     (data['BOROUGH'] != 'BROOKLYN')]

