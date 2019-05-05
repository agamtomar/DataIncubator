

# Importing modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('NYPD_Motor_Vehicle_Collisions.csv', low_memory=False)
data['DATE'] = pd.to_datetime(data['DATE'], format='%m/%d/%Y')

# Filtering dataset upto December 31, 2018
data = data[data['DATE'] <= pd.to_datetime('12/31/2018', format='%m/%d/%Y')]


# What is the total number of persons injured in the dataset (up to December 31, 2018?)
print('Total number of persons injured in the dataset (up to December 31, 2018: %d'
      % data[data['DATE'] <= pd.to_datetime('12/31/2018', format='%m/%d/%Y')]['NUMBER OF PERSONS INJURED'].sum())


# What proportion of all collisions in 2016 occured in Brooklyn? Only consider entries with a non-null value for BOROUGH.
BOROUGH_set = set(data['BOROUGH'].value_counts().keys())
data_2016 = data[(data['DATE'] >= pd.to_datetime('01/01/2016', format='%m/%d/%Y')) &
     (data['DATE'] <= pd.to_datetime('12/31/2016', format='%m/%d/%Y'))]

print('proportion of all collisions in 2016 occurred in Brooklyn: %.10f',
      data_2016[data_2016['BOROUGH'] == 'BROOKLYN'].shape[0]/
                  (data_2016[data_2016['BOROUGH'].isin(BOROUGH_set)].shape[0]))


# What proportion of collisions in 2016 resulted in injury or death of a cyclist?
# P(Injury or death) = 1 - P(No injury and No death)
print('Proportion of collisions in 2016 resulted in injury or death of a cyclist: %.10f'
      % (data_2016[(data_2016['NUMBER OF CYCLIST INJURED'] != 0) | (data_2016['NUMBER OF CYCLIST KILLED'] != 0)].shape[0]
      /(data_2016.shape[0])))


# For each borough, compute the number of accidents per capita involving alcohol in 2017. Report the highest rate among
# the 5 boroughs. Use populations as given by https://en.wikipedia.org/wiki/Demographics_of_New_York_City.

BOROUGH_Population = {'BRONX': 1471160, 'BROOKLYN': 2648771, 'MANHATTAN': 1664727,
                      'QUEENS': 2358582, 'STATEN ISLAND': 479458}


data_2017 = data[(data['DATE'] >= pd.to_datetime('01/01/2017', format='%m/%d/%Y')) &
     (data['DATE'] <= pd.to_datetime('12/31/2017', format='%m/%d/%Y'))]

# accidents involving alcohol in 2017
accidents_alcohol_2017 = data_2017[(data_2017['CONTRIBUTING FACTOR VEHICLE 1'] == 'Alcohol Involvement')
          |(data_2017['CONTRIBUTING FACTOR VEHICLE 2'] == 'Alcohol Involvement')
          |(data_2017['CONTRIBUTING FACTOR VEHICLE 3'] == 'Alcohol Involvement')
          |(data_2017['CONTRIBUTING FACTOR VEHICLE 4'] == 'Alcohol Involvement')
        |(data_2017['CONTRIBUTING FACTOR VEHICLE 5'] == 'Alcohol Involvement')]['BOROUGH'].value_counts()

# accidents per capita involving alcohol
accidents_borough_per_capita_dict = {}
for bor in BOROUGH_set:
    accidents_borough_per_capita_dict[bor] = accidents_alcohol_2017[bor]/BOROUGH_Population[bor]
    print('Accidents per capita involving alcohol in %s during 2017: %f' % (bor, accidents_alcohol_2017[bor]/BOROUGH_Population[bor]))

# Maximum value
print(max(accidents_borough_per_capita_dict, key=accidents_borough_per_capita_dict.get),
      accidents_borough_per_capita_dict[max(accidents_borough_per_capita_dict, key=accidents_borough_per_capita_dict.get)])





