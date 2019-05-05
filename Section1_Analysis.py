

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


# Obtain the number of vehicles involved in each collision in 2016. Group the collisions by zip code and compute
# the sum of all vehicles involved in collisions in each zip code, then report the maximum of these values.

# making vehicle type set for whole dataset
vehicle_set = set(data['VEHICLE TYPE CODE 1'].value_counts().keys())
vehicle_set.intersection(set(data['VEHICLE TYPE CODE 2'].value_counts().keys()))
vehicle_set.intersection(set(data['VEHICLE TYPE CODE 3'].value_counts().keys()))
vehicle_set.intersection(set(data['VEHICLE TYPE CODE 4'].value_counts().keys()))
vehicle_set.intersection(set(data['VEHICLE TYPE CODE 5'].value_counts().keys()))


# Took a lot of time to run
# saved the output as csv updated_NY_motor_collision.csv
"""
data = pd.concat([data, pd.Series(data=np.zeros((data.shape[0])), index=data.index, name='VEHICLES INVOLVED')], axis=1)
for idx in list(data.index):
    print(idx)
    temp = 0
    for i in range(1, 6):
        if data.loc[idx, 'VEHICLE TYPE CODE %s' % i] in vehicle_set:
            temp += 1

    data.loc[idx, 'VEHICLES INVOLVED'] = temp
"""

data_2016 = data[(data['DATE'] >= pd.to_datetime('01/01/2016', format='%m/%d/%Y')) &
     (data['DATE'] <= pd.to_datetime('12/31/2016', format='%m/%d/%Y'))]

ZIPCODES_set = set(data_2016['ZIP CODE'].value_counts().keys())

total_collisions = data_2016['VEHICLES INVOLVED'].groupby(data_2016['ZIP CODE'])

# Sum of all vehicles involved in collisions in each zip code
print(total_collisions.count())


print('Maximum Collision in zip code %s: %d' % (str(total_collisions.count().argmax()), int(total_collisions.count().max())))


# Consider the total number of collisions each year from 2013-2018. Is there an apparent trend?
# Fit a linear regression for the number of collisions per year and report its slope.

collisions_year_dict= {'2013': data[(data['DATE'] >= pd.to_datetime('01/01/2013', format='%m/%d/%Y')) &
     (data['DATE'] <= pd.to_datetime('12/31/2013', format='%m/%d/%Y'))].shape[0],
                       '2014': data[(data['DATE'] >= pd.to_datetime('01/01/2014', format='%m/%d/%Y')) &
     (data['DATE'] <= pd.to_datetime('12/31/2014', format='%m/%d/%Y'))].shape[0],
                       '2015': data[(data['DATE'] >= pd.to_datetime('01/01/2015', format='%m/%d/%Y')) &
     (data['DATE'] <= pd.to_datetime('12/31/2015', format='%m/%d/%Y'))].shape[0],
                       '2016': data[(data['DATE'] >= pd.to_datetime('01/01/2016', format='%m/%d/%Y')) &
     (data['DATE'] <= pd.to_datetime('12/31/2016', format='%m/%d/%Y'))].shape[0],
                       '2017': data[(data['DATE'] >= pd.to_datetime('01/01/2017', format='%m/%d/%Y')) &
     (data['DATE'] <= pd.to_datetime('12/31/2017', format='%m/%d/%Y'))].shape[0],
                       '2018': data[(data['DATE'] >= pd.to_datetime('01/01/2018', format='%m/%d/%Y')) &
     (data['DATE'] <= pd.to_datetime('12/31/2018', format='%m/%d/%Y'))].shape[0]}


from sklearn.linear_model import LinearRegression
X = np.array(list(collisions_year_dict.keys()), dtype=int).reshape(-1,1)
Y = np.array(list(collisions_year_dict.values()), dtype=int).reshape(-1,1)

reg = LinearRegression()
reg.fit(X, Y)

plt.scatter(X, Y)
plt.plot(np.arange(2000, 2050, 0.5), reg.predict(np.arange(2000, 2050, 0.5).reshape(-1,1)))
plt.show()

print('Slope of Linear fit: %f', reg.coef_[0][0])


