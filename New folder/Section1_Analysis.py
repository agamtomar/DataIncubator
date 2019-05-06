

# Importing modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading the dataset
data = pd.read_csv('NYPD_Motor_Vehicle_Collisions.csv', low_memory=False)
data['DATE'] = pd.to_datetime(data['DATE'], format='%m/%d/%Y')

# Filtering dataset upto December 31, 2018
data = data[data['DATE'] <= pd.to_datetime('12/31/2018', format='%m/%d/%Y')]


# What is the total number of persons injured in the dataset (up to December 31, 2018?)
print('1(a)')
print('Total number of persons injured in the dataset (up to December 31, 2018: %d'
      % data[data['DATE'] <= pd.to_datetime('12/31/2018', format='%m/%d/%Y')]['NUMBER OF PERSONS INJURED'].sum())


# What proportion of all collisions in 2016 occured in Brooklyn? Only consider entries with a non-null value for BOROUGH.
BOROUGH_set = set(data['BOROUGH'].value_counts().keys())
data_2016 = data[(data['DATE'] >= pd.to_datetime('01/01/2016', format='%m/%d/%Y')) &
     (data['DATE'] <= pd.to_datetime('12/31/2016', format='%m/%d/%Y'))]

print('1(b)')
print('proportion of all collisions in 2016 occurred in Brooklyn: %.10f',
      data_2016[data_2016['BOROUGH'] == 'BROOKLYN'].shape[0]/
                  (data_2016[data_2016['BOROUGH'].isin(BOROUGH_set)].shape[0]))


# What proportion of collisions in 2016 resulted in injury or death of a cyclist?
# P(Injury or death) = 1 - P(No injury and No death)
print('1(c)')
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
    # print('Accidents per capita involving alcohol in %s during 2017: %f' % (bor, accidents_alcohol_2017[bor]/BOROUGH_Population[bor]))

# Maximum value
print('1(d)')
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

"""
# Took a lot of time to run, # Using alternate approach below
# saved the output as csv updated_NY_motor_collision.csv
vehicle_involved_data = pd.read_csv('updated_NY_motor_collision.csv', low_memory=False)
vehicle_involved_series = pd.Series(data=vehicle_involved_data['VEHICLES INVOLVED'], index=vehicle_involved_data['index'])
data = pd.concat([data, vehicle_involved_series], axis=1)
del vehicle_involved_data


data = pd.concat([data, pd.Series(data=np.zeros((data.shape[0])), index=data.index, name='VEHICLES INVOLVED')], axis=1)
for idx in list(data.index):
    print(idx)
    temp = 0
    for i in range(1, 6):
        if data.loc[idx, 'VEHICLE TYPE CODE %s' % i] in vehicle_set:
            temp += 1

    data.loc[idx, 'VEHICLES INVOLVED'] = temp
"""

data_copy = pd.read_csv('NYPD_Motor_Vehicle_Collisions.csv', low_memory=False)
# Converting categorical data to numeric
for i in range(1, 6):
    data_copy['VEHICLE TYPE CODE %s' % i] = data_copy['VEHICLE TYPE CODE %s' % i].astype('category')

vehicle_categories = data_copy['VEHICLE TYPE CODE 1'].cat.categories
cat_columns = data_copy.select_dtypes(['category']).columns
data_copy[cat_columns] = data_copy[cat_columns].apply(lambda x: x.cat.codes)

# Converting numeric data to binary
for i in range(1, 6):
    data_copy['VEHICLE TYPE CODE %s' % i] = np.where(data_copy['VEHICLE TYPE CODE %s' % i] > 0, 1, 0)

# Creating vehicles involved column
data['VEHICLES INVOLVED'] = 0
for i in range(1, 6):
    data['VEHICLES INVOLVED'] += data_copy['VEHICLE TYPE CODE %s' % i]

del data_copy  # deleting data_copy

data_2016 = data[(data['DATE'] >= pd.to_datetime('01/01/2016', format='%m/%d/%Y')) &
     (data['DATE'] <= pd.to_datetime('12/31/2016', format='%m/%d/%Y'))]

ZIPCODES_set = set(data_2016['ZIP CODE'].value_counts().keys())
total_collisions = data_2016['VEHICLES INVOLVED'].groupby(data_2016['ZIP CODE'])

# Sum of all vehicles involved in collisions in each zip code
print('1(e)')
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

# Plotting code
# plt.scatter(X, Y)
# plt.plot(np.arange(2000, 2050, 0.5), reg.predict(np.arange(2000, 2050, 0.5).reshape(-1,1)))
# plt.show()

print('1(f)')
print('Slope of Linear fit: %f', reg.coef_[0][0])


# Do winter driving conditions lead to more multi-car collisions? Compute the rate of multi car collisions as
# the proportion of the number of collisions involving 3 or more cars to the total number of collisions for
# each month of 2017. Calculate the chi-square test statistic for testing whether a collision is more likely
# to involve 3 or more cars in January than in May.

data_2017 = data.loc[(data['DATE'] >= pd.to_datetime('01/01/2017', format='%m/%d/%Y')) &
     (data['DATE'] <= pd.to_datetime('12/31/2017', format='%m/%d/%Y')), ['DATE', 'VEHICLES INVOLVED']]

data_2017 = data_2017.set_index('DATE')
df1 = data_2017.resample('M')
df1.count()

multi_collision_2017 = data_2017[data_2017['VEHICLES INVOLVED'] >=3]
df2 = multi_collision_2017.resample('M')
df2.count()

multi_collision_2017_rate = df2.count()['VEHICLES INVOLVED']/df1.count()['VEHICLES INVOLVED']


# Do Winter driving conditions lead to more multi-car collisions?
print('1(g)')
multi_collision_2017_rate.plot()
plt.show()
print('Yes. Since rate for multi car collision is higher in winter months (Jan - May)')

#  chi-square test statistic for testing whether a collision is more likely to involve 3 or more cars
#  in January than in May

# Null Hypothesis - Jan and May have same rate for multi car collision

from scipy.stats import chisquare
obs = [2120, 6276]
exp = [int(df2.count().sum())/12, int(df2.count().sum())/12]
chi_squared_test = chisquare(f_obs=obs, f_exp=exp)

print('Chi-square test statistic: %f and p-value: %f' % (chi_squared_test[0], chi_squared_test[1]))
print('As p value is close to 0.0, we reject the null hypothesis')



# We can use collision locations to estimate the areas of the zip code regions. Represent each as an ellipse with
# semi-axes given by a single standard deviation of the longitude and latitude. For collisions in 2017, estimate
# the number of collisions per square kilometer of each zip code region. Considering zipcodes with at least 1000
# collisions, report the greatest value for collisions per square kilometer. Note: Some entries may have invalid
# or incorrect (latitude, longitude) coordinates. Drop any values that are invalid or seem unreasonable for New
# York City.

from geopy import distance

data_2017 = data.loc[(data['DATE'] >= pd.to_datetime('01/01/2017', format='%m/%d/%Y')) &
     (data['DATE'] <= pd.to_datetime('12/31/2017', format='%m/%d/%Y')), ['ZIP CODE', 'LATITUDE', 'LONGITUDE', 'LOCATION']]

# Indices where latitude or longitude locations are 0
missing_location_indices = data_2017[(data_2017['LATITUDE'].round() == 0.0) | (data_2017['LATITUDE'].round() == 0.0)].index

data_2017 = data_2017.drop(list(missing_location_indices))
data_2017 = data_2017.dropna()

df1 = data_2017['LOCATION'].groupby(data_2017['ZIP CODE'])
s1 = pd.Series(data = df1.count(), index = df1.count().index)
s2 = s1[s1 >= 1000]   # Zip codes where collisions are more than thousand

zip_codes_to_consider = set(s2.index)

# Accessing each group by zip code
# df1.get_group('10000')

Zip_CollisionLoc_dict = {}

for zip in zip_codes_to_consider:

    if zip not in Zip_CollisionLoc_dict:
        Zip_CollisionLoc_dict[zip] = [[],[]]  # Lat and Long lists

    locations = df1.get_group(zip)

    for loc in locations:
        lat, long = float(loc[1:-1].split(',')[0]), float(loc[1:-1].split(',')[1])
        Zip_CollisionLoc_dict[zip][0].append(lat)
        Zip_CollisionLoc_dict[zip][1].append(long)

Zip_CollisionArea_df = pd.DataFrame(index=zip_codes_to_consider, columns=['Mean Lat', 'Mean Long', 'std Lat', 'std Long',
                                                                          'Area', 'Collision Count', 'Collision per sq km'])

for zip in zip_codes_to_consider:
    Zip_CollisionArea_df.loc[zip, 'Mean Lat'] = np.mean(Zip_CollisionLoc_dict[zip][0])
    Zip_CollisionArea_df.loc[zip, 'Mean Long'] = np.mean(Zip_CollisionLoc_dict[zip][1])
    Zip_CollisionArea_df.loc[zip, 'std Lat'] = np.std(Zip_CollisionLoc_dict[zip][0])
    Zip_CollisionArea_df.loc[zip, 'std Long'] = np.std(Zip_CollisionLoc_dict[zip][1])

    center = (Zip_CollisionArea_df.loc[zip, 'Mean Lat'], Zip_CollisionArea_df.loc[zip, 'Mean Long'])
    a = distance.distance(center, (center[0] + Zip_CollisionArea_df.loc[zip, 'std Lat'], center[1])).km
    b = distance.distance(center, (center[0], center[1] + Zip_CollisionArea_df.loc[zip, 'std Long'])).km

    Zip_CollisionArea_df.loc[zip, 'Area'] = np.pi * a * b
    Zip_CollisionArea_df.loc[zip, 'Collision Count'] = s2[zip]

    Zip_CollisionArea_df.loc[zip, 'Collision per sq km'] = s2[zip]/Zip_CollisionArea_df.loc[zip, 'Area']

print('1(h)')
print(Zip_CollisionArea_df['Collision per sq km'].max())
