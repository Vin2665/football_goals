

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup as bs
import time
import sklearn
import seaborn as sns

#Collecting  the match scores and venue details of WCQ — AFC (M) from last 4 seasons"""

years = list(range(2022,2005,-4))
years

# creating an empty dataset
combined_years = []
url = "https://fbref.com/en/comps/7/schedule/WCQ----AFC-M-Scores-and-Fixtures"

# for loop to get the data from the specific years that we need and combining the data
for year in years:
    data = requests.get(url)
    soup = bs(data.text)
    previous_season = soup.select("a.prev")[0].get("href")
    url = f"https://fbref.com{previous_season}"
    matches = pd.read_html(data.text, match = "Scores & Fixtures")[0]
    combined_years.append(matches)
    time.sleep(1)

Asian_qual = pd.concat(combined_years)

#Collecting  the match scores and venue details of
#AFC Asian Cup qualification from last 4 seasons


years = list(range(2019,2006,-4))
years

# creating an empty dataset
combined_years = []
url = "https://fbref.com/en/comps/665/2019/schedule/2019-AFC-Asian-Cup-qualification-Scores-and-Fixtures"

# for loop to get the data from the specific years that we need and combining the data
for year in years:
    data = requests.get(url)
    soup = bs(data.text)
    previous_season = soup.select("a.prev")[0].get("href")
    url = f"https://fbref.com{previous_season}"
    matches = pd.read_html(data.text, match = "Scores & Fixtures")[0]
    combined_years.append(matches)
    time.sleep(1)

Asiacup = pd.concat(combined_years)

#Collecting  the match scores and venue details of
#Friendlies(M) from last 5 years


years = list(range(2021,2014,-1))
years

# creating an empty dataset
combined_years = []
url = "https://fbref.com/en/comps/218/2021/schedule/2021-Friendlies-M-Scores-and-Fixtures"

# for loop to get the data from the specific years that we need and combining the data
for year in years:
    data = requests.get(url)
    soup = bs(data.text)
    previous_season = soup.select("a.prev")[0].get("href")
    url = f"https://fbref.com{previous_season}"
    matches = pd.read_html(data.text, match = "Scores & Fixtures")[0]
    combined_years.append(matches)
    time.sleep(1)

friendly_comp = pd.concat(combined_years)

#Collecting  the match scores and venue details of
#Saudi Professional League from last 5 years


years = list(range(2021,2014,-1))
years

# creating an empty dataset
combined_years = []
url = "https://fbref.com/en/comps/70/2021-2022/schedule/2021-2022-Saudi-Professional-League-Scores-and-Fixtures"

# for loop to get the data from the specific years that we need and combining the data
for year in years:
    data = requests.get(url)
    soup = bs(data.text)
    previous_season = soup.select("a.prev")[0].get("href")
    url = f"https://fbref.com{previous_season}"
    matches = pd.read_html(data.text, match = "Scores & Fixtures")[0]
    combined_years.append(matches)
    time.sleep(1)

Saudi_professional_league = pd.concat(combined_years)

#Collecting  the match scores and venue details of
#Persian Gulf Pro League from last 5 years


years = list(range(2021,2014,-1))
years

# creating an empty dataset
combined_years = []
url = "https://fbref.com/en/comps/64/2021-2022/schedule/2021-2022-Persian-Gulf-Pro-League-Scores-and-Fixtures"

# for loop to get the data from the specific years that we need and combining the data
for year in years:
    data = requests.get(url)
    soup = bs(data.text)
    previous_season = soup.select("a.prev")[0].get("href")
    url = f"https://fbref.com{previous_season}"
    matches = pd.read_html(data.text, match = "Scores & Fixtures")[0]
    combined_years.append(matches)
    time.sleep(1)

# Combining all the stats in one dataframe
Persian_Gulf__Pro_league = pd.concat(combined_years)

#After looking at all the tables we find a few columns that are not common.
#Let us remove these columns from the tables


Asiacup = Asiacup.drop(['Round','Wk'], axis=1)
Asian_qual = Asian_qual.drop(['Round','Wk'], axis=1)
Persian_Gulf__Pro_league = Persian_Gulf__Pro_league.drop(['Wk'], axis=1)
Saudi_professional_league = Saudi_professional_league.drop(['Wk'], axis=1)

#Let us combine all the datarframes"""

all_frames = [Asian_qual, Asiacup, friendly_comp, Persian_Gulf__Pro_league, Saudi_professional_league]
scores1 = pd.concat(all_frames, ignore_index=True)

#now let us remove the rows in which all the values are null"""

scores = scores1.dropna(axis=0, how='all', thresh=None, subset=None, inplace=False)

#Let us save this as a csv file"""

scores.to_csv('football_scores.csv')

#Reading the CSV file using pandas"""

scores = pd.read_csv('football_scores.csv')

#Let us drop the columns that are not necessary for us"""

scores = scores.drop(['Referee','Match Report','Notes'], axis=1)

#Let us check for missing values or null values in our dataset"""

scores.isna().sum()

#let us drop the rows having missing values from Venue, Time and Score and the duplicates"""

scores = scores.dropna(axis=0, subset=['Venue','Time','Score'])
scores.isna().sum()
scores = scores.drop_duplicates()
scores.info

#Let us drop the table index that got copied during combining the dataframes"""

scores = scores.drop(index = 24)
scores = scores.reset_index(drop=True)
scores.isna().sum()

#scores.to_csv(r'D:\DS&AI\Project 74\scores.csv', index=False)

#as the number of missing values in the attendance column is huge, it is better we replace the missing values with the mean value using MeanImputer method


from sklearn.impute import SimpleImputer
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
scores["Attendance"] = pd.DataFrame(mean_imputer.fit_transform(scores[["Attendance"]]))
scores.isna().sum()

#there are no null values

#Let us drop the matches that have penalty scores


scores = scores.drop([scores.index[601],scores.index[630],scores.index[1701],scores.index[1703],scores.index[1717],scores.index[1746],scores.index[1874],scores.index[2238],scores.index[2499],scores.index[2576],scores.index[2916]])
scores = scores.reset_index(drop=True)

#let us combine the score column as sum of goals to show the total goals scored in the match"""

scores[['Team Points','Opp Points']] = scores['Score'].str.split('–',expand=True)
scores[['Team Points','Opp Points']] = scores[['Team Points','Opp Points']].astype(int)

sum = scores['Team Points'] + scores['Opp Points']
scores['total_goals'] = scores['Team Points'] + scores['Opp Points']

#Let us drop the Scores, Team Points and Opp Points column as these are not necessary"""

scores = scores.drop(['Team Points','Opp Points','Score'], axis=1)

import geopy
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent='Getloc',timeout=None)

#since this is a large dataset Let us split this dataframe into 3

s1 = scores.iloc[:2500,:]
s2 = scores.iloc[2500:4500,:]
s3 = scores.iloc[4500:,:]

s1['latitude'] = s1['Venue'].apply(lambda x: geolocator.geocode(x).latitude if geolocator.geocode(x) != None else "nan")
s1['longitude'] = s1['Venue'].apply(lambda x: geolocator.geocode(x).longitude if geolocator.geocode(x) != None else "nan")
s1 = s1[s1.latitude != "nan"]

s2['latitude'] = s2['Venue'].apply(lambda x: geolocator.geocode(x).latitude if geolocator.geocode(x) != None else "nan")
s2['longitude'] = s2['Venue'].apply(lambda x: geolocator.geocode(x).longitude if geolocator.geocode(x) != None else "nan")
s2 = s2[s2.latitude != "nan"]

s3['latitude'] = s3['Venue'].apply(lambda x: geolocator.geocode(x).latitude if geolocator.geocode(x) != None else "nan")
s3['longitude'] = s3['Venue'].apply(lambda x: geolocator.geocode(x).longitude if geolocator.geocode(x) != None else "nan")
s3 = s3[s3.latitude != "nan"]

all_frames1 = [s1, s2, s3]
newdf = pd.concat(all_frames1, ignore_index=True)

#newdf.to_csv(r'D:\DS&AI\Project 74\Final74\f1.csv', index=False)

newdf.columns

#We will not need the day column

newdf = newdf.drop(['Day'], axis=1)
newdf = newdf.reset_index(drop=True)
newdf.dtypes

#Getting the weather conditions based on the latitude and longitude based on the time and date

import math
import datetime
import urllib.request
import json

records=[]
labels = ['Date','Time','Home','Away','Attendance','Venue','total_goals','latitude','longitude','hourly_time','Temperature','Relative_humidity','Dewpoint','SurfacePressure','Precipitation','Rain','Wind_speed','Wind_direction']

for i, row in newdf.iterrows():

    date=row['Date']
    time=row['Time']    
    home=row['Home']
    away=row['Away']
    attendance=row['Attendance']
    venue=row['Venue']
    totalgoals=row['total_goals']
    latitude=row['latitude']
    longitude=row['longitude']


    latitude='{:.4f}'.format(latitude)
    longitude='{:.4f}'.format(longitude)

    
    query_params = 'latitude={}&longitude={}&start_date={}&end_date={}&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,surface_pressure,precipitation,rain,windspeed_10m,winddirection_10m'
    query_params=query_params.format(latitude,longitude,date,date)
    
    try:
        response = urllib.request.urlopen('https://archive-api.open-meteo.com/v1/era5' +"?"+ query_params)
        data = response.read()
        
    except Exception:
        print("Error reading from {}".format('https://archive-api.open-meteo.com/v1/era5' +"?"+ query_params))
        continue
    weatherData = json.loads(data.decode('utf-8')) 
    for x in weatherData:
        values = weatherData['hourly']
    for x in values:
        records.append((date,time,home,away,attendance,venue,totalgoals,latitude,longitude,values["time"],values["temperature_2m"],values["relativehumidity_2m"],values["dewpoint_2m"],values["surface_pressure"],values["precipitation"],values["rain"],values["windspeed_10m"],values["winddirection_10m"]))

newdf = pd.DataFrame.from_records(records, columns=labels)


#newdf.to_csv(r'D:\DS&AI\Project 74\\Final74\f2.csv')
#newdf = pd.read_csv(r"D:\DS&AI\Project 74\Final74\f2.csv")

#Removing the duplictaes comparing data, latitude and longitude


newdf = newdf.drop_duplicates(subset = ['Date', 'latitude','longitude'],).reset_index(drop = True)

#Let us separate all the values in the climatic conditions columns

newdf['Temperature'] = newdf['Temperature'].str[1:-1].str.split(", ").tolist()
newdf['Relative_humidity'] = newdf['Relative_humidity'].str[1:-1].str.split(', ').tolist()
newdf['Dewpoint'] = newdf['Dewpoint'].str[1:-1].str.split(', ').tolist()
newdf['SurfacePressure'] = newdf['SurfacePressure'].str[1:-1].str.split(', ').tolist()
newdf['Precipitation'] = newdf['Precipitation'].str[1:-1].str.split(', ').tolist()
newdf['Rain'] = newdf['Rain'].str[1:-1].str.split(', ').tolist()
newdf['Wind_speed'] = newdf['Wind_speed'].str[1:-1].str.split(', ').tolist()
newdf['Wind_direction'] = newdf['Wind_direction'].str[1:-1].str.split(', ').tolist()

#Converting the Date and Time to DateTime stamp and then splitting the date and time


newdf['datestamp'] = pd.to_datetime(newdf.Date + ' ' + newdf.Time)

#Rounding off the time to nearest hour

from datetime import datetime, timedelta

now = datetime.now()

def hour_rounder(t):
    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
               +timedelta(hours=t.minute//30))

newdf['datestamp'] = newdf['datestamp'].apply(hour_rounder)

#Splitting the DateTime stamp into separate date and time columns

newdf['date'] = newdf['datestamp'].dt.date
newdf['time'] = newdf['datestamp'].dt.time

#Rearranging and considering the new processed columns of date and time

newdf = newdf.iloc[:, [19,20,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17]]

#Taking time as an integer in a new column to use in comparison

newdf['intTime'] = newdf['time'].astype(str).str[0:2].astype(int)

newdf.time.unique()

#We can see that the lowest time of the day at which match starts is 11:00
#let us group the data according to the time at which the game begins


dataf1 = newdf[newdf['intTime'] <= 11]
dataf2 = newdf[(newdf['intTime'] > 13) & (newdf['intTime'] <= 15)]
dataf3 = newdf[(newdf['intTime'] > 15) & (newdf['intTime'] <= 17)]
dataf4 = newdf[(newdf['intTime'] > 17) & (newdf['intTime'] <= 19)]
dataf5 = newdf[(newdf['intTime'] > 19)]

#For each grouped dataset, let us assign climate conditions 1 hr prior and 4 hr after the match timing

#let us assign our climate conditions of the respective hours for dataf1


dataf1['Temperature'] = dataf1['Temperature'].apply(lambda x: x[10:15])
dataf1['Relative_humidity'] = dataf1['Relative_humidity'].apply(lambda x: x[10:15])
dataf1['Dewpoint'] = dataf1['Dewpoint'].apply(lambda x: x[10:15])
dataf1['SurfacePressure'] = dataf1['SurfacePressure'].apply(lambda x: x[10:15])
dataf1['Precipitation'] = dataf1['Precipitation'].apply(lambda x: x[10:15])
dataf1['Rain'] = dataf1['Rain'].apply(lambda x: x[10:15])
dataf1['Wind_speed'] = dataf1['Wind_speed'].apply(lambda x: x[10:15])
dataf1['Wind_direction'] = dataf1['Wind_direction'].apply(lambda x: x[10:15])

#let us assign our climate conditions of the respective hours for dataf2

dataf2['Temperature'] = dataf2['Temperature'].apply(lambda x: x[13:19])
dataf2['Relative_humidity'] = dataf2['Relative_humidity'].apply(lambda x: x[13:19])
dataf2['Dewpoint'] = dataf2['Dewpoint'].apply(lambda x: x[13:19])
dataf2['SurfacePressure'] = dataf2['SurfacePressure'].apply(lambda x: x[13:19])
dataf2['Precipitation'] = dataf2['Precipitation'].apply(lambda x: x[13:19])
dataf2['Rain'] = dataf2['Rain'].apply(lambda x: x[13:19])
dataf2['Wind_speed'] = dataf2['Wind_speed'].apply(lambda x: x[13:19])
dataf2['Wind_direction'] = dataf2['Wind_direction'].apply(lambda x: x[13:19])

#let us assign our climate conditions of the respective hours for dataf3

dataf3['Temperature'] = dataf3['Temperature'].apply(lambda x: x[15:21])
dataf3['Relative_humidity'] = dataf3['Relative_humidity'].apply(lambda x: x[15:21])
dataf3['Dewpoint'] = dataf3['Dewpoint'].apply(lambda x: x[15:21])
dataf3['SurfacePressure'] = dataf3['SurfacePressure'].apply(lambda x: x[15:21])
dataf3['Precipitation'] = dataf3['Precipitation'].apply(lambda x: x[15:21])
dataf3['Rain'] = dataf3['Rain'].apply(lambda x: x[15:21])
dataf3['Wind_speed'] = dataf3['Wind_speed'].apply(lambda x: x[15:21])
dataf3['Wind_direction'] = dataf3['Wind_direction'].apply(lambda x: x[15:21])

#let us assign our climate conditions of the respective hours for dataf4

dataf4['Temperature'] = dataf4['Temperature'].apply(lambda x: x[17:23])
dataf4['Relative_humidity'] = dataf4['Relative_humidity'].apply(lambda x: x[17:23])
dataf4['Dewpoint'] = dataf4['Dewpoint'].apply(lambda x: x[17:23])
dataf4['SurfacePressure'] = dataf4['SurfacePressure'].apply(lambda x: x[17:23])
dataf4['Precipitation'] = dataf4['Precipitation'].apply(lambda x: x[17:23])
dataf4['Rain'] = dataf4['Rain'].apply(lambda x: x[17:23])
dataf4['Wind_speed'] = dataf4['Wind_speed'].apply(lambda x: x[17:23])
dataf4['Wind_direction'] = dataf4['Wind_direction'].apply(lambda x: x[17:23])

#let us assign our climate conditions of the respective hours for dataf5

dataf5['Temperature'] = dataf5['Temperature'].apply(lambda x: x[19:24])
dataf5['Relative_humidity'] = dataf5['Relative_humidity'].apply(lambda x: x[19:24])
dataf5['Dewpoint'] = dataf5['Dewpoint'].apply(lambda x: x[19:24])
dataf5['SurfacePressure'] = dataf5['SurfacePressure'].apply(lambda x: x[19:24])
dataf5['Precipitation'] = dataf5['Precipitation'].apply(lambda x: x[19:24])
dataf5['Rain'] = dataf5['Rain'].apply(lambda x: x[19:24])
dataf5['Wind_speed'] = dataf5['Wind_speed'].apply(lambda x: x[19:24])
dataf5['Wind_direction'] = dataf5['Wind_direction'].apply(lambda x: x[19:24])

#Let us merge all the divided dataframes

all_frames3 = [dataf1, dataf2, dataf3, dataf4, dataf5]
newdf = pd.concat(all_frames3, ignore_index=True)

#Let us drop the rows that we do not have any climatic data. Sorting the rows according the tempearute data"""

newdf = newdf.sort_values(by=['Temperature'])
newdf = newdf.reset_index(drop=True)

#We can see that the rows from 4296 have no tempearature data. So we can drop these rows as they do have any useful data. Considering only the rows that have useful data"""

newdf = newdf.iloc[:4297,:]
newdf = newdf.reset_index(drop=True)

#We still find that there are some values in Wind_direction that are None, since these are only a few rows, we can drop these rows"""

newdf = newdf.drop([newdf.index[1041],newdf.index[1089],newdf.index[1222],newdf.index[1644],newdf.index[2798]])
newdf = newdf.reset_index(drop=True)

#newdf.to_csv(r'D:\DS&AI\Project 74\Final74\f3.csv', index=False)

#Considering the hourly values by converting them to float for further calculation


newdf['Temperature'] = newdf['Temperature'].apply(lambda x: [float(a) for a in x])
newdf["Relative_humidity"] = newdf["Relative_humidity"].apply(lambda x: [float(a) for a in x])
newdf["Dewpoint"] = newdf["Dewpoint"].apply(lambda x: [float(a) for a in x])
newdf["SurfacePressure"] = newdf["SurfacePressure"].apply(lambda x: [float(a) for a in x])
newdf["Precipitation"] = newdf["Precipitation"].apply(lambda x: [float(a) for a in x])
newdf["Rain"] = newdf["Rain"].apply(lambda x: [float(a) for a in x])
newdf["Wind_speed"] = newdf["Wind_speed"].apply(lambda x: [float(a) for a in x])
newdf["Wind_direction"] = newdf["Wind_direction"].apply(lambda x: [float(a) for a in x])

#Considering the mean of each of the climatic conditions values

newdf['Temperature'] = newdf['Temperature'].apply(np.mean)
newdf['Relative_humidity'] = newdf['Relative_humidity'].apply(np.mean)
newdf['Dewpoint'] = newdf['Dewpoint'].apply(np.mean)
newdf['SurfacePressure'] = newdf['SurfacePressure'].apply(np.mean)
newdf['Precipitation'] = newdf['Precipitation'].apply(np.mean)
newdf['Rain'] = newdf['Rain'].apply(np.mean)
newdf['Wind_speed'] = newdf['Wind_speed'].apply(np.mean)
newdf['Wind_direction'] = newdf['Wind_direction'].apply(np.mean)

#newdf.to_csv(r'D:\DS&AI\Project 74\Final74\final_data.csv', index=False)

#newdf = pd.read_csv(r'D:\DS&AI\Project 74\Final74\final_data.csv')

newdf.columns

#There are some columns that we will not need in our dataset, Considering only the values that we require"""


#Considering the columns that we need

df = newdf.iloc[:, [4,9,10,11,12,14,15,16,6]]

df.columns

df.head()

df.tail()

df.shape

df.describe()

df.dtypes

df.Attendance = round(df['Attendance'])
df.Attendance = df.Attendance.astype('int64')
df.total_goals = df.total_goals.astype('int64')
df.Temperature = df.Temperature.astype('float')

#checking for any null values

df.isnull().sum()

#we do not have any null values in our dataset

#Let us check for duplicates in the dataset


duplicate = df.duplicated()
sum(duplicate)

#There are no duplicates in our data

#Lets look at the relationship between various variables


corelation = df.corr()

sns.heatmap(corelation, xticklabels=corelation.columns, yticklabels=corelation.columns,annot=True)

sns.pairplot(df)

sns.distplot(df['total_goals'])

sns.distplot(df['Temperature'])

sns.distplot(df['Relative_humidity'])

sns.distplot(df['Wind_speed'])

sns.distplot(df['Wind_direction'])



#We get BayesianRidge as the best model to use

#Using Bayesian Ridge regression algorithm


from sklearn import datasets, ensemble
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

predictors = df.loc[:, df.columns!="total_goals"]

target = df["total_goals"]

#Train Test partition of the data

x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)

reg = sklearn.linear_model.BayesianRidge()
reg.fit(x_train, y_train)

y_pred_gb = reg.predict(x_test)

mae = mean_absolute_error(y_test, y_pred_gb)
print("The mean absolute error (MAE) on test set: {:.4f}".format(mae))

mse = mean_squared_error(y_test, y_pred_gb)
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

mape = mean_squared_error(y_test, y_pred_gb)
print("The mean squared error (MAPE) on test set: {:.4f}".format(mape))


import pickle

pickle.dump(reg, open('BR_reg.pkl', 'wb'))
pickle.dump(df, open('data.pkl', 'wb'))




