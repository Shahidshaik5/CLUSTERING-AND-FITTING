import pandas as pd
import numpy as np
from skimpy import clean_columns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


#read data and  set the column names to lowercase
df = pd.read_csv('co2emission.csv', encoding="ISO-8859-1")
df = clean_columns(df)

df2 = pd.read_csv('population_by_country_2020.csv')
df2 = clean_columns(df2)


def prep():
    df['percentage_of_world'] = df['percentage_of_world'].str.replace("%", "", regex=True)
    df['percentage_of_world'] = df['percentage_of_world'].astype(float)
    
    df['density_km_2'] = df['density_km_2'].str.replace("/kmÂ²", "", regex=True)
    df['density_km_2'] = df['density_km_2'].str.replace(',', '.', regex=True)
    df['density_km_2'] = df['density_km_2'].astype(float)

    df2['yearly_change'] = df2['yearly_change'].str.replace(" %", "", regex=True)
    df2['yearly_change'] = df2['yearly_change'].astype(float)
    df2['urban_pop_percentage'] = df2['urban_pop_percentage'].str.replace(" %", "", regex=True)
    df2['urban_pop_percentage'] = df2['urban_pop_percentage'].astype(float)
    df2.dropna()

    print(df.head())


def world_emission():
    yearly_world_emission_df = df.groupby("year").sum()
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    plt.plot(yearly_world_emission_df.index, yearly_world_emission_df["co_2_emission_tons"])
    plt.title("World yearly CO2 emission")
    plt.show()

def top4densily_populated_countries():
    co2_2020 = df[df['year'] >= 2010]
    co2_2020 = co2_2020.sort_values(by=['population_2022'], ascending=False)
    df2 = co2_2020[(co2_2020['country'] == 'China') | (co2_2020['country'] == 'India') | (co2_2020['country'] == 'United States') | (co2_2020['country'] == 'Indonesia')]
    plt.figure(figsize=[10, 6])
    sns.lineplot(x='year', y='co_2_emission_tons',hue='country' ,style='country', data=df2)
    plt.title('Top Densily Populated Countries\' co2 Emission' )
    plt.show()


def median_age_vs_urban_population():
    X = np.array(df2['median_age']).reshape(-1, 1)
    y = np.array(df2['urban_pop_percentage'])
    reg = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
    reg.fit(X_train, y_train)
    # print(reg.score(X_test, y_test))
    y_pred = reg.predict(X_test)

    plt.figure(figsize=[20, 8])
    sns.scatterplot(x='median_age', y='urban_pop_percentage',data=df2)
    plt.plot(X_test, y_pred, color ='r')
    plt.title('Median Vs Urban population Across the Globe')
    plt.show()


prep()
world_emission()
top4densily_populated_countries()
median_age_vs_urban_population()