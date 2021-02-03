import sqlalchemy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import calendar
from scipy.signal import argrelextrema


def select_plant(plant_id, plant_table, readings_table):
    """Returns the plant name and sensor readings for the particular plant based on the plant id number provided and DataFrames"""
    plant_name = plant_table[plant_table.index == plant_id].name_common.item()
    plant_readings = readings_table[readings_table.index == plant_id]
    return plant_name, plant_readings


def plot_time_series(x, y, title, ylabel, figure='None'):
    """Creates time series plots given x and y series data, title and data label for y axis"""

    if figure = 'None':
        fig = plt.figure()
    else:
        fig = figure  # FIXME - Determine how to plot on the same figure
    plt.plot(x, y)
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.title(title)
    return fig


def plot_day(x, y, title, ylabel):
    """Creates time series plots given x and y series data, title and data label for y axis"""
    fig = plt.figure()
    plt.bar(x, y, color='orange')
    plt.xlabel('Hour in a Day')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(np.arange(0, 25, 3))
    plt.ylim(0, 1000)
    return fig


# Connect to local SQL database
engine = sqlalchemy.create_engine(
    "mysql+pymysql://root:plant-outside-123-World@localhost/plant_data")
connection = engine.connect()

# Load SQL data into pandas DataFrames
plant_table = pd.read_sql('plants', connection, index_col='id', columns=[
                          'name_common', 'name_latin', 'soil_moist_min', 'soil_moist_max', 'light_min', 'light_max'])
readings_table = pd.read_sql('readings', connection, index_col='plant_id', columns=[
                             'datetime', 'light', 'soil_moist', 'soil_fert', 'temp'])

# Ask for user input about graphs
entry = 'null'
while entry not in ['1', '2', '3', '4', 's', 'a']:
    entry = input(
        'Enter (a) for all light data for each plant, plant id (#) for graphs of average light data each month or (s) to skip: ')
    if entry == 's':
        break
    if entry == 'a':  # Graph of light data in its entirety for each plant
        for plant_id in readings_table.index.unique():
            plant_name, plant_readings = select_plant(
                plant_id, plant_table, readings_table)
            x = plant_readings.datetime
            y = plant_readings.light
            title = f'Full Light Data for {plant_name} Plant'
            plot_time_series(x, y,
                             title, ylabel='Light (mmol)')
    # Graph of light data over course of an average day each month for each plant
    elif int(entry) in [1, 2, 3, 4]:
        # Create new column "month"
        readings_table['month'] = readings_table['datetime'].dt.month
        plant_name, plant_readings = select_plant(
            int(entry), plant_table, readings_table)
        for month in np.arange(1, 13):
            if month in plant_readings.month.values:
                # select month of data
                plant_data_month = plant_readings.loc[plant_readings.month == month]
                plant_data_month = plant_data_month.reset_index()
                readings_avg_day = plant_data_month.groupby(
                    plant_data_month['datetime'].dt.hour).mean()  # group data by hour in day and average for each hour
                x = readings_avg_day.index
                y = readings_avg_day.light
                plot_day(x, y,
                         title=f'Light Data for Average Day in {calendar.month_abbr[month]} for {plant_name} Plant ', ylabel='Light (mmol)')

plt.show()

# TODO - Graph of data soil moisture data showing what happens when the plant is watered

# Find local peaks
ilocs_min = argrelextrema(
    readings_table.soil_moist.values, np.less_equal, order=100)[0]  # Searches range of 100 hours (4 days) on both sides for minimum
ilocs_max = argrelextrema(
    readings_table.soil_moist.values, np.greater_equal, order=100)[0]  # Searches range of 100 hours (4 days) on both sides for maximum

# Add soil moisture local min and max to table
readings_table['local_max_moist'] = False
readings_table['local_min_moist'] = False
readings_table.loc[readings_table.iloc[ilocs_max].index,
                   'local_max_moist'] = True
readings_table.loc[readings_table.iloc[ilocs_min].index,
                   'local_min_moist'] = True

# Create graphs
plant_name, plant_readings = select_plant(1, plant_table, readings_table)
x = plant_readings.datetime[:750]
y = plant_readings.soil_moist[:750]
moist_plot = plot_time_series(
    x, y, title='Soil Moisture Data', ylabel='Soil Moisture (%)')

plot_time_series(
    x, y, title='Soil Moisture Data', ylabel='Soil Moisture (%)', figure=moist_plot)
plt.show()

# TODO - Incorporate global solar radiation for NYC to normalize data based on month

# TODO - Comparision of light levels before/after move on 12/1/20

# TODO - Learning algorithm that determines based on current soil moist, sunlight plant has been receiving and avg. temperature when it is next expected to need water
# Need to first determine when each plant is watered - use the mean or median of the soil moist reading everytime it is detected that that plant was watered
# Need to look back some length of time to see how much sunlight it has been getting in order to predict how much sun it will get in next few days
# Connect to weather prediction API (sunny/cloudy) for light predictions? Correlate to light detected in training data
# For purposes of the algorithm, maybe look at avgs/totals for each day?
# Need to scale data first so that all parameters are weighted equally - fit on the trained data, and transform on both train and test - USE PIPELINE
# Arrow of time - everything in test set must occur after training - no shuffling!

# TODO - For each plant type, compute the avg number of days between watering. Does this vary by season?

# TODO - Is each plant getting the light it needs? Compare to min/max levels
# https://www.houseplantjournal.com/bright-indirect-light-requirements-by-plant/
# Calculate % of daylight hours that plant is getting light above thresholds

# Can use matplotlib and seaborn for histogram graphs:
#   plt.figure(figsize=(10,6))
#   sns.distplot(series data, bins=10)
