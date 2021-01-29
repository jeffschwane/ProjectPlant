import sqlalchemy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import calendar


def plot_time_series(x, y, title, ylabel):
    """Creates time series plots given x and y series data, title and data label for y axis"""

    fig = plt.figure()
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
metadata = sqlalchemy.MetaData()
plant_table = sqlalchemy.Table(
    'plants', metadata, autoload=True, autoload_with=engine)
readings_table = sqlalchemy.Table(
    'readings', metadata, autoload=True, autoload_with=engine)

# Graph of light data in its entirety for each plant
# Load data light data from SQL
light_data = {}
for plant_id in [1, 2, 3, 4]:
    query = sqlalchemy.select([readings_table.columns.datetime, readings_table.columns.light]).where(
        readings_table.columns.plant_id == plant_id)
    result_proxy = connection.execute(query)
    result_set = result_proxy.fetchall()
    light_data[plant_id] = result_set
    light_data[plant_id] = [
        (x[0].strftime('%Y-%m-%dT%H:%M:%S'), x[1]) for x in light_data[plant_id]]  # Convert datetime objects to strings to put into numpy array
    light_data[plant_id] = np.array(
        light_data[plant_id], dtype=[('x', 'datetime64[us]'), ('y', 'i8')])  # Convert to numpy array
    x = light_data[plant_id]['x']
    y = light_data[plant_id]['y']
    # plot_time_series(x, y,
    #                  title = f'Full Light Data for Plant {plant_id}', ylabel = 'Light (mmol)')

# TODO - Graph of light data over course of an average day each month for each plant

for plant_id in [1]:
    plant_light_data = light_data[plant_id]
    plant_light_data = pd.DataFrame(
        plant_light_data)  # Store in pandas dataframe
    plant_light_data['month'] = plant_light_data['x'].dt.month
    for month in np.arange(1, 13):
        if month in plant_light_data.month.values:
            # select month of data
            plant_light_data_month = plant_light_data.loc[plant_light_data['month'] == month]
            light_avg_day = plant_light_data.groupby(
                plant_light_data_month['x'].dt.hour).mean()  # group data by hour in day and average for each hour
            x = light_avg_day.index
            y = light_avg_day['y']
            plot_day(x, y,
                     title=f'Light Data for Average Day in {calendar.month_abbr[month]} for Plant {plant_id}', ylabel='Light (mmol)')
plt.show()

# Can use matplotlib and seaborn for histogram graphs:
#   plt.figure(figsize=(10,6))
#   sns.distplot(series data, bins=10)

# TODO - Graph of data soil moisture data showing what happens when the plant is watered

# TODO - Incorporate global solar radiation for NYC to normalize data based on month

# TODO - Comparision of light levels before/after move on 12/1/20

# TODO - Learning algorithm that determines based on current soil moist, sunlight plant has been receiving and avg. temperature when it is next expected to need water
# Need to first determine when each plant is watered - use the mean or median of the soil moist reading everytime it is detected that that plant was watered
# Need to look back some length of time to see how much sunlight it has been getting in order to predict how much sun it will get in next few days
# For purposes of the algorithm, maybe look at avgs/totals for each day?
# Need to scale data first so that all parameters are weighted equally - fit on the trained data, and transform on both train and test - USE PIPELINE
# Arrow of time - everything in test set must occur after training - no shuffling!

# TODO - For each plant type, compute the avg number of days between watering. Does this vary by season?

# TODO - Could you use an API to the weather forecast and tie that to light depending on if it's cloudy or not?

# TODO - Is each plant getting the light it needs? Compare to min/max levels
# https://www.houseplantjournal.com/bright-indirect-light-requirements-by-plant/
# Calculate % of daylight hours that plant is getting light above thresholds
