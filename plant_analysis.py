import sqlalchemy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import sys
import os
from IPython.display import display
from scipy.signal import argrelextrema
from scipy.signal import lfilter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def select_plant(plant_id, plant_table, readings_table):
    """Returns the plant name and sensor readings for the particular plant based on the plant id number provided and DataFrames"""
    plant_name = plant_table[plant_table.index == plant_id].name_common.item()
    plant_readings = readings_table[readings_table.plant_id == plant_id]
    return plant_name, plant_readings


def plot_time_series(x, y, title, ylabel, figure='None'):
    """Creates time series plots given x and y series data, title and data label for y axis"""

    if figure == 'None':
        fig = plt.figure()
    else:
        fig = figure  # FIXME - plot on the same figure
    plt.plot(x, y, label=ylabel)
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.title(title)
    return fig


def plot_day(x, y, row, col, axis, title, ylabel):
    """Creates time series plots given x and y series data, title and data label for y axis"""
    axis[row, col].bar(x, y, color='orange')
    axis[row, col].set_xlabel('Hour in a Day')
    axis[row, col].set_ylabel(ylabel)
    axis[row, col].set_title(title)
    axis[row, col].set_xticks(np.arange(0, 25, 3))
    axis[row, col].set_ylim(0, 1000)
    return axis


# Connect to local SQL database
engine = sqlalchemy.create_engine(
    "mysql+pymysql://root:plant-outside-123-World@localhost/plant_data")
connection = engine.connect()

# Load SQL data into pandas DataFrames
plant_table = pd.read_sql('plants', connection, index_col='id', columns=[
                          'name_common', 'name_latin', 'soil_moist_min', 'soil_moist_max', 'light_min', 'light_max'])
readings_table = pd.read_sql('readings', connection, columns=[
                             'plant_id', 'datetime', 'light', 'soil_moist', 'soil_fert', 'temp'])

# Create new column "month" for monthly analysis later on
readings_table['month'] = readings_table['datetime'].dt.month

# Apply smoothing function to soil_moist data
n = 15  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1
readings_table.soil_moist = lfilter(b, a, readings_table.soil_moist)

# Ask for user input about graphs
entry = 'null'
num = 'null'
while entry not in ['a', 'l', 'm', 'g', 'w', 's']:
    entry = input(
        'Graphs: Enter \n(a) for all light data for each plant\n(l) for average light data each month\n(m) for soil moisture and watering\n(g) for monthly global solar radiation for NYC\n(s) to skip\n: ')

    if entry == 's':
        break

    elif entry == 'a':  # Graph of light data in its entirety for each plant
        for plant_id in readings_table.plant_id.unique():
            plant_name, plant_readings = select_plant(
                plant_id, plant_table, readings_table)
            x = plant_readings.datetime
            y = plant_readings.light
            title = f'Full Light Data for {plant_name} Plant'
            plot_time_series(x, y,
                             title, ylabel='Light (mmol)')

    # Graph of light data over course of an average day each month for each plant
    elif entry == 'l':
        while num != 'q':
            num = input(
                'Enter plant number (1-4) to graph avg light for each month or (q) to quit and show plots: ')
            try:
                plant_name, plant_readings = select_plant(
                    int(num), plant_table, readings_table)
            except ValueError:
                break
            fig, axs = plt.subplots(2, 6)
            fig.suptitle(
                f'Light Data for Average Day in Each Month for {plant_name} Plant')
            for month in np.arange(1, 13):
                if month in plant_readings.month.values:
                    # select month of data
                    plant_data_month = plant_readings.loc[plant_readings.month == month]
                    # plant_data_month = plant_data_month.reset_index()
                    readings_avg_day = plant_data_month.groupby(
                        plant_data_month['datetime'].dt.hour).mean()  # group data by hour in day and average for each hour
                    x = readings_avg_day.index
                    y = readings_avg_day.light
                    axs = plot_day(x, y,
                                   title=f'{calendar.month_abbr[month]}', ylabel='Light (mmol)', axis=axs, row=(month - 1) // 6, col=(month - 1) % 6)
            # Hide x labels and tick labels for top plots and y ticks for right plots.
            for ax in axs.flat:
                ax.label_outer()

    elif entry == 'g':  # Incorporate global solar radiation for NYC to normalize data based on month

        # Import data from National Solar Radiation Database API
        # https://nsrdb.nrel.gov/data-sets/api-instructions.html
        lat, lon = 40.6872854, -73.9757991
        api_key = os.environ['nsrdb_api_key']
        attributes = 'ghi'
        year = '2019'
        leap_year = 'false'
        # Set time interval in minutes, i.e., '30' is half hour intervals. Valid intervals are 30 & 60.
        interval = '60'
        # Specify Coordinated Universal Time (UTC), 'true' will use UTC, 'false' will use the local time zone of the data.
        # local time zone.
        utc = 'false'
        your_name = 'Jeff+Schwane'
        reason_for_use = 'personal+project'
        your_affiliation = 'N/A'
        your_email = 'jschwane@gmail.com'
        mailing_list = 'false'

        # Declare url string
        url = f'https://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap_year}&interval={interval}&utc={utc}&full_name={your_name}&email={your_email}&affiliation={your_affiliation}&mailing_list={mailing_list}&reason={reason_for_use}&api_key={api_key}&attributes={attributes}'

        # Return just the first 2 lines to get metadata:
        info = pd.read_csv(url, nrows=1)

        # Return all but first 2 lines of csv to get data:
        df = pd.read_csv(
            f'https://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap_year}&interval={interval}&utc={utc}&full_name={your_name}&email={your_email}&affiliation={your_affiliation}&mailing_list={mailing_list}&reason={reason_for_use}&api_key={api_key}&attributes={attributes}', skiprows=2)

        # Set the time index in the pandas dataframe:
        df = df.set_index(pd.date_range(
            f'1/1/{year}', freq=interval+'Min', periods=525600/int(interval)))

        # plot GHI over average month and sum GHI per month
        ghi_month_sum = {}
        fig, axs = plt.subplots(2, 6)
        fig.suptitle(
            'Global Horizontal Irradiance for Average Day in Each Month in NYC')
        for month in np.arange(1, 13):
            month_data = df.loc[df.Month == month]
            readings_avg_day = month_data.groupby(
                month_data.Hour).mean()  # group data by hour in day and average for each hour
            x = readings_avg_day.index
            y = readings_avg_day.GHI
            axs = plot_day(x, y,
                           title=f'{calendar.month_abbr[month]}', ylabel='Light ($W/m^2$)', axis=axs, row=(month-1)//6, col=(month-1) % 6)
            # Sum GHI for each month
            ghi_month_sum[month] = month_data.GHI.sum()
        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()

        # Comparision of light levels before/after move on 12/1/20
        while num != 'q':
            num = input(
                'Enter plant number (1-4) to graph soil moisture or (q) to quit and show plots: ')
            try:
                plant_name, plant_readings = select_plant(
                    int(num), plant_table, readings_table)
            except ValueError:
                break
            plant_month_sum = {}
            for month in np.arange(1, 13):
                if month in plant_readings.month.values:
                    # Grab only data for the month
                    plant_data_month = plant_readings.loc[plant_readings.month == month]
                    plant_month_sum[month] = plant_data_month.light.sum()
            df_ghi_month = pd.DataFrame.from_dict(
                ghi_month_sum, orient='index')
            df_plant_month = pd.DataFrame.from_dict(
                plant_month_sum, orient='index')
            # Normalize light levels based on GHI for the month
            df_plant_month_norm = df_plant_month / df_ghi_month

            # Percentage difference before/after 12/1/20
            before = df_plant_month_norm[7:12].mean()
            after = (df_plant_month_norm.loc[12] +
                     df_plant_month_norm.loc[1]) / 2
            pct_diff = round(((after-before)/before*100)[0])

            # Graph of normalized light levels and comparison between before/after 12/1/20
            x = df_plant_month_norm.index
            y = df_plant_month_norm[0]
            fig = plt.figure()
            plt.bar(x, y, color='orange')
            plt.xlabel('Month')
            plt.ylabel('Normalized Light Levels')
            plt.title(
                f'Comparison of Light Levels Before & After Move on 12/1/20 for {plant_name}')
            plt.ylim(0, .5)
            plt.text(
                4, .45, f'Difference before/after 12/1/20: {pct_diff}%')

            # Train supervised ML model based on target variable

    elif entry == 'm':  # Graph of soil moisture data in its entirety for each plant
        # Find local peaks
        ilocs_min = argrelextrema(
            readings_table.soil_moist.values, np.less_equal, order=125)[0]  # Searches range of 125 hours (5+ days) on both sides for minimum
        ilocs_max = argrelextrema(
            readings_table.soil_moist.values, np.greater_equal, order=125)[0]  # Searches range of 125 hours (5+ days) on both sides for maximum

        # Add soil moisture local min and max to table
        readings_table['local_max_moist'] = False
        readings_table['local_min_moist'] = False
        # Assign values to true where they are equal to local maxes
        readings_table.loc[readings_table.iloc[ilocs_max].index,
                           'local_max_moist'] = True
        # Assign values to true where they are equal to local mins
        readings_table.loc[readings_table.iloc[ilocs_min].index,
                           'local_min_moist'] = True

        # TODO - Learning algorithm that determines based on current soil moist, sunlight plant has been receiving and avg. temperature when it is next expected to need water

        while num != 'q':
            num = input(
                'Enter plant number (1-4) to graph soil moisture or (q) to quit and show plots: ')

            try:
                plant_name, plant_readings = select_plant(
                    int(num), plant_table, readings_table)

            except ValueError:
                break

            pd.set_option('mode.chained_assignment', None)
            # Clean mins of soil moisture readings by removing all but the first max and min when there are duplicate maxs and mins
            mask = plant_readings.local_min_moist

            # FIXME - Marks duplicate based on ever seeing it again, but this actually needs to be done locally in case there are two actual mins of the same value in the future
            duplicates = plant_readings.loc[mask].duplicated(
                subset=['soil_moist'], keep='first')  # Store duplicates in boolean series
            mask2 = mask & duplicates
            # Add column to dataframe
            plant_readings.loc[:,
                               'local_min_moist_dropped'] = plant_readings.loc[mask2, 'local_min_moist']
            # Fill in 'None' readings with False
            plant_readings.fillna(
                {'local_min_moist_dropped': False}, inplace=True)
            plant_readings.loc[plant_readings['local_min_moist_dropped'],
                               'local_min_moist'] = False  # Set duplicate minimums to False
            plant_readings.drop('local_min_moist_dropped',
                                axis=1, inplace=True)  # drop uneeded column

            # Determine soil moisture value when each plant is watered - use the mean of the soil moist reading everytime it is detected that that plant was watered (mean of soil_moist_min)
            # watering_value = plant_readings.loc[plant_readings.local_min_moist, 'soil_moist'].mean(
            # )
            # watering_value += 2  # Adds 2% to watering value in order to capture more watering times and increase size of target variable set

            # Use values stored in SQL table
            watering_value = plant_table.loc[int(num), 'soil_moist_min']

            # Create "days until next watering" target variable by backfilling

            # Dates plant should be watered is anytime soil moisture value is less than or equal to watering value
            watering_dates = plant_readings.loc[plant_readings.soil_moist <=
                                                watering_value, 'datetime']

            plant_readings.loc[:, 'days_until_watering'] = None
            plant_readings.loc[:, 'days_between_waterings'] = None
            # Set "days until watering" to zero for index positions in watering_dates
            plant_readings.loc[watering_dates.index, 'days_until_watering'] = 0

            # Backfill from zeros
            # Set first 'days until watering' value to zero
            plant_readings.at[plant_readings.first_valid_index(),
                              'days_until_watering'] = 0

            # Update values in 'days_until_watering'
            counter = 'zero'
            for index_label, row_series in plant_readings.iterrows():
                if row_series[9] != 0:  # 9th position is 'days_until_watering'
                    counter = 'nonzero'
                if row_series[9] == 0 and counter == 'nonzero':
                    counter = 'zero'
                    backsteps = 1
                    # Start backfilling either until the next zero is reached, or once it has hit a max. Otherwise model will not learn correctly the plant may have been watered before the soil moisture value hit the watering_value
                    i = index_label - backsteps
                    while plant_readings.loc[i, 'days_until_watering'] != 0 and plant_readings.loc[i, 'local_max_moist'] == False:
                        plant_readings.at[i,
                                          'days_until_watering'] = backsteps / 24
                        if plant_readings.loc[i - 1, 'local_max_moist'] == True:
                            # caluclate days between watering for each watering
                            plant_readings.at[i, 'days_between_waterings'] = plant_readings.loc[i,
                                                                                                'days_until_watering']
                        backsteps += 1
                        i = index_label - backsteps

            # Plot soil moisture and days until watering
            x = plant_readings.datetime
            y = plant_readings.soil_moist
            title = f'Soil Moisture Data for {plant_name} Plant'
            fig_1 = plot_time_series(
                x, y, title, 'Soil Moisture (%)')

            y = plant_readings.days_until_watering
            plt.plot(x, y, figure=fig_1, label='Days until watering')

            y = plant_readings[plant_readings['local_max_moist']].soil_moist
            max_idx = y.index
            x = plant_readings.datetime[max_idx]
            plt.scatter(x, y, linewidths=1, c='red',
                        marker="v", figure=fig_1)

            y = plant_readings[plant_readings['local_min_moist']].soil_moist
            min_idx = y.index
            x = plant_readings.datetime[min_idx]
            plt.scatter(x, y, linewidths=1, c='green',
                        marker="^", figure=fig_1)

            xmin = plant_readings.datetime.iloc[0]
            xmax = plant_readings.datetime.iloc[-1]
            plt.hlines(
                watering_value, xmin, xmax, label='Average watering threshold')
            plt.text(
                xmax, watering_value, f'{round(watering_value)}%')

            plt.legend()

            # # Plot each sensor vs. days until watering to notice trends

            # # Plot light vs. days until watering
            # fig_2 = plt.figure()
            # x = plant_readings.days_until_watering
            # y = plant_readings.light
            # plt.plot(x, y)
            # plt.xlabel('Days Until Watering')
            # plt.ylabel('Light (mmol)')
            # plt.title(f'Light vs. Days Until Watering for {plant_name} Plant')

            # # Plot soil moisture vs. days until watering
            # fig_3 = plt.figure()
            # x = plant_readings.days_until_watering
            # y = plant_readings.soil_moist
            # plt.plot(x, y)
            # plt.xlabel('Days Until Watering')
            # plt.ylabel('Soil Moisture (%)')
            # plt.title(
            #     f'Soil Moisture vs. Days Until Watering for {plant_name} Plant')

            # # Plot soil temperature vs. days until watering
            # fig_4 = plt.figure()
            # x = plant_readings.days_until_watering
            # y = plant_readings.temp
            # plt.plot(x, y)
            # plt.xlabel('Days Until Watering')
            # plt.ylabel('Temperature (℃)')
            # plt.title(
            #     f'Temperature vs. Days Until Watering for {plant_name} Plant')

            # # Plot soil fertility vs. days until watering
            # fig_5 = plt.figure()
            # x = plant_readings.days_until_watering
            # y = plant_readings.soil_fert
            # plt.plot(x, y)
            # plt.xlabel('Days Until Watering')
            # plt.ylabel('Soil Fertility (μs/cm)')
            # plt.title(
            #     f'Soil Fertility vs. Days Until Watering for {plant_name} Plant')

            # # Create column for rolling sum of light to analyze if that affects days until watering
            # plant_readings['light_roll'] = plant_readings.light.rolling(
            #     72).sum()
            # fig_6 = plt.figure()
            # x = plant_readings.days_until_watering
            # y = plant_readings.light_roll
            # plt.plot(x, y)
            # plt.xlabel('Days Until Watering')
            # plt.ylabel('Light Received Over Past 3 Days')
            # plt.title(
            #     f'Past Light Received vs. Days Until Watering for {plant_name} Plant')

            # Create Dummy Model which uses the average time between waterings to predict days until watering
            plant_readings.loc[:, 'dummy_days_until_watering'] = None
            avg_days_between_watering = plant_readings.days_between_waterings.mean()

            counter = 'false'
            for index_label, row_series in plant_readings.iterrows():
                if row_series[7] == True:  # 7th position true indicates plant was watered
                    counter = 'true'
                    plant_readings.at[index_label,
                                      'dummy_days_until_watering'] = 0

                if row_series[7] == False and counter == 'true':
                    counter = 'false'
                    forwardsteps = 0
                    i = index_label + forwardsteps
                    plant_readings.at[i-1,
                                      'dummy_days_until_watering'] = avg_days_between_watering
                    try:
                        while plant_readings.loc[i, 'local_max_moist'] == False:
                            if plant_readings.loc[i-1, 'dummy_days_until_watering'] <= 0:
                                plant_readings.at[i-1,
                                                  'dummy_days_until_watering'] = 0
                                plant_readings.at[i,
                                                  'dummy_days_until_watering'] = 0
                            elif plant_readings.loc[i-1, 'dummy_days_until_watering'] == 0:
                                plant_readings.at[i,
                                                  'dummy_days_until_watering'] = 0
                            else:
                                plant_readings.at[i, 'dummy_days_until_watering'] = avg_days_between_watering - (
                                    forwardsteps/24)  # Fill forward with predictions
                            forwardsteps += 1
                            i = index_label + forwardsteps
                    except KeyError:
                        break

            # Fit linear Regression Model

            # Drop rows where days until watering coudn't be calculated due to soil mosisture never reaching watering threshold before getting watered
            df = plant_readings.dropna(subset=['days_until_watering'])
            X = df.loc[:, ['light', 'soil_moist', 'temp', 'soil_fert']]
            y = df.loc[:, 'days_until_watering']
            y_dumb = df.loc[:, 'dummy_days_until_watering']
            # Arrow of time - everything in test set must occur after training - no shuffling!
            X_train, X_test, y_train, y_test, y_dumb_train, y_dumb_test = train_test_split(
                X, y, y_dumb, test_size=0.3, shuffle=False)

            # Scale the training data with fit transform, and the testing data with transform only, so that each parameter counts equally toward learning
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            reg = LinearRegression()
            reg.fit(X_train_scaled, y_train)
            y_pred = reg.predict(X_test_scaled)

            reg_score = mean_squared_error(y_test, y_pred)
            dummy_score = mean_squared_error(y_test, y_dumb_test)

            print(f"The MSE for our regressor is:{reg_score}")
            print(f"The MSE for our dummy is:{dummy_score}")

            # TODO - Try adding in rolling sum of light to see if that improves model

            # For purposes of the algorithm, maybe look at avgs/totals for each day?

            # Need to scale data first so that all parameters are weighted equally - fit on the trained data, and transform on both train and test - USE PIPELINE

            # Resource - https://towardsdatascience.com/predictive-maintenance-of-turbofan-engines-ec54a083127


# TODO - Need to look back some length of time to see how much sunlight it has been getting in order to predict how much sun it will get in next few days

# TODO - Connect to weather prediction API (sunny/cloudy) for light predictions? Correlate to light detected in training data


# TODO - For each plant type, compute the avg number of days between watering. Does this vary by season?

# TODO - Is each plant getting the light it needs? Compare to min/max levels
# https://www.houseplantjournal.com/bright-indirect-light-requirements-by-plant/
# Calculate % of daylight hours that plant is getting light above thresholds

# Can use matplotlib and seaborn for histogram graphs:
#   plt.figure(figsize=(10,6))
#   sns.distplot(series data, bins=10)
plt.show()
