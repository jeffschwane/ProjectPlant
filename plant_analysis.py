import sqlalchemy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import datetime as dt
import sys
import os
from IPython.display import display
from scipy.signal import argrelextrema
from scipy.signal import lfilter
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


def select_plant(plant_id, plant_table, readings_table):
    """Returns the plant name and sensor readings for the particular plant based on the plant id number provided and DataFrames"""
    plant_name = plant_table[plant_table.index == plant_id].name_common.item()
    plant_readings = readings_table[readings_table.plant_id == plant_id]
    return plant_name, plant_readings


def plot_time_series(x, y, title, ylabel, color, figure='None'):
    """Creates time series plots given x and y series data, title and data label for y axis"""

    if figure == 'None':
        fig = plt.figure()
    else:
        fig = figure  # plot on the same figure
    plt.plot(x, y, label=ylabel, c=color)
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


def return_score(y_test, y_pred, y_dumb_test, metric):
    """Returns the regression scores for the regression and dummy model for the metric type used: MSE, MAE, or r2"""
    if metric == 'MSE':
        reg_score = mean_squared_error(y_test, y_pred)
        dummy_score = mean_squared_error(y_test, y_dumb_test)
    elif metric == 'MAE':
        reg_score = mean_absolute_error(y_test, y_pred)
        dummy_score = mean_absolute_error(y_test, y_dumb_test)
    elif metric == 'r2':
        reg_score = r2_score(y_test, y_pred)
        dummy_score = r2_score(y_test, y_dumb_test)
    return reg_score, dummy_score


# Connect to local SQL database
sql_pass = os.environ['sql_password']
engine = sqlalchemy.create_engine(
    f"mysql+pymysql://root:{sql_pass}@localhost/plant_data")
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
sns.set_theme()  # Apply default seaborn theme
entry = 'null'
num = 'null'
while entry not in ['a', 'l', 'm', 'g', 'w', 's']:
    entry = input(
        'Graphs: Enter \n(a) for all light data for each plant\n(l) for average light data each month\n(m) for soil moisture and watering\n(g) for monthly global solar radiation for NYC and move comparison\n(s) to skip\n: ')

    if entry == 's':
        break

    elif entry == 'a':  # Graph of light data in its entirety for each plant
        while num not in ['m', 'r']:
            num = input(
                '\nPlot data showing:\n(m) for before/after move \n(r) light requirement \n: ')
        for plant_id in readings_table.plant_id.unique():
            plant_name, plant_readings = select_plant(
                plant_id, plant_table, readings_table)
            x = plant_readings.datetime
            y = plant_readings.light
            title = f'Full Light Data for {plant_name} Plant'
            plot_time_series(x, y,
                             title, ylabel='Light (mmol)', color='orange')
            if num == 'm':
                bottom, top = plt.ylim()
                plt.vlines(dt.date(2020, 12, 1), bottom, top)

            elif num == 'r':
                # Calculate % of prime daylight hours (8am-5pm) that plant is getting light above min thresholds
                mask = (plant_readings['datetime'].dt.hour >= 8) & (
                    plant_readings['datetime'].dt.hour <= 17)  # Grab values between 8am and 5pm
                plant_light_min = plant_table.loc[plant_id, 'light_min']
                light_above_threshold_pct = plant_readings.groupby(mask)['light'].apply(
                    lambda c: (c > plant_light_min).sum() / len(c)
                )[True]
                xmin = plant_readings.datetime.iloc[0]
                xmax = plant_readings.datetime.iloc[-1]
                xtext = plant_readings.datetime.iloc[int(
                    .65*len(plant_readings.datetime))]
                plt.hlines(
                    plant_light_min, xmin, xmax, label='Average watering threshold')
                bottom, top = plt.ylim()
                plt.text(
                    xtext, plant_light_min - .1*plant_light_min, f'Min. light threshold')
                plt.text(
                    xmin, .7*top, f'Percentage of prime daylight hours (8am-5pm) \nthat plant receives light above min threshold: {round(light_above_threshold_pct*100)}%')

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
            plant_month_mean = {}
            for month in np.arange(1, 13):
                if month in plant_readings.month.values:
                    # Grab only data for the month
                    plant_data_month = plant_readings.loc[plant_readings.month == month]
                    plant_month_mean[month] = plant_data_month.light.mean()
            df_ghi_month = pd.DataFrame.from_dict(
                ghi_month_sum, orient='index')
            df_plant_month = pd.DataFrame.from_dict(
                plant_month_mean, orient='index')
            # Normalize light levels based on GHI for the month
            df_plant_month_norm = df_plant_month / df_ghi_month

            # Percentage difference before/after 12/1/20
            before = df_plant_month_norm[7:12].mean()
            after = (df_plant_month_norm.loc[12] +
                     df_plant_month_norm.loc[1]) / 2
            pct_diff = round(((after-before)/before*100)[0])

            # Graph of normalized light levels and comparison between before/after 12/1/20
            months = [calendar.month_abbr[i] for i in range(1, 13)]
            df_plant_month_norm.index = months
            x = df_plant_month_norm.index
            y = df_plant_month_norm[0]
            fig = plt.figure()
            plt.bar(x, y, color='orange')
            plt.xlabel('Month')
            plt.ylabel('Normalized Relative Light Levels')
            plt.yticks([])
            plt.title(
                f'Comparison of Light Levels Before & After Move on 12/1/20 for {plant_name}')
            bottom, top = plt.ylim()
            plt.text(
                4, .9*top, f'Difference before/after 12/1/20: {pct_diff}%')

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

        # Learning algorithm that determines based on current soil moist, sunlight plant has been receiving and avg. temperature when it is next expected to need water

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

            # Determine soil moisture value when each plant is watered
            # Mean of the soil moist reading everytime it is detected that that plant was watered (mean of soil_moist_min)
            # avg_time_between_watering = plant_readings.loc[plant_readings.local_min_moist, 'soil_moist'].mean(
            # )
            # print(
            #     f'The average time between watering for {plant_name} is {avg_time_between_watering}')

            # Use values stored in SQL table to determine when each plant should be watered
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
                x, y, title, 'Soil Moisture (%)', color='blue')

            y = plant_readings.days_until_watering
            plt.plot(x, y, figure=fig_1,
                     label='Days until watering', color='orange')

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
            # plt.xlim(max(filter(None.__ne__, x)), 0)  # reverse x-axis

            # # Plot soil moisture vs. days until watering
            # fig_3 = plt.figure()
            # x = plant_readings.days_until_watering
            # y = plant_readings.soil_moist
            # plt.plot(x, y)
            # plt.xlabel('Days Until Watering')
            # plt.ylabel('Soil Moisture (%)')
            # plt.title(
            #     f'Soil Moisture vs. Days Until Watering for {plant_name} Plant')
            # plt.xlim(max(filter(None.__ne__, x)), 0) # reverse x-axis

            # # Plot temperature vs. days until watering
            # fig_4 = plt.figure()
            # x = plant_readings.days_until_watering
            # y = plant_readings.temp
            # plt.plot(x, y)
            # plt.xlabel('Days Until Watering')
            # plt.ylabel('Temperature (deg C)')
            # plt.title(
            #     f'Temperature vs. Days Until Watering for {plant_name} Plant')

            # # Plot soil fertility vs. days until watering
            # fig_5 = plt.figure()
            # x = plant_readings.days_until_watering
            # y = plant_readings.soil_fert
            # plt.plot(x, y)
            # plt.xlabel('Days Until Watering')
            # plt.ylabel('Soil Fertility (μS/cm)')
            # plt.title(
            #     f'Soil Fertility vs. Days Until Watering for {plant_name} Plant')
            # plt.xlim(max(filter(None.__ne__, x)), 0)  # reverse x-axis

            # Create column for rolling sum of light to analyze if that affects days until watering
            plant_readings['light_roll'] = plant_readings.light.rolling(
                96).sum()
            # fig_6 = plt.figure()
            # x = plant_readings.days_until_watering
            # y = plant_readings.light_roll
            # plt.plot(x, y)
            # plt.xlabel('Days Until Watering')
            # plt.ylabel('Light Received Over Past 6 Days')
            # plt.title(
            #     f'Past Light Received vs. Days Until Watering for {plant_name} Plant')
            # plt.xlim(max(filter(None.__ne__, x)), 0)  # reverse x-axis

            # Create column for rolling sum of temperature to analyze if that affects days until watering
            plant_readings['temp_roll'] = plant_readings.temp.rolling(
                96).mean()
            # fig_7 = plt.figure()
            # x = plant_readings.days_until_watering
            # y = plant_readings.temp_roll
            # plt.plot(x, y)
            # plt.xlabel('Days Until Watering')
            # plt.ylabel('Average Temperature Over Past 6 Days')
            # plt.title(
            #     f'Past Temperature vs. Days Until Watering for {plant_name} Plant')
            # plt.xlim(max(filter(None.__ne__, x)), 0)  # reverse x-axis

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
            df = plant_readings.dropna(
                subset=['days_until_watering', 'light_roll', 'temp_roll'])
            df.set_index('datetime', inplace=True)

            X = df.loc[:, ['light', 'soil_moist', 'temp', 'soil_fert']]
            # # uses rolling sums/avgs for light and temp
            # X = df.loc[:, ['light_roll',
            #                'soil_moist', 'temp_roll', 'soil_fert']]
            y = df.loc[:, 'days_until_watering']
            y_dumb = df.loc[:, 'dummy_days_until_watering']
            # Arrow of time - everything in test set must occur after training - no shuffling!
            X_train, X_test, y_train, y_test, y_dumb_train, y_dumb_test = train_test_split(
                X, y, y_dumb, test_size=0.35, shuffle=False)

            # Scale the training data with fit transform, and the testing data with transform only, so that each parameter counts equally toward learning
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            reg = LinearRegression()
            # reg = Lasso()
            reg.fit(X_train_scaled, y_train)
            y_pred = reg.predict(X_test_scaled)

            # reg_score = mean_squared_error(y_test, y_pred)
            # dummy_score = mean_squared_error(y_test, y_dumb_test)

            metric = 'r2'
            reg_score, dummy_score = return_score(
                y_test, y_pred, y_dumb_test, metric=metric)

            print(
                f"The {metric} for {plant_name} for the average days between watering ({round(avg_days_between_watering)} days) is:  {round(dummy_score, 2)}")
            print(
                f"The {metric} for {plant_name} for the regression model is:  {round(reg_score, 2)}\n")

            # Plot predicted vs. acutal on the same plot

            x = y_test.index
            title = f'Predicted vs. Actual Days Until Watering for {plant_name} Plant'
            fig_1 = plot_time_series(
                x, y_pred, title, 'Predicted days Until Watering', color='red')

            plt.scatter(x, y_test, figure=fig_1,
                        label='Actual days until watering', color='black', s=1)
            plt.legend()


# TODO - Forward-looking sunlight prediction - Connect to weather prediction API (sunny/cloudy) for light predictions? Correlate to light detected in training data


plt.show()
