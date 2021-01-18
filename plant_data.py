import csv
import codecs
import re
import sqlalchemy
import datetime as dt
import pymysql
# import numpy as np
# import pandas as pd


def insert_data_sql(dates_array, data, plant_id):
    # iterate through each day
    for day_index, day in enumerate(dates_array):
        # iterate through each hour
        for row in data[plant_id*24: (plant_id+1)*24]:
            plant_datetime = dt.datetime.strptime(
                f'{day} {row[0]}', "%Y-%m-%d %H:%M")
            query = sqlalchemy.insert(readings_table).values(
                plant_id=plant_id+1,
                datetime=plant_datetime,
                light=row[(day_index * num_columns) + 1],
                soil_moist=row[(day_index * num_columns) + 2],
                temp=row[(day_index*num_columns) + 3],
                soil_fert=row[(day_index * num_columns) + 4]
            )
            result_proxy = connection.execute(query)
    # update new last import date
    new_last_import_date = last_date_data[plant_name].strftime(
        "%Y-%m-%d")
    query = sqlalchemy.update(plant_table).values(
        last_import_date=new_last_import_date).where(plant_table.columns.id == plant_id + 1)
    result = connection.execute(query)


# class Plant():
#     """Data container for each plant
#     ...

#     Attributes
#     ----------
#     name: str
#         plant common name
#     light: list
#         light data
#     soil_humid: list
#         soil humidity data
#     temp: list
#         temperature data
#     soil_fert: list
#          soil fertility data

#     Methods
#     -------
#     __str__(self)
#         Prints the plant's name
#     """

# def __init__(self, name, light=[], soil_humid=[], temp=[], soil_fert=[]):
#     self.name = name
#     self.light = light
#     self.soil_humid = soil_humid
#     self.temp = temp
#     self.soil_fert = soil_fert

# def __str__(self):
#     return self.name

# # Create class instances of each plant
# plant_list = []
# for plant_name in plant_list_titles:
#     plant_list.append(Plant(plant_name))

# Open and read-in the exported plant data
data = []
plant_dates = {}
last_date_data = {}
title_row = 'null'
start_row = 5
with codecs.open('2021-01-07-18-HHCC.csv', 'rU', 'utf-16') as f:
    csv_reader = csv.reader(f, delimiter='\t')
    for row_index, row in enumerate(csv_reader):
        try:
            if 'Flower Care' in row[0]:
                title_row = row_index + 1  # then the next line contains the table title
                start_row = row_index + 3  # the following 3rd line contains the start of the data
        except IndexError:  # necessary since some rows contain no data in columns
            pass
        if row_index == title_row:  # store plant names, dates, and last date in the data
            plant_name = row[0]
            plant_dates[plant_name] = ' '.join(row[1::4]).split()
            # Remove last date so that last date has all hours
            plant_dates[plant_name].pop()
            last_date_data[plant_name] = plant_dates[plant_name][-1]
            last_date_data[plant_name] = dt.datetime.strptime(
                last_date_data[plant_name], "%Y-%m-%d").date()  # Convert to date object
        elif row_index >= start_row:
            # Clean by replacing "--" elements with zeros
            row[:] = [0 if x == '--' else x for x in row]
            data.append(row)

# Store data into local database via tables

# Connect to local database
engine = sqlalchemy.create_engine(
    "mysql+pymysql://root:plant-outside-123-World@localhost/plant_data")
connection = engine.connect()
metadata = sqlalchemy.MetaData()

# Create new tables to store plant data
plant_table = sqlalchemy.Table('plants', metadata,
                               sqlalchemy.Column(
                                   'id', sqlalchemy.Integer(), primary_key=True),
                               sqlalchemy.Column(
                                   'name_common', sqlalchemy.String(100), nullable=False),
                               sqlalchemy.Column(
                                   'name_latin', sqlalchemy.String(100)),
                               sqlalchemy.Column(
                                   'soil_moist_min', sqlalchemy.Numeric()),
                               sqlalchemy.Column(
                                   'soil_moist_max', sqlalchemy.Numeric()),
                               sqlalchemy.Column(
                                   'soil_fert_min', sqlalchemy.Numeric()),
                               sqlalchemy.Column(
                                   'soil_fert_max', sqlalchemy.Numeric()),
                               sqlalchemy.Column(
                                   'light_min', sqlalchemy.Numeric()),
                               sqlalchemy.Column(
                                   'light_max', sqlalchemy.Numeric()),
                               sqlalchemy.Column(
                                   'temp_min', sqlalchemy.Numeric()),
                               sqlalchemy.Column(
                                   'temp_max', sqlalchemy.Numeric()),
                               sqlalchemy.Column(
                                   'last_import_date', sqlalchemy.Date()
                               ))
readings_table = sqlalchemy.Table('readings', metadata,
                                  sqlalchemy.Column(
                                      'plant_id', sqlalchemy.Integer(), primary_key=True),
                                  sqlalchemy.Column(
                                      'datetime', sqlalchemy.DateTime(), primary_key=True),
                                  sqlalchemy.Column(
                                      'light', sqlalchemy.Integer()),
                                  sqlalchemy.Column(
                                      'soil_moist', sqlalchemy.Integer()),
                                  sqlalchemy.Column(
                                      'soil_fert', sqlalchemy.Integer()),
                                  sqlalchemy.Column(
                                      'temp', sqlalchemy.Integer()),
                                  )

# Execute table creation
metadata.create_all(engine)

# Select last import date from database for each plant
query = sqlalchemy.select(
    [plant_table.columns.name_common, plant_table.columns.last_import_date])
result_proxy = connection.execute(query)
result_set = result_proxy.fetchall()

# Store last import dates as dictionary, store plant names
last_import_date = {}
sql_plant_list = []
for plant in result_set:
    last_import_date[plant[0]] = plant[1]
    sql_plant_list.append(plant[0])

# Insert plant data into database tables
num_columns = 4  # four columns of sensor data
plant_id = 0
for plant_name, dates_array in plant_dates.items():
    common_name = plant_name[0:plant_name.find('(') - 1]
    latin_name = re.search(r'\((.*?)\)', plant_name).group(1)

    # Check if plant name in SQL table does not exist, if so, create entry in plant table
    if common_name not in sql_plant_list:
        query = sqlalchemy.insert(plant_table).values(
            id=plant_id + 1,
            name_common=common_name,
            name_latin=latin_name)
        result_proxy = connection.execute(query)

    # Check against date of last import and only update for dates not in database

    if last_import_date[common_name] is None:
        insert_data_sql(dates_array, data, plant_id)
        print(
            f'Data was successfully imported for {plant_name} through {last_date_data[plant_name].strftime("%Y-%m-%d")}')
    elif last_import_date[common_name] is not None:
        if last_date_data[plant_name] > last_import_date[common_name]:
            insert_data_sql(dates_array, data, plant_id)
            print(
                f'More data was imported for {plant_name} through {last_date_data[plant_name].strftime("%Y-%m-%d")}')
        else:
            print(f'Data for {plant_name} is up to date.')
    plant_id += 1
