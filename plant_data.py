import csv
import codecs
import sqlalchemy
import datetime as dt
import pymysql
# import numpy as np
# import pandas as pd


class Plant():
    """Data container for each plant
    ...

    Attributes
    ----------
    name: str
        plant common name
    light: list
        light data
    soil_humid: list
        soil humidity data
    temp: list
        temperature data
    soil_fert: list
         soil fertility data

    Methods
    -------
    __str__(self)
        Prints the plant's name
    """

    def __init__(self, name, light=[], soil_humid=[], temp=[], soil_fert=[]):
        self.name = name
        self.light = light
        self.soil_humid = soil_humid
        self.temp = temp
        self.soil_fert = soil_fert

    def __str__(self):
        return self.name


# Open and read-in the exported plant data
data = []
plant_list_titles = []
dates = set()  # FIXME - dates need to be unique for every plant in the plant list since they are different - make dates an object that has a plant attribute
plant_dates = {}
title_row = 'null'
start_row = 5
# TODO - Ensure that only the newest data is imported and stored - use the SQL databased to store the date of last import
with codecs.open('2021-01-07-18-HHCC.csv', 'rU', 'utf-16') as f:
    csv_reader = csv.reader(f, delimiter='\t')
    for row_index, row in enumerate(csv_reader):
        try:
            if 'Flower Care' in row[0]:
                title_row = row_index + 1  # then the next line contains the table title
                start_row = row_index + 3  # the following 3rd line contains the start of the data
        except IndexError:  # necessary since some rows contain no data in columns
            pass
        if row_index == title_row:  # store plant names and dates
            plant_name = row[0]
            plant_dates[plant_name] = ' '.join(row[1::4]).split()
            plant_list_titles.append(plant_name)
        elif row_index >= start_row:
            # Clean by removing "--" elements
            row[:] = [0 if x == '--' else x for x in row]
            data.append(row)


# # Create class instances of each plant
# plant_list = []
# for plant_name in plant_list_titles:
#     plant_list.append(Plant(plant_name))

# # loop through data and parse out sensor data into plant class instances
# # TODO - Expand this to include all the plants
# column = 1
# while column != 'end':
#     try:
#         pass
#     except IndexError:
#         column = 'end'
#         break
# for row in data[0:24]:
#     try:
#         plant_list[0].light.append(row[column])
#         plant_list[0].soil_humid.append(row[column + 1])
#         plant_list[0].temp.append(row[column + 2])
#         plant_list[0].soil_fert.append(row[column + 3])
#     except IndexError:
#         pass
# iterate betweenall plants

# write_plant_data_for(i, column)
# column += 4

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
                                   'temp_max', sqlalchemy.Numeric())
                               )
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

# Insert plant data into database tables
num_columns = 4  # four columns of sensor data
plant_id = 0  # TODO check if entry already exists, and if not don't insert
for plant_name, dates_array in plant_dates.items():
    query = sqlalchemy.insert(plant_table).values(
        id=plant_id + 1,
        name_common=plant_name)
    result_proxy = connection.execute(query)
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
    plant_id += 1
