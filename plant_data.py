import csv
import codecs
import sqlalchemy

# Open and read-in the exported plant data
row_list = []
with codecs.open('2021-01-07-18-HHCC.csv', 'rU', 'utf-16') as f:
    csv_reader = csv.reader(f, delimiter='\t')
    for row in csv_reader:
        try:
            # then the next line contains the table title
            if 'Flower Care' in row[0]:
                title = True
        except IndexError:  # necessary since some lines contain no columns
            pass
        row_list.append(row)


# Organize data into seperate tables for each sensor, where columns are 'L=Light(mmol)', ' * S=Soil humidity(%)', ' * T=Temperature(℃)', ' * E=EC(μs/cm)


# Store data into local database via tables

# Connect to local database
engine = sqlalchemy.create_engine(
    "mysql+pymysql://root:plant-outside-123-World@localhost/plant_data")
connection = engine.connect()
metadata = sqlalchemy.MetaData()

# Create new tables to store tweets
plant_table = sqlalchemy.Table('plant', metadata,
                               sqlalchemy.Column(
                                   'id', sqlalchemy.Integer(), primary_key=True),
                               sqlalchemy.Column(
                                   'name_common', sqlalchemy.String(100), nullable=False),
                               sqlalchemy.Column(
                                   'name_latin', sqlalchemy.String(100), nullable=False),
                               sqlalchemy.Column(
                                   'soil_moist_min', sqlalchemy.Numeric(), nullable=False),
                               sqlalchemy.Column(
                                   'soil_moist_max', sqlalchemy.Numeric(), nullable=False),
                               sqlalchemy.Column(
                                   'soil_fert_min', sqlalchemy.Numeric(), nullable=False),
                               sqlalchemy.Column(
                                   'soil_fert_max', sqlalchemy.Numeric(), nullable=False),
                               sqlalchemy.Column(
                                   'light_min', sqlalchemy.Numeric(), nullable=False),
                               sqlalchemy.Column(
                                   'light_max', sqlalchemy.Numeric(), nullable=False),
                               sqlalchemy.Column(
                                   'temp_min', sqlalchemy.Numeric(), nullable=False),
                               sqlalchemy.Column(
                                   'temp_max', sqlalchemy.Numeric(), nullable=False)
                               )
readings_table = sqlalchemy.Table('readings', metadata,
                                  sqlalchemy.Column(
                                      'id', sqlalchemy.Integer(), primary_key=True),
                                  sqlalchemy.Column(
                                      'datetime', sqlalchemy.DateTime(), primary_key=True),
                                  sqlalchemy.Column(
                                      'light', sqlalchemy.Integer()),
                                  sqlalchemy.Column(
                                      'light', sqlalchemy.Integer()),
                                  )
