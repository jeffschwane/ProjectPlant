# ProjectPlant
This is the capstone project of the online [Python Data Science and Machine Learning course](https://codingnomads.co/courses/data-science-machine-learning-course?portfolioCats=3) by CodingNomads.

## Background
Over the course of the pandemic and quarantine, I’ve gotten more passionate about houseplants - collecting them and giving them their optimal growing conditions to make for a beautiful environment. Over the past year or so, I’ve learned which plants need the most light and the best placement for that, how frequently which plants need water, fertilizer, etc. However, it is a constant learning process, and by collecting data, it can take a bit of the guesswork around proper care. For the beginner houseplant owner, getting to this level of knowledge can be daunting and often involves killing many plants before getting it right. For instance overwatering plants and leaving them sitting in soggy soil is usually the #1 killer of plants, since it can lead to the growth of harmful microorganisms and root rot.

I was gifted four off-the-shelf digital houseplant sensors that I’ve been collecting data from for ~6 months now. Each sensor collects hourly data on:
- Light
- Temperature
- Soil moisture
- Soil fertility (basically a measure of fertilizer - nitrate, phosphate, and potassium)

Though the sensors come with an app to look at some of this data, the UI is lacking and I’d like to glean more insights from the data. 

## Problem Statement
I wanted to get data insights from the plant sensors, and answer questions such as:
- **When should I next plan to water this plant?** I.e. predict the number of days until plant needs to be watered
- My partner and I  moved to a new apartment in Dec 2020 - **does the new location get better light than the previous one?** If so, I want to quantify the increase.
- **Does the average number of days between watering vary** by plant type?
- **Is each plant type getting adequate light levels** for growth?


## Dataset
The dataset I used is from four *North Connected Home Plant Monitor sensors*, which have actively been collecting data in houseplants located in my apartment in Brooklyn, NY since May 15, 2020. The sensors store data locally, which is pushed to the phone via bluetooth when an app syncs with the sensors. Though the sensors collect hourly data, if enough time lapses between syncs, sensor data can be overwritten due to limited memory. Therefore, the data is not contiguous. I exported the data to CSV files to work with python.

## Solution
For this project, I did the following data collection and analysis to obtain results:
- Created a python file which reads csv data files from plant sensors and formats it correctly to create and update as needed all the data in a SQL relational database
- Created a separate python file which queries the database and stores the information in pandas DataFrames for the purpose of graphing / analysis and applying ML algorithms
- Created graphs for analysis including:
  - Light data in its entirety for each plant, including optimal light thresholds for each plant type
  - Light data over course of an average day each month for each plant
  - Global light data over the course of an average day each month for NYC
  - Soil moisture data showing what happens when the plant is watered
  - Comparison of light levels before/after move on 12/1/20
- Created a learning algorithm that determines based on current soil moist, sunlight the plant has been receiving and avg. temperature when it is next expected to need water (used linear, lasso, and ridge regressions)
  - The target variable I created for this was the metric “days until watering”
  - This variable was determined by setting an ideal soil moisture threshold for each plant based on previous watering data and experience


## Benchmark Model
The baseline I want to beat in order to determine the project is a success is to measure against the “dummy model” of the average time between waterings for each plant. If the model does better than the average, that tells me it has actually learned something about the data fed to it.

## Evaluation Metrics
1. Mean-squared error
2. Mean-absolute Error
3. R-squared

## Requirements
- Python external libraries are documented in the requirements.txt file
- I used Python 3.8.3 and MySQL Workbench 8.0
- Set up API access to the [National Solar Radiation Database](https://nsrdb.nrel.gov/data-sets/api-instructions.html)
- API keys and SQL password are stored in a .env file as environment variables

