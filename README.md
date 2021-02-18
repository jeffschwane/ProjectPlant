# ProjectPlant

## Background
Over the course of the pandemic and quarantine, I’ve gotten more passionate about houseplants - collecting them and giving them their optimal growing conditions to make for a beautiful environment. Over the past year or so, I’ve learned which plants need the most light and the best placement for that, how frequently which plants need water, fertilizer, etc. However, it is a constant learning process, and by collecting data, it can take a bit of the guesswork around proper care.

I was gifted four off-the-shelf digital houseplant sensors that I’ve been collecting data from for ~6 months now. Each sensor collects hourly data on:
- Light
- Temperature
- Soil moisture
- Soil fertility (basically a measure of fertilizer - nitrate, phosphate, and potassium)

Though the sensors come with an app to look at some of this data, the UI is lacking and I’d like to glean more insights from the data. 

## Problem Statement
I would like to get data insights from the plant sensors, and answer questions such as:
- When should I next plan to water this plant? I.e. predict the number of days until plant needs to be watered (As an aspiring plant dad, one of the things I’ve learned is that overwatering plants is usually the #1 killer of plants, since it can lead to root rot.)
- My partner and I  moved to a new apartment in Dec 2020 - does the new location get better light than the previous one? If so, I want to quantify the increase.
- Does the average number of days between watering vary by plant type?
- Is each plant type getting adequate light levels for growth?


## Dataset
The dataset I will be using is from four *North Connected Home Plant Monitor sensors*, which have actively been collecting data in houseplants located in my apartment in Brooklyn, NY since May 15, 2020. The sensors store data locally, which is pushed to the phone via bluetooth when an app syncs with the sensors. Though the sensors collect hourly data, if enough time lapses between syncs, sensor data can be overwritten due to limited memory. Therefore, the data is not contiguous. I am able to export the data to CSV files to work with python. Each sensor collects four different measurements every hour:
- light
- soil moisture
- temperature
- soil fertility

## Solution Statement
My solution to the problem is to input the data into a pandas dataframe and use matplotlib to create graphs to analyze the data. I will use a linear regression model based on a training dataset where I label “days until watering” based on the past data when I know I have watered. It should be easy to see this in the past data by analyzing the soil moisture readings. Every time I actually went ahead and watered a monitored plant in the past, I will see a spike from say 5% soil moisture to ~40% soil moisture within one hour. I know that I did a pretty good job of watering it when it actually needed water since I use a separate analog soil moisture meter to probe the depth of the soil and make sure things were dry enough before I watered again. So all I have to do is take the mean of all the local minimums of the soil moisture readings for any given plant to get a good indicator about when it should be watered based on the soil moisture sensor readings.
What I will try and predict is the value “days until next watering”. I will therefore need to create labels for this for the ML algorithm for my past data. I figure I can just write some python code to do this (after I figure out the previous question) and just store it as a new column in the pandas dataframe. My theory is that the primary factors that I have access to that will affect this are:
- light readings from the past day or past several days (plants suck up water through transpiration, and light is a factor that drives this)
- temperature
- current soil moisture
- other fixed factors inherent to the plant/setup that hopefully the model will learn - plant type, pot type/size/shape, soil type/drainage

There’s one more factor that I could see really driving this, which is “what will the sunlight look like over the course of the next few days?”. With a week of all sunny days, a plant might need water say on day 4, whereas a week of all cloudy days (with the same starting conditions), it could be day 7. So something I plan to look at at the end is connecting to a weather prediction API, and using those predictions to try and make better forecasts of “days until next watering”.

## Benchmark Model
The baseline I want to beat in order to feel like the project is a success is to measure against the “dummy model” of the average time between waterings for each plant. If the model does better than the average, that tells me it has actually learned something about the data fed to it.

## Evaluation Metrics
1. Mean-squared error
2. TBD
3. TBD

## Project Design
1. Create a python file which takes csv data dumps that I get from plant sensors and formats it correctly to create and update everything in a SQL database
2. Create a separate python file which queries the database and stores the information in pandas DataFrames for the purpose of graphing / analysis and applying ML algorithms
   - Connect and load SQL data into pandas DataFrames
   - Ask user for input on types of graphs to display
     - Light data in its entirety for each plant
     - Light data over course of an average day each month for each plant
     - Soil moisture data showing what happens when the plant is watered
     - Comparison of light levels before/after move on 12/1/20
   - Create a learning algorithm that determines based on current soil moist, sunlight plant has been receiving and avg. temperature when it is next expected to need water

