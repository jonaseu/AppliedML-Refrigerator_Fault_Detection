# Refrigeration Machine Learning Exploration for Diagnostics

## 1. Objective 

This repository focuses on exploring machine learning for refrigeration category by using dataset from real products from Torino project to detect faults. The manner that this exploration is done is trough multiple jupyter notebooks, with step by step of the code as well as the rationale behind those code.

![MLOBJECTIVE](Notebooks\figures\ML_Objective.png)

### 1.1 Why this matter?

Troughout industry, No fault found (NFF) accounts for about 30-70\% of the returned faulty products. In the end that drastically impacts Whirlpool costs and perceived quality.

The capability to detect faulty products precisely is essential for Whirlpool in order to:
 - Achieve Premium Quality, by reacting faster to problematic products at costumer house
 - Reducing Quality costs by having more precise fault detection, thus avoiding service reocurrances and repeated part swaps
 - Deepen the knowledge about the refrigeration faults, thus improving engineering for future and current running projects

### 1.2 What is the output of this repo?

The main manner that this repo contributs to Whirlpool is by:
- ✅ Provide a ready to use pre-processed dataset of real refrigerator temperature data
- ⬜️ Share knowledge between Machine Learning, Software Engineers and Cooling Engineers of Faults and overall data of our products
- ⬜️ Test and document multiple approaches to machine learning to detect faults
- ⬜️ Use the knowledge obtained to kick off the main actions needed for Whirlpool to improve on diagnostics for Refrigeration

***
## 2. Summary of the Data Source

The raw dataset is obtainged from data collected from RTS Servers for Torino Project.

### 2.1 RTS

RTS is the most common tool used by Whirlpool to collect data from our Refrigerators

RTS logs most of the cooling units during CV/DV and even production. The logs usually consists of:
- Readings of temperature from thermocouples
- Product wattage
- Sometimes contains WIN Information, thus usually leading to 

RTS then collects all this data and keeps publishing them live on their main software, which can be installed [here](https://drive.google.com/file/d/1XR1GcppAtwFrlljwIm1m1HZnkwb4ECor/view?usp=share_link), and for products that have finished their logs, the logs are stored on RTS servers that can be accessed [here](http://adc-rtswebp1.na.ad.whirlpool.com/labresults/mainapp.aspx) (both require VPN). Both links are related to Cooling/Cooling Test Engineers, they can provide more details about such tools. 

Below is an example of such logs that can be analyzed by the RTS Post Analyzer (software included on the RTS toolset).

![RTSLOG](Notebooks\figures\RTS_Log.png)

### 2.2 Torino Product

[Torino](https://rm.whirlpool.oncloudone.com/rm/web#action=com.ibm.rdm.web.pages.showArtifact&artifactURI=https%3A%2F%2Frm.whirlpool.oncloudone.com%2Frm%2Fresources%2F_huaXsXzYEemvisFqZmkFAA&vvc.configuration=https%3A%2F%2Frm.whirlpool.oncloudone.com%2Frm%2Fcm%2Fstream%2F_qZzlsCEkEea1esD_TUzLUg&componentURI=https%3A%2F%2Frm.whirlpool.oncloudone.com%2Frm%2Frm-projects%2F_qYFucSEkEea1esD_TUzLUg%2Fcomponents%2F_qZvUQCEkEea1esD_TUzLUg) is a SxS product with FC,RC and Pantry compartments and an stand-alone icemaker with Ice and Water Dispensing.
From Software perspective that can be summarized as:

Inputs:
- **RC Sensor**: Refrigerator Compartment (RC) Temperature sensor
- **FC Sensor**: Freezer Compartment (FC) Temperature sensor
- **Pantry Sensor**: Pantry Compartment (temperature controllable drawer) Temperature sensor
- **FC Evap Sensor**: Freezer evaporator Sensor

Loads:
- **Compressor**: motor that moves cooling fluid to the evaporator, thus cooling it down
- **FC Fan**: fan that circulates the air from the evaporator to the other compartments
- **RC Damper**: flapper door that allows air to flow from Freezer to Refrigerator
- **Pantry Damper**: flapper door that allows air to flow from Freezer to Pantry
- **Defrost Heater**: resistance that will ocasionally melt the ice formed on the freezer evaporator
- **Condenser Fan**: fan that removes the heat from the evaporator to allow compressor to work properly
- **Ice and Water Dispensing Loads**: Water Valve, Isolation Water Valve, Auger Motor and Ice Door Motor, which faults cannot easily be predicted by temperature charts

For who is not so familiar with a refrigerator, as a big summary:
- Whenever the RC or FC temperature gets too hot, the Compressor turns on, with a speed defined by a PI controller. Compressor turning ON reduces the temperature of the evaporator
- Whenever RC Temperature gets too hot, the RC Damper opens, allowwing air to flow from FC to RC. Similarly for Pantry sensor/damper
- Whenever RC,FC or Pantry are too hot, the FC Fan turns on to circulate the air

A compartment is defined as "too hot" or, in the usual verbatin, COOLING, whenever its sensor is above the setpoint temperature plus a threshold. When the compartment is below the setpoint minus a threshold, it's considered as SATISFIED.

***
## 3. Setup and Getting Started

This reposotitory contains multiple Jupyter Notebooks, which are blocks of code together with blocks of texts and image to create a 

### 3.1 Setup
In order to run jupyter notebooks, it's recommend that you use:
- Python 3.11.1 or above
- Visual studio code to run the Jupyter notebooks

It's also recommend that you create your own virtual environment, to do that, on your Visual Studio Code Terminal run:
```console
python venv -m venv
```

After that, you can start up this virtual environment by:
```console
venv/Scripts/activate.bat
```

Finally, when engaged on that venv, you can install all the necessary libraries in advance by:
```console
python venv -m venv
```


### 3.2 Getting Started
The Notebooks folder contains all the Jupyter Notebooks with different phases of a machine learning process by order. The initial notebooks, from 0 up to 2 show some analyzes of the raw data 

Below is a more detailed explanation of what this repo is about, where most of the processes highlighted in blue have their own notebook

![title](Notebooks\figures\Detailed_Pipeline.png)


It's recommend to start at:
- [2.EDA - Analyzing Pre Processed Data](Notebooks\2.EDA-Analyzing_Pre_Processed.ipynb): as in this step the data has already been pre processed and it's shown the basic temperature charts and their analysis

