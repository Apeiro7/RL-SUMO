# RL-SUMO

## How to train new Networks.

you need to download SUMO GUI for running simulations.

## SUMO Installation Guide

download sumo gui from [here](https://sumo.dlr.de/docs/Downloads.php)


This guide will help you install the SUMO (Simulation of Urban MObility) traffic simulator on both Linux (Ubuntu/Debian-based) and macOS.

## Linux (Ubuntu/Debian-based)

1. **Add the SUMO repository**:
    ```bash
    sudo add-apt-repository ppa:sumo/stable
    ```

2. **Update your package list**:
    ```bash
    sudo apt-get update
    ```

3. **Install SUMO**:
    ```bash
    sudo apt-get install sumo sumo-tools sumo-doc
    ```

## Mac (using Homebrew)

1. **Install Homebrew** (if not already installed):
    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

2. **Install SUMO**:
    ```bash
    brew install sumo
    ```

## Verifying Installation

After installation, verify that SUMO is installed correctly by typing the following command in your terminal:

```bash
sumo
```

## Step1: create network and route file

Use SUMO netedit tool to create a network<br/>
for example 'network.net.xml' and save it in the maps folder.

cd into maps folder and run the following command

```bash
python randomTrips.py -n network.net.xml -r routes.rou.xml -e 500
```

This will create a routes.rou.xml file for 500 simulation steps for the network "network.net.xml"

### Step2: Set Configuration file.

You need to provide network and route files to the Configuration file.<br/>
change net-file and route-files in input.

```bash
<input>        
  <net-file value='maps/city1.net.xml'/>
  <route-files value='maps/city1.rou.xml'/>
</input>
```

### Step3: Train the model.

Now use the train.py file to train a model for this network.<br/>

```bash
python train.py --train -e 50 -m model_name -s 500
```

This code will train the model for 50 epochs.<br/>
-e is to set the epochs.<br/>
-m for model_name which will be saved in the model folder.<br/>
-s tells simulation to run for 500 steps.<br/>
--train tells the train.py to train the model if not specified it will load model_name from the model's folder.

At the end of the simulation, it will show time_vs_epoch graphs and save them to plots folder with name time_vs_epoch_{model_name}.png

### Step4: Running trained model.

You can use train.py to run a pre-trained model on GUI.

```bash
python train.py -m model_name -s 500
```

This will open GUI which you can run to see how your model performs.
To get accurate results set a value of -s the same for testing and training.

### Extra: Running Ardunio
Currently, Arduino works only for a single crossroad.<br/>
More than one cross road will return an error.<br/>

For running Arduino for testing use --ard.

```bash
python train.py -m model_name -s 500 --ard
```

### DATA extraction

0) Sumo Floating Car Data (FCD) Trace File<br/>

```bash
sumo -c configuration.sumocfg --fcd-output sumoTrace.xml
```

1) Raw vehicle positions dump: <br/>

```bash
sumo -c configuration.sumocfg --netstate-dump my_dump_file.xml
```

2) Emission Output: Amount of CO2, CO, HC, NOX, fuel, electricity, noise, emitted by the vehicle in the actual simulation step<br/>

```bash
sumo -c configuration.sumocfg --emission-output my_emission_file.xml
```

3) Full Output: 
dump every information contained in the network, including emission, position, speed, lane. 
Warning!!! takes a lot of time to accomplish this task and the file size is very big (~GB) <br/>

```bash
sumo -c configuration.sumocfg --full-output my_full_output.xml
```

4) SUMO Lane change Output<br/>

```bash
sumo -c configuration.sumocfg --lanechange-output my_lane_change_file.xml
```

5) SUMO VTK Output<br/>

```bash
sumo -c configuration.sumocfg --vtk-output my_vkt_file.xml
```

## QL Graph
![QL Graph](ql/notes/maps/Figure_3.png)

### Credits: 
1. Amit Bhardwaj<br/>
`SOS E&T Guru Ghasidas University`<br/>
`Bachelor of Technology - Computer Science and Engineering`<br/>
`2021 - 2025`
