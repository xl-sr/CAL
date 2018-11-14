Conditional Affordance Learning 
===============

Installation
------


```Shell
# install anaconda2 if you don't have it yet
wget https://repo.continuum.io/archive/Anaconda2-4.4.0-Linux-x86_64.sh
bash Anaconda2-4.4.0-Linux-x86_64.sh
source ~/.profile
# or use source ~/.bashrc - depending on where anaconda was added to PATH as the result of the installation
# now anaconda is assumed to be in ~/anaconda2
```

Now we will:
1. create a conda environment named CAL and install all dependencies
2. download the binaries for CARLA version 0.8.2 [[CARLA releases](https://github.com/carla-simulator/carla/releases)]
3. download the model weights

```Shell
git clone https://github.com/xl-sr/CAL.git
cd CAL

# create conda environment
./setup.sh

# run download script
./download_binaries_and_models.sh

```

Run the Agent
------
In CARLA_0.8.2/ start the server with for example: (see the [CARLA documentation](https://carla.readthedocs.io/en/stable/) for details)


```Shell
./CarlaUE4.sh Town01 -carla-server -windowed -benchmark -fps=20 -ResX=800 - ResY=600
```

Open a second terminal, cd into CAL/PythonClient/ and run:

```Shell
python driving_benchmark.py -c Town02 -v -n test

```
This runs the basic_experiment benchmark. '-n' is the naming flag (in this example the run is named "test"). If you want to run the CORL 2017 benchmark you need to run 

```Shell
python driving_benchmark.py -c Town02 -v -n test --corl-2017

``` 

If you want to continue an experiment, you can add the 'continue-experiment' flag.

Paper
-----

If you use this implementation, please cite our CoRL 2018 paper.

Conditional Affordance Learning for Driving in Urban Environments. 
<br> Sauer, Axel and Savinov, Nikolay and Geiger, Andreas. 
<br> CORL 2018 [[PDF](http://www.cvlibs.net/publications/Sauer2018CORL.pdf)]


```
@INPROCEEDINGS{Sauer2018CORL,
  author={Sauer, Axel and Savinov, Nikolay and Geiger, Andreas},
  title={Conditional Affordance Learning for Driving in Urban Environments},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2018}
}

