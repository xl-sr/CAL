:star2: We updated the repo with training code! :star2:

## Conditional Affordance Learning  ##
#### In CoRL 2018 [[Paper]](https://arxiv.org/abs/1806.06498) [[Video]](https://www.youtube.com/watch?v=UtUbpigMgr0) [[Talk]](https://www.youtube.com/watch?v=SceH3Al9w_M)

[Axel Sauer<sup>1, 2</sup>](https://www.msrm.tum.de/rsi/team/wissenschaftliche-mitarbeiter/sauer-axel/), 
[Nikolay Savinov<sup>1</sup>](http://people.inf.ethz.ch/nsavinov/), 
[Andreas Geiger<sup>1, 3</sup>](https://ps.is.tuebingen.mpg.de/person/ageiger)
<br/>
<sup>1</sup> ETH Zurich, <sup>2</sup> TU Munich, <sup>3</sup> MPI for Intelligent Systems and University of Tubingen<br/>

<p align="center">
  <img src="CAL.gif" width="480">
</p>

If you use this implementation, please cite our CoRL 2018 paper.
```
@inproceedings{Sauer2018CORL,
  author={Sauer, Axel and Savinov, Nikolay and Geiger, Andreas},
  title={Conditional Affordance Learning for Driving in Urban Environments},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2018}
}
```
### Installation

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
conda env create -f requirements.yml
source activate CAL

# run download script
./download_binaries_and_models.sh

```

### Run the Agent

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

### Training
```Shell
cd training/

# download and untar the dataset
wget https://s3.eu-central-1.amazonaws.com/avg-projects/conditional_affordance_learning/dataset.tar.gz
tar -xzvf dataset.tar.gz

# create the training environment
conda env create -f requirements.yml
source activate training_CAL
``` 

Now, open training_CAL.ipynb. The notebook walks you through the steps to train a network on the dataset.
