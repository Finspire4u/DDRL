# MADDPG_MAC
## This project contains **three** folders: MADDPG, MARLENV and MARLENV_baseline.
MADDPG is the algorithm folder, MARLENV is the simulation folder with the algorithm, MARLENV_baseline is the one without the algorithm.
To use the project, you may need to **install those repositories**:
python, tensorflow, gym, numpy, pandas, time, random, sklearn, matplotlib, openpyxl
If you’re using TensorFlow 2, then no need to change the import codes/lines in “MARLENV”

## Then **install MADDPG**, in the command line, type: > pip install -e [your MADDPG directory]
Third, go to your MARLENV folder and type: > python train.py to train the model.
Last, after training, type:> python test.py to test the model.
This algorithm was built using [GitHub Pages](https://github.com/openai/maddpg).

## Within **xxx_baseline.py**:
If you want to change the number of nodes (line 13 in test.py), then you need to change their x_axis position in line 101 of env_fortest0.py
Output: 10 times testing results about the delay, lost packets amount, throughput amount and throughput percentage.
Figures: 10 times environment observations. The upper left subfigure is the whole observation before noises, the upper right subfigure is the single observation from node 3 before noises, the lower left subfigure is the whole observation after noises, the lower right subfigure is the single observation from node 3 after noises.

## Within **xxx_train/xxx_test**:
The major differences between env_fortrain7.py and env_fortest7.py are the size of the waiting transmitted data of the source node.
The major differences between train.py and test.py are save_model or load_model.
### Change the number of nodes
If you want to **change the number of nodes** (line 25 in train.py), then you need to change their x_axis position in line 108 of env_fortrain.py. Please also find the same locations in test.py and env_fortest.py
### Change the area size of degree of traffic load
If you want to **change the area size of degree of traffic load**, please also change the observation size (line 54 in env_fortrain.py and env_fortest.py)

## **Training Output**: 
“Episodes/saving rate” times training results and last 10 models:
Training results contains throughput and throughput percentage within last saving rate episodes; Rewards of agents within last saving rate episodes; Lost packets within last saving rate episodes and Delays of agents within last saving rate episodes.

### How to switch ‘surrounding pixels’ to ‘pixels in specific quadrant’?
Command all lines include ‘Get_drone_obs’ and ‘Check_surrounding’
Uncommand all lines include ‘Get_drone_obs_toward’ and ‘Check_surrounding_toward’

## **Before testing**:
Change your favorite saved model’s file names to format ones, for example:
-500.data-00000-of-00001  |  -500.index  |  -500.meta  ======>  .data-00000-of-00001    |    .index    |    .meta

### **Testing Output**: 
Testing results about the delay, lost packets amount, throughput amount and throughput percentage.
