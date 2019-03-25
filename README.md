# DQN
To create deep neural network to play atari games by using Deep Q-learning <br />
with more advance DQN modification such as: Double DQN, Dueling DQN


![breakout](https://user-images.githubusercontent.com/27904418/54891358-50c37f00-4e6a-11e9-8c11-1b8e0e750b86.gif)<br />
(the AI after 200000 games self-training)




# How to use it 
notice:
in the following tutorial, you need to
1. run in to certain directory with name "solve_xxx" (xxx is the game name)
2. add the repository into python system path

## train model 
### runing code:
```
usage: train_model.py [-h] [-p SAVE_PATH] [-t THREAD_NUM] [-n GAME_NUM]
                           [-u UPDATE_PERIOD] [-e1 EPSILON_MIN]
                           [-e2 EPSILON_DECREASE] [-g GAMMA]
                           [-l LEARNING_RATE] [-d1] [-d2]
                           run_name
                           
*positional arguments:
  run_name              the name of the training model

*optional arguments:
  -h, --help            show this help message and exit
  -p SAVE_PATH, --save_path SAVE_PATH
                        the save path of checkpoint
  -t THREAD_NUM, --thread_num THREAD_NUM
                        the number of playing thread
  -n GAME_NUM, --game_num GAME_NUM
                        the number of training games
  -u UPDATE_PERIOD, --update_period UPDATE_PERIOD
                        the update period of target network
  -e1 EPSILON_MIN, --epsilon_min EPSILON_MIN
                        the epsilon of exploration
  -e2 EPSILON_DECREASE, --epsilon_decrease EPSILON_DECREASE
                        the epsilon decrease
  -g GAMMA, --gamma GAMMA
                        the gamma of DQN learning
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        the learning rate of DQN training
  -d1, --double_DQN     adopt double DQN if call
  -d2, --dueling_DQN    adopt dueling DQN if call
```
This command will train the game model and output the model checkpoint and logs in the result folder which looks like

```
|- result/
  |- <run_name>/ # the model name
    |- tensorboard/ # the log files of training
    |- a bunch of check point files...
```
If there is a another result files with the same run_name while running the code, we can select either 
1. exit
2. resume the training from last check point
3. start a new training and overwrite the old one
4. rename the run name and start a new training


## test the model
### running code:
```
usage: test_model.py [-h] [-e EPISLON] [-n GAMES_NUM] [-p LOAD_PATH] [-s]
                     [-d1] [-d2]
                     run_name

positional arguments:
  run_name              the name of the training model

optional arguments:
  -h, --help            show this help message and exit
  -e EPISLON, --epislon EPISLON
                        the epislon of exploration
  -n GAMES_NUM, --games_num GAMES_NUM
                        the number of training games
  -p LOAD_PATH, --load_path LOAD_PATH
                        the load path of checkpoint
  -s, --show            whether to show the gameplay screen
  -d1, --double_DQN     adopt double DQN if call
  -d2, --dueling_DQN    adopt dueling DQN if call
```
This command will test the <load_path>/<run_name> model by playing several games <br />
*NOTICE: the optional argument d1, d2 should be set the same as you train
