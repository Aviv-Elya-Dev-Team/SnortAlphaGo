# SnortAlphaGo
Snort game played by an AlphaGo inspired model

# Requirements
First of all, please run `python -m pip install -r requirements.txt`

# Play
to play the game:
`python Game.py <player_type> <encode_type_RED> <encode_type_BLUE>`

example: `python Game.py 2 2` will have you play against the "both" model according to the current config.ini file

# Config.ini
its a file to control the board size and a few other parameters. You shouldn't touch it for the most part, however, we trained a model on a board of size 8 by 8 first for performance reasons, and then we moved to a 10 by 10 once we had a more optimized model, so if you want to see our progress you can change it.

### how to change it:
if you want to play against the 8x8 board models, you should change the board_size to 8, but also change screen_height, screen_width and cell_size such that `screen_height/cell_size = board_size` and `screen_height = screen_width`

same goes for going back to a 10 by 10 board, the models are different and will change depending on the board_size

Our best model is model_0_10

# Train
to train the models run 
`python Agent.py <encode_type>`
(stop it when you want, is infinite loop)

# Scripts
You can also run the scripts from the scripts folder, they work just like running the commands above for train and play

# Info

### player_type:
0: player Vs player

1: cpu Vs cpu

2: player Vs cpu

### encode_type:
0: legal moves

1: board

2: both

3: MCTS