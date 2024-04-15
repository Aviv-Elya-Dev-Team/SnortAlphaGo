# SnortAlphaGo
Snort game played by AlphaGo insired model

# Requirements
First of all, please run `python -m pip install -r requirements.txt`

# Play
to play the game:
py game.py <player_type> <encode_type_RED> <encode_type_BLUE>


# Train
to train the models run one of train_<encode_type>.bat (stop it when you want, is infinite loop)

# Info

player_type:
	0: player Vs player
	1: cpu Vs cpu
	2: player Vs cpu

encode_type:
	0: legal moves
	1: board
	2: both