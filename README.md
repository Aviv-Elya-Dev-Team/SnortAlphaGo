# SnortAlphaGo
Snort game played by AlphaGo insired model

to train the models run one of train_<encode_type>.bat (stop it when you want, is infinite loop)

to play the game:
py game.py <player_type> <encode_type_RED> <encode_type_BLUE>

player_type:
	0: player Vs player
	1: cpu Vs cpu
	2: player Vs cpu

encode_type:
	0: legal moves
	1: board
	2: both