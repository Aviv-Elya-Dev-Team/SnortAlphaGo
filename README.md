# SnortAlphaGo
Snort game played by AlphaGo insired model

# Requirements
First of all, please run `python -m pip install -r requirements.txt`

# Play
to play the game:
`python Game.py <player_type> <encode_type_RED> <encode_type_BLUE>`

example: `python Game.py 2 2`

# Train
to train the models run 
`python Agent.py <encode_type>`
(stop it when you want, is infinite loop)

# Info

### player_type:
0: player Vs player

1: cpu Vs cpu

2: player Vs cpu

### encode_type:
0: legal moves

1: board

2: both