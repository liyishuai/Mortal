import os
import tomllib

config_file = os.environ.get('MORTAL_CFG', 'config.toml')
with open(config_file, 'rb') as f:
    config = tomllib.load(f)
