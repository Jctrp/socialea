import argparse
from train_eval.preprocessor import preprocess_data
import yaml


parser = argparse.ArgumentParser()
parser.add_argument("-config", default="")
parser.add_argument("-data_root", default="")
parser.add_argument("-data_dir", default="")
args = parser.parse_args()

# Read config file
with open(args.config, 'r') as yaml_file:
    cfg = yaml.safe_load(yaml_file)

preprocess_data(cfg, args.data_root, args.data_dir)
