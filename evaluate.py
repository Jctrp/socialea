import argparse
import yaml
from train_eval.evaluator import Evaluator
import os

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-config", default="D:/DL/socialea-main/configs/ant.yml")
parser.add_argument("-data_root", default="D:\Datasets/nuScenes")
parser.add_argument("-data_dir", default="D:\Datasets/nuScenes_processed")
parser.add_argument("-output_dir", default="D:\DL\socialea-main\outputs")
parser.add_argument("-checkpoint", default="D:\DL\socialea-main\outputs/checkpoints/best.tar")
args = parser.parse_args()


# Make directories
if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)
if not os.path.isdir(os.path.join(args.output_dir, 'results')):
    os.mkdir(os.path.join(args.output_dir, 'results'))


# Load config
with open(args.config, 'r') as yaml_file:
    cfg = yaml.safe_load(yaml_file)


# Evaluate
evaluator = Evaluator(cfg, args.data_root, args.data_dir, args.checkpoint)
evaluator.evaluate(output_dir=args.output_dir)
