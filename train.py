import argparse
import yaml
from train_eval.trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
import os

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-config", default="D:/DL/socialea-main/configs/new.yml")
parser.add_argument("-data_root", default="D:\Datasets/nuScenes")
parser.add_argument("-data_dir", default="D:\Datasets/nuScenes_processed")
parser.add_argument("-output_dir", default="D:\DL\socialea-main\outputs")
parser.add_argument("-num_epochs",  default=120)
parser.add_argument("-checkpoint", required=False)
args = parser.parse_args()

# Make directories
if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)
if not os.path.isdir(os.path.join(args.output_dir, 'checkpoints')):
    os.mkdir(os.path.join(args.output_dir, 'checkpoints'))
if not os.path.isdir(os.path.join(args.output_dir, 'tensorboard_logs')):
    os.mkdir(os.path.join(args.output_dir, 'tensorboard_logs'))

# Load config
with open(args.config, 'r') as yaml_file:
    cfg = yaml.safe_load(yaml_file)

# Initialize tensorboard writer
writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard_logs'))

# Train
trainer = Trainer(cfg, args.data_root, args.data_dir, checkpoint_path=args.checkpoint, writer=writer)
trainer.train(num_epochs=int(args.num_epochs), output_dir=args.output_dir)

# Close tensorboard writer
writer.close()
