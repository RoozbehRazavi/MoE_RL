import imageio
import torch
import os
import re
import yaml
import csv
import logging
from logging import Handler
import shutil


def create_directory(dir_path):
    # Check if the directory exists
    if os.path.exists(dir_path):
        return

    # Create the directory again
    os.makedirs(dir_path)
    print(f"Created new directory: {dir_path}")


def recreate_directory(dir_path):
    # Check if the directory exists
    if os.path.exists(dir_path):
        # Remove the directory and its contents
        shutil.rmtree(dir_path)
        print(f"Removed existing directory: {dir_path}")

    # Create the directory again
    os.makedirs(dir_path)
    print(f"Created new directory: {dir_path}")


def save_frames_as_video(frames, filename):
    writer = imageio.get_writer(filename, fps=20)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def pad_sequences(sequences, max_len, dim, batch_size):
    padded = torch.full((batch_size, max_len, dim), 0, dtype=torch.float32)
    for i, batch in enumerate(sequences):
        for j, transition in enumerate(batch):
            padded[i, j] = torch.tensor(transition)
    return padded


def find_latest_checkpoint(directory):
    # This pattern matches "model_checkpoint" followed by any number (the iteration), and ends with ".pth"
    pattern = re.compile(r'model_checkpoint(\d+)\.pth$')

    highest_iteration = -1
    latest_file = None

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Check if the filename matches the specified format
        match = pattern.match(filename)
        if match:
            # Extract the iteration number and convert to an integer
            iteration = int(match.group(1))
            # If this iteration is higher than the current highest, update the highest and latest file
            if iteration > highest_iteration:
                highest_iteration = iteration
                latest_file = filename

    # Return the path to the file with the highest iteration
    if latest_file:
        return os.path.join(directory, latest_file)
    else:
        return None


def save_config_to_yaml(args, filepath):
    with open(filepath, 'w') as yaml_file:
        # Convert args namespace to a dictionary before serialization
        yaml.dump(vars(args), yaml_file, default_flow_style=False)


def load_config_from_yaml(filepath):
    with open(filepath, 'r') as yaml_file:
        config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return config_dict


class RLCSVLogHandler(Handler):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.fieldnames = ['episode', 'time_step', 'return']
        # Ensure the file is set up with headers if it's new
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            if csvfile.tell() == 0:  # Write header only if file is empty
                writer.writeheader()

    def emit(self, record):
        # Assume record.message is a list of (time_step, return) tuples
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            for time_step, ret in record.message:
                writer.writerow({'episode': record.episode, 'time_step': time_step, 'return': ret})


class ListLogFormatter(logging.Formatter):
    def format(self, record):
        # This custom format expects the 'episode' to be passed in extra
        record.episode = record.args['episode']
        record.message = record.args['returns']
        return super().format(record)


def setup_rl_logger(path):
    logger = logging.getLogger('RLLogger')
    logger.setLevel(logging.INFO)  # Typically, we log info-level in such cases
    handler = RLCSVLogHandler(f'{path}/rl_returns.csv')
    handler.setFormatter(ListLogFormatter())
    logger.addHandler(handler)
    return logger

