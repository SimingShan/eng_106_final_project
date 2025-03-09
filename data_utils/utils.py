import argparse
import yaml
import os

def dict2namespace(config):
    # If config is a string, assume it's a file path and load the file
    if isinstance(config, str):
        with open(config, 'r') as f:
            config = yaml.safe_load(f)

    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            # Recursively handle nested dictionaries
            value = dict2namespace(value)
        setattr(namespace, key, value)
    return namespace


def setup_logger(config):
    log_file = config.Training.log_file_path

    # Ensure the directory for the log file exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log_message(message):
        print(message)  # Print to standard output
        with open(log_file, 'a') as f:  # Append message to log file
            f.write(f"{message}\n")

    return log_message