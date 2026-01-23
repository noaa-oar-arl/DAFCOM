import os

import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../config.yaml")


def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    config = load_config()
    print("project_name:", config["project"]["name"])
    print("database aod folder:", config["database"]["dynamic_inputs"]["aod"])
    print("database chem folder:", config["database"]["dynamic_inputs"]["surface_pollutants"])
