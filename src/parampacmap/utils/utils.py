import os

DEFAULT_MODEL_DICT = {"backbone": "ANN", "layer_size": [100, 100, 100]}


def read_yaml(config: str):
    import yaml
    with open(config, "r") as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise exc
    return config_dict


def makedir(dirname: str):
    if not os.path.exists(dirname):
        try:
            os.mkdir(dirname)
        except FileExistsError:
            pass


def impute_default(config: dict, default_config: dict):
    """
    Impute the training config with a set of default values.
    """
    for key in default_config.keys():
        if key not in config:
            config[key] = default_config[key]

    return config
