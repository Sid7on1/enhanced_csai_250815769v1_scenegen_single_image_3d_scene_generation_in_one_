import logging
import os
import yaml
from typing import Dict, List, Optional
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from torch import nn
import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class Config:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> Dict:
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def get_config(self) -> Dict:
        return self.config

class ModelConfig:
    def __init__(self, config: Config):
        self.config = config.get_config()
        self.model_name = self.config["model_name"]
        self.model_config = self.load_model_config()

    def load_model_config(self) -> Dict:
        model_config = AutoConfig.from_pretrained(self.model_name)
        return model_config

    def get_model_config(self) -> Dict:
        return self.model_config

class TrainingConfig:
    def __init__(self, config: Config):
        self.config = config.get_config()
        self.train_config = self.load_train_config()

    def load_train_config(self) -> Dict:
        train_config = TrainingArguments(
            output_dir=self.config["output_dir"],
            num_train_epochs=self.config["num_train_epochs"],
            per_device_train_batch_size=self.config["per_device_train_batch_size"],
            per_device_eval_batch_size=self.config["per_device_eval_batch_size"],
            evaluation_strategy=self.config["evaluation_strategy"],
            learning_rate=self.config["learning_rate"],
            save_steps=self.config["save_steps"],
            load_best_model_at_end=self.config["load_best_model_at_end"],
            metric_for_best_model=self.config["metric_for_best_model"],
            greater_is_better=self.config["greater_is_better"],
            save_total_limit=self.config["save_total_limit"],
            no_cuda=self.config["no_cuda"],
            seed=self.config["seed"],
            fp16=self.config["fp16"],
        )
        return train_config

    def get_train_config(self) -> Dict:
        return self.train_config

class DataConfig:
    def __init__(self, config: Config):
        self.config = config.get_config()
        self.data_config = self.load_data_config()

    def load_data_config(self) -> Dict:
        data_config = {
            "train_file": self.config["train_file"],
            "validation_file": self.config["validation_file"],
            "test_file": self.config["test_file"],
            "max_length": self.config["max_length"],
            "batch_size": self.config["batch_size"],
        }
        return data_config

    def get_data_config(self) -> Dict:
        return self.data_config

class Model:
    def __init__(self, config: Config):
        self.config = config.get_config()
        self.model_name = self.config["model_name"]
        self.model = self.load_model()

    def load_model(self) -> nn.Module:
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        return model

    def get_model(self) -> nn.Module:
        return self.model

class Trainer:
    def __init__(self, config: Config):
        self.config = config.get_config()
        self.train_config = TrainingConfig(config).get_train_config()
        self.data_config = DataConfig(config).get_data_config()
        self.model = Model(config).get_model()

    def train(self):
        trainer = Trainer(
            model=self.model,
            args=self.train_config,
            train_dataset=self.load_train_dataset(),
            eval_dataset=self.load_eval_dataset(),
        )
        trainer.train()
        return trainer

    def load_train_dataset(self) -> pd.DataFrame:
        train_df = pd.read_csv(self.data_config["train_file"])
        return train_df

    def load_eval_dataset(self) -> pd.DataFrame:
        eval_df = pd.read_csv(self.data_config["validation_file"])
        return eval_df

def load_config(config_path: str = "config.yaml") -> Config:
    config = Config(config_path)
    return config

def get_model_config(config: Config) -> ModelConfig:
    model_config = ModelConfig(config)
    return model_config

def get_train_config(config: Config) -> TrainingConfig:
    train_config = TrainingConfig(config)
    return train_config

def get_data_config(config: Config) -> DataConfig:
    data_config = DataConfig(config)
    return data_config

def get_trainer(config: Config) -> Trainer:
    trainer = Trainer(config)
    return trainer

def main():
    config_path = "config.yaml"
    config = load_config(config_path)
    model_config = get_model_config(config)
    train_config = get_train_config(config)
    data_config = get_data_config(config)
    trainer = get_trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()