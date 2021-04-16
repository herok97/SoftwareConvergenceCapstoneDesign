# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pprint

project_dir = Path(__file__).resolve().parent
dataset_dir = Path('C:\VS_data/').resolve()
video_list = ['OVP', 'SumMe', 'TvSum']
save_dir = Path('./results/save/')
score_dir = Path('./results/score/')

class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.set_dataset_dir(self.video_type)

    def set_dataset_dir(self, video_type='OVP'):
        if self.preprocessed:
            # self.video_root_dir = dataset_dir.joinpath('resnet101_feature', video_type, self.mode)
            self.video_root_dir = dataset_dir.joinpath(video_type, self.mode)
        else:
            # self.video_root_dir = dataset_dir.joinpath('video_subshot', video_type, 'test')
            self.video_root_dir = dataset_dir.joinpath(video_type, self.mode)
        self.save_dir = save_dir.joinpath(video_type)
        self.log_dir = self.save_dir
        self.ckpt_path = self.save_dir.joinpath(f'epoch-{self.epoch}.pkl')
        self.score_dir = score_dir
    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--verbose', type=str, default='true')
    parser.add_argument('--preprocessed', type=str, default='True')
    parser.add_argument('--video_type', type=str, default='OVP')

    # Model
    parser.add_argument('--summary_rate', type=float, default=0.3)

    # Train
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--discriminator_lr', type=float, default=1e-4)
    parser.add_argument('--discriminator_slow_start', type=int, default=15)

    # load epoch
    parser.add_argument('--epoch', type=int, default=49)

    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)


if __name__ == '__main__':
    config = get_config()
    import ipdb
    ipdb.set_trace()
