from configs import get_config
from solver import Solver
from data_loader import get_loader


if __name__ == '__main__':
    config = get_config(mode='train')
    train_loader = get_loader(config.video_root_dir, config.mode)
    solver = Solver(config, train_loader)
    solver.build()
    solver.train()
