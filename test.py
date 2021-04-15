from configs import get_config
from solver import Solver
from data_loader import get_loader


if __name__ == '__main__':
    model_path = ''
    config = get_config(mode='test')
    test_loader = get_loader(config.video_root_dir, config.mode)
    solver = Solver(config, test_loader)
    solver.build()
    solver.evaluate(model_path)
