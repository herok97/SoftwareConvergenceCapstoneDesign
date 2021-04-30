from configs import Config
from solver import Solver
from data_loader import get_loader


if __name__ == '__main__':
    model_path = ''
    config = Config
    test_loader = get_loader(config.video_root_dir, config.mode)
    solver = Solver(config, test_loader)
    solver.build()
    solver.evaluate(model_path)
