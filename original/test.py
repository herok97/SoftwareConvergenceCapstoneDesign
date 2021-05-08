from configs import Config
from solver import Solver
from data_loader import get_loader


if __name__ == '__main__':
    config = Config('test')
    print(config.video_root_dir, config.mode)
    test_loader = get_loader(config.video_root_dir, config.mode)
    solver = Solver(config=config, test_loader=test_loader)
    solver.build()
    solver.evaluate(config.model_path)
