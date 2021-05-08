
from pathlib import Path

dataset_dir = Path('C:/Users/01079/video_summarization_data\h5_googlenet').resolve()
video_list = ['OVP', 'SumMe', 'TvSum']
save_dir = Path('./results')
score_dir = Path('/save')


class Config:
    def __init__(self, mode='train'):
        self.input_size = 1024
        self.hidden_size = 512
        self.num_layers = 2
        self.video_type = 'TvSum'
        self.save_dir = save_dir.joinpath(self.video_type)
        self.log_dir = save_dir
        self.epoch = 49
        self.ckpt_path = save_dir.joinpath(f'epoch-{self.epoch}.pkl')
        self.score_dir = score_dir
        self.mode = mode
        self.video_root_dir = dataset_dir.joinpath(self.video_type, mode)
        self.verbose = 'true'
        self.preprocessed = 'true'
        self.summary_rate = 0.3
        self.n_epochs = 1
        self.clip = 5.0
        self.lr = 1e-4
        self.discriminator_lr = 1e-5
        self.discriminator_slow_start = 0
        self.pre_trained = True
        self.model_dir = "./results/SumMe_epoch-30.pkl"
        self.model_path = "./results/TvSum_epoch-50.pkl" # 테스트할 모델 경로
