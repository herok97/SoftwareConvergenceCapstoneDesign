
from pathlib import Path

dataset_dir = Path('/content/drive/MyDrive/Capstone/googlenet/VS_data_googlenet').resolve()
video_list = ['OVP', 'SumMe', 'TvSum']
save_dir = Path('/content/drive/MyDrive/Capstone/googlenet/result')
score_dir = Path('/content/drive/MyDrive/Capstone/googlenet/save')


class Config:
    video_type = 'TvSum'
    save_dir = save_dir.joinpath(video_type)
    log_dir = save_dir
    epoch = 49
    ckpt_path = save_dir.joinpath(f'epoch-{epoch}.pkl')
    score_dir = score_dir
    mode = 'train'
    video_root_dir = dataset_dir.joinpath(video_type, mode)
    verbose = 'true'
    preprocessed = 'true'
    summary_rate = 0.3
    n_epochs = 50
    clip = 5.0
    lr = 1e-3
    discriminator_lr = 1e-4
    discriminator_slow_start = 5
    pre_trained = True
    model_dir = "/content/drive/MyDrive/Capstone/googlenet/result/TvSum_epoch-11.pkl"

