import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import json
from tqdm import tqdm, trange
from utils import TensorboardWriter
from network import AE


def save_checkpoint(epoch, model, optimizer, path):
    state = {
        'Epoch': epoch,
        'State_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, path)


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['State_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


class Solver(object):
    def __init__(self, config=None, train_loader=None, test_loader=None):
        """Class that Builds, Trains and Evaluates SUM-GAN model"""
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

    def build(self):
        self.linear_compress = nn.Linear(
            self.config.input_size,
            self.config.hidden_size).cuda()

        self.AE = AE(input_size=self.config.hidden_size,
                     hidden_size=self.config.hidden_size,
                     num_layers=self.config.num_layers).cuda()
        self.model = nn.ModuleList([
            self.linear_compress, self.AE])
        # Build Modules
        if self.config.mode == 'train':
            # Build Optimizers
            self.ae_optimizer = optim.Adam(
                list(self.AE.e_lstm.parameters())
                + list(self.AE.d_lstm.parameters())
                + list(self.linear_compress.parameters()),
                lr=self.config.lr)

            # 저장된 모델 불러와서 이어서 학습
            if self.config.pre_trained:
                print('모델 가중치 가져오기')
                load_checkpoint(self.model, self.ae_optimizer, self.config.model_dir)

            self.AE.train()
            self.writer = TensorboardWriter(self.config.log_dir)

    def train(self):
        step = 0
        # 이어서 학습하려면 epochs를 이어서 계산
        if self.config.pre_trained:
            md = self.config.model_dir
            n_epochs = int(md[md.find('epoch-') + 6:md.find('pkl') - 1])
            print(f'{n_epochs} epoch 부터 학습 시작')
            epochs = tqdm(range(n_epochs, n_epochs + self.config.n_epochs), desc='Epoch', ncols=80)
        else:
            n_epochs = self.config.n_epochs
            epochs = tqdm(range(n_epochs), desc='Epoch', ncols=80)

        mse_loss = nn.MSELoss()

        # tqdm 설정
        for epoch_i in epochs:
            loss_history = []
            if self.config.verbose:
                tqdm.write('\nTraining')
            for batch_i, image_features in enumerate(tqdm(
                    self.train_loader, desc='Batch', ncols=80, leave=False)):
                # 내가 수정한 코드 / 이미지 장수로 건너뛰기 일단 제한 없이
                tqdm.write(f'\n------{batch_i}th Batch: {image_features.size(1)} size')

                image_features = image_features.view(-1, 1024)
                image_features_ = Variable(image_features).cuda()

                # ---- Train sLSTM, eLSTM ----#

                original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)
                decoded_features = self.AE(original_features)

                loss = mse_loss(original_features, decoded_features)
                tqdm.write(f'loss: {loss}')
                self.ae_optimizer.zero_grad()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.AE.parameters(), self.config.clip)
                self.ae_optimizer.step()
                loss_history.append(loss.data)

                self.writer.update_loss(loss.data, step, 'reconstruct_loss')
                step += 1

            epoch_loss = torch.stack(loss_history).mean()

            # Plot
            if self.config.verbose:
                tqdm.write('Plotting...')
            self.writer.update_loss(epoch_loss, epoch_i, 'loss_epochs')

            # Save parameters at checkpoint every five epoch
            if (epoch_i + 1) % 10 == 0:
                ckpt_path = str(self.config.save_dir) + f'_epoch-{epoch_i + 1}.pkl'
                tqdm.write(f'Save parameters at {ckpt_path}')
                save_checkpoint(epoch_i+1, self.model,
                                self.ae_optimizer, ckpt_path)

    def evaluate(self, model_path):
        self.AE.load_state_dict(torch.load(model_path))

        self.AE.eval()
        mse_loss = nn.MSELoss()
        out_dict = {}
        print(self.test_loader)
        losses = []
        for video_tensor, video_name in tqdm(self.test_loader, desc='Evaluate', ncols=80, leave=False):
            # [seq_len, batch=1, 2048]
            video_tensor = video_tensor.view(-1, self.config.input_size)

            video_feature = Variable(video_tensor, volatile=True).cuda()

            # [seq_len, 1, hidden_size]
            original_feature = self.linear_compress(video_feature.detach()).unsqueeze(1)
            decoded_feature = self.AE(original_feature)
            loss = mse_loss(original_feature, decoded_feature)
            losses.append(loss)
        mean_loss = np.mean(losses).squeeze()
        tqdm.write(f'loss = {mean_loss.data}.')
