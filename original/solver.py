import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import json
from tqdm import tqdm, trange
from layers.summarizer import Summarizer
from layers.discriminator import Discriminator
from utils import TensorboardWriter

def save_checkpoint(epoch, model, optimizers, path):
    os = []
    for optimizer in optimizers:
        os.append(optimizer.state_dict())
    state = {
        'Epoch': epoch,
        'State_dict': model.state_dict(),
        'optimizers': os
    }
    torch.save(state, path)


def load_checkpoint(model, optimizers, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['State_dict'])
    for i, optimizer in enumerate(optimizers):
        optimizer.load_state_dict(checkpoint['optimizers'][i])

class Solver(object):
    def __init__(self, config=None, train_loader=None, test_loader=None):
        """Class that Builds, Trains and Evaluates SUM-GAN model"""
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

    def build(self):
        torch.cuda.empty_cache()
        # 내가 추가한 코드 AMP
        self.scalar = torch.cuda.amp.GradScaler()

        # Build Modules
        self.linear_compress = nn.Linear(
            self.config.input_size,
            self.config.hidden_size).cuda()
        self.summarizer = Summarizer(
            input_size=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers).cuda()
        self.discriminator = Discriminator(
            input_size=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers).cuda()
        self.model = nn.ModuleList([
            self.linear_compress, self.summarizer, self.discriminator])

        if self.config.mode == 'train':
            # Build Optimizers
            self.s_e_optimizer = optim.Adam(
                list(self.summarizer.s_lstm.parameters())
                + list(self.summarizer.vae.e_lstm.parameters())
                + list(self.linear_compress.parameters()),
                lr=self.config.lr)
            self.d_optimizer = optim.Adam(
                list(self.summarizer.vae.d_lstm.parameters())
                + list(self.linear_compress.parameters()),
                lr=self.config.lr)
            self.c_optimizer = optim.Adam(
                list(self.discriminator.parameters())
                + list(self.linear_compress.parameters()),
                lr=self.config.discriminator_lr)

            # 저장된 모델 불러와서 이어서 학습
            if self.config.pre_trained:
                self.model.load_state_dict(torch.load(self.config.model_dir))

            # Overview Parameters
            print(self.model)
            print('Model Parameters')
            for name, param in self.model.named_parameters():
                print('\t' + name + '\t', list(param.size()))
                print('\t train: ' + '\t', param.requires_grad)

            self.model.train()
            self.writer = TensorboardWriter(self.config.log_dir)

    @staticmethod
    def freeze_model(module):
        for p in module.parameters():
            p.requires_grad = False

    def reconstruction_loss(self, h_origin, h_fake):
        """L2 loss between original-regenerated features at cLSTM's last hidden layer"""

        return torch.norm(h_origin - h_fake, p=2)

    # 줄여야하는 KL Divergence
    def prior_loss(self, mu, log_variance):
        """KL( q(e|x) || N(0,1) )"""
        return 0.5 * torch.sum(-1 + log_variance.exp() + mu.pow(2) - log_variance)

    # 이게 왜 Summary-Length랑 관련이 있는지 이해가 안되네
    def sparsity_loss(self, scores):
        """Summary-Length Regularization"""
        return torch.abs(torch.mean(scores) - self.config.summary_rate)

    def gan_loss(self, original_prob, fake_prob, uniform_prob):
        """Typical GAN loss + Classify uniformly scored features"""

        gan_loss = torch.mean(torch.log(original_prob) + torch.log(1 - fake_prob)
                              + torch.log(1 - uniform_prob))  # Discriminate uniform score

        return gan_loss

    def train(self):
        step = 0
        # 이어서 학습하려면 epochs를 이어서 계산
        if self.config.pre_trained:
            md = self.config.model_dir
            n_epochs = int(md[md.find('epoch-') + 6:md.find('pkl') - 1])
            epochs = tqdm(range(n_epochs, n_epochs + self.config.n_epochs), desc='Epoch', ncols=80)
        else:
            n_epochs = self.config.n_epochs
            epochs = tqdm(range(n_epochs), desc='Epoch', ncols=80)
        print(n_epochs)

        # tqdm 설정

        for epoch_i in epochs:
            s_e_loss_history = []
            d_loss_history = []
            c_loss_history = []
            for batch_i, image_features in enumerate(tqdm(
                    self.train_loader, desc='Batch', ncols=80, leave=False)):

                # 내가 수정한 코드 / 이미지 장수로 건너뛰기 일단 제한 없이
                tqdm.write(f'\n------{batch_i}th Batch: {image_features.size(1)} size')
                # if image_features.size(1) > 15000:
                # continue

                # [batch_size=1, seq_len, 2048]
                # [seq_len, 2048]
                image_features = image_features.view(-1, 1024)

                # [seq_len, 2048]
                image_features_ = Variable(image_features).cuda()

                # 내가 추가한 코드 / amp
                with torch.cuda.amp.autocast():
                    # ---- Train sLSTM, eLSTM ----#
                    if self.config.verbose:
                        tqdm.write('\nTraining sLSTM and eLSTM...')

                    # [seq_len, 1, hidden_size]
                    # LSTM에 넣으려고 3-d tensor로 만듬
                    original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)

                    # sLSTM 부터 dSLTM 까지가 Summarizer
                    scores, h_mu, h_log_variance, generated_features = self.summarizer(
                        original_features)
                    _, _, _, uniform_features = self.summarizer(
                        original_features, uniform=True)

                    h_origin, original_prob = self.discriminator(original_features)
                    h_fake, fake_prob = self.discriminator(generated_features)
                    h_uniform, uniform_prob = self.discriminator(uniform_features)

                    tqdm.write(
                        f'original_p: {original_prob.data:.3f}, fake_p: {fake_prob.data:.3f}, uniform_p: {uniform_prob.data:.3f}')

                    reconstruction_loss = self.reconstruction_loss(h_origin, h_fake)
                    prior_loss = self.prior_loss(h_mu, h_log_variance)
                    sparsity_loss = self.sparsity_loss(scores)
                    s_e_loss = reconstruction_loss + prior_loss + sparsity_loss
                    torch.cuda.empty_cache()
                tqdm.write(
                    f'recon loss {reconstruction_loss.data:.3f}, prior loss: {prior_loss.data:.3f}, sparsity loss: {sparsity_loss.data:.3f}')

                # pytorch는 backpropagation 과정 중 값을 축적하기 떄문이라는데 잘 모르겠음
                self.s_e_optimizer.zero_grad()

                # 내가 추가한 코드 / amp
                self.scalar.scale(s_e_loss).backward()  # retain_graph=True)

                # 논문과 다른 점 / Gradient cliping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.scalar.step(self.s_e_optimizer)
                self.scalar.update()
                torch.cuda.empty_cache()
                s_e_loss_history.append(s_e_loss.data)
                torch.cuda.empty_cache()

                # 내가 추가한 코드 / amp
                with torch.cuda.amp.autocast():

                    # ---- Train dLSTM ----#
                    if self.config.verbose:
                        tqdm.write('Training dLSTM...')

                    # [seq_len, 1, hidden_size]
                    original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)

                    scores, h_mu, h_log_variance, generated_features = self.summarizer(
                        original_features)

                    _, _, _, uniform_features = self.summarizer(
                        original_features, uniform=True)

                    h_origin, original_prob = self.discriminator(original_features)
                    h_fake, fake_prob = self.discriminator(generated_features)
                    h_uniform, uniform_prob = self.discriminator(uniform_features)

                    tqdm.write(
                        f'original_p: {original_prob.data:.3f}, fake_p: {fake_prob.data:.3f}, uniform_p: {uniform_prob.data:.3f}')

                    reconstruction_loss = self.reconstruction_loss(h_origin, h_fake)
                    gan_loss = self.gan_loss(original_prob, fake_prob, uniform_prob)
                    d_loss = reconstruction_loss + gan_loss

                self.d_optimizer.zero_grad()
                self.scalar.scale(d_loss).backward()  # retain_graph=True)

                # Gradient cliping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.scalar.step(self.d_optimizer)
                self.scalar.update()
                torch.cuda.empty_cache()
                d_loss_history.append(d_loss.data)
                torch.cuda.empty_cache()

                # ---- Train cLSTM ----#
                if batch_i > self.config.discriminator_slow_start:
                    # Maximization
                    # 내가 추가한 코드 / amp
                    with torch.cuda.amp.autocast():
                        if self.config.verbose:
                            tqdm.write('Training cLSTM...')

                        # [seq_len, 1, hidden_size]
                        original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)

                        scores, h_mu, h_log_variance, generated_features = self.summarizer(
                            original_features)
                        _, _, _, uniform_features = self.summarizer(
                            original_features, uniform=True)

                        h_origin, original_prob = self.discriminator(original_features)
                        h_fake, fake_prob = self.discriminator(generated_features)
                        h_uniform, uniform_prob = self.discriminator(uniform_features)
                        c_loss = -1 * self.gan_loss(original_prob, fake_prob, uniform_prob)
                        tqdm.write(
                            f'original_p: {original_prob.data:.3f}, fake_p: {fake_prob.data:.3f}, uniform_p: {uniform_prob.data:.3f}')

                    self.c_optimizer.zero_grad()
                    self.scalar.scale(c_loss).backward()

                    # Gradient cliping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)

                    self.scalar.step(self.c_optimizer)
                    self.scalar.update()
                    torch.cuda.empty_cache()
                    c_loss_history.append(c_loss.data)

                if self.config.verbose:
                    tqdm.write('Plotting...')

                self.writer.update_loss(reconstruction_loss.data, step, 'recon_loss')
                self.writer.update_loss(prior_loss.data, step, 'prior_loss')
                self.writer.update_loss(sparsity_loss.data, step, 'sparsity_loss')
                self.writer.update_loss(gan_loss.data, step, 'gan_loss')

                self.writer.update_loss(original_prob.data, step, 'original_prob')
                self.writer.update_loss(fake_prob.data, step, 'fake_prob')
                self.writer.update_loss(uniform_prob.data, step, 'uniform_prob')

                step += 1

            s_e_loss = torch.stack(s_e_loss_history).mean()
            d_loss = torch.stack(d_loss_history).mean()
            c_loss = torch.stack(c_loss_history).mean()

            # Plot
            if self.config.verbose:
                tqdm.write('Plotting...')
            self.writer.update_loss(s_e_loss, epoch_i, 's_e_loss_epoch')
            self.writer.update_loss(d_loss, epoch_i, 'd_loss_epoch')
            self.writer.update_loss(c_loss, epoch_i, 'c_loss_epoch')

            # Save parameters at checkpoint every five epoche
            if (epoch_i + 1) % 5 == 0:
                ckpt_path = str(self.config.save_dir) + f'_epoch-{epoch_i + 1}.pkl'
                tqdm.write(f'Save parameters at {ckpt_path}')
                torch.save(self.model.state_dict(), ckpt_path)

    def evaluate(self, model_path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['State_dict'])
        # self.model.load_state_dict(torch.load(model_path))

        self.model.eval()

        out_dict = {}

        for video_tensor, video_name in tqdm(
                self.test_loader, desc='Evaluate', ncols=80, leave=False):
            # [seq_len, batch=1, 2048]
            video_tensor = video_tensor.view(-1, self.config.input_size)

            video_feature = Variable(video_tensor, volatile=True).cuda()

            # [seq_len, 1, hidden_size]
            video_feature = self.linear_compress(video_feature.detach()).unsqueeze(1)

            # [seq_len]
            scores = self.summarizer.s_lstm(video_feature).squeeze(1)

            scores = np.array(scores.cpu().data).tolist()

            out_dict[video_name] = scores

            import os
            score_save_path = f'result.json'
            with open(score_save_path, 'w') as f:
                json.dump(out_dict, f)
