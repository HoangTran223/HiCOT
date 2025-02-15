import numpy as np
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from utils import static_utils
import logging
import os
import scipy


class BasicTrainer:
    def __init__(self, model, model_name='HiCOT', epochs=200, learning_rate=0.002, batch_size=200, lr_scheduler=None, lr_step_size=125, log_interval=5, 
                     device='cuda'):
        self.model = model
        self.model_name = model_name
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.lr_step_size = lr_step_size
        self.log_interval = log_interval

        self.device = device
        self.logger = logging.getLogger('main')


    def make_adam_optimizer(self,):
        args_dict = {
            'params': self.model.parameters(),
            'lr': self.learning_rate,
        }

        optimizer = torch.optim.Adam(**args_dict)
        return optimizer
    
    def make_lr_scheduler(self, optimizer):
        if self.lr_scheduler == "StepLR":
            lr_scheduler = StepLR(
                optimizer, step_size=self.lr_step_size, gamma=0.5, verbose=False)
        else:
            raise NotImplementedError(self.lr_scheduler)
        return lr_scheduler

    def fit_transform(self, dataset_handler, num_top_words=15, verbose=False):
        self.train(dataset_handler, verbose)
        top_words = self.export_top_words(dataset_handler.vocab, num_top_words)
    
        if self.model_name == 'HiCOT':
            train_theta = self.test(dataset_handler.train_data, dataset_handler.doc_embeddings)
        else:
            print(f"Wrong model")

        return top_words, train_theta

    def train(self, dataset_handler, verbose=False):
        adam_optimizer = self.make_adam_optimizer()

        if self.lr_scheduler:
            print("===>using lr_scheduler")
            self.logger.info("===>using lr_scheduler")
            lr_scheduler = self.make_lr_scheduler(adam_optimizer)

        data_size = len(dataset_handler.train_dataloader.dataset)

        for epoch_id, epoch in enumerate(tqdm(range(1, self.epochs + 1))):
            self.model.train()
            loss_rst_dict = defaultdict(float)

            for batch_id, batch in enumerate(dataset_handler.train_dataloader): 
                if self.model_name == 'HiCOT':
                    *inputs, indices, doc_embeddings = batch
                else:
                    print(f"Wrong model")

                batch_data = inputs
                rst_dict = self.model(indices, batch_data, epoch_id=epoch, doc_embeddings=doc_embeddings)
                batch_loss = rst_dict['loss']

                batch_loss.backward()
                adam_optimizer.step()
                adam_optimizer.zero_grad()

                for key in rst_dict:
                    try:
                        loss_rst_dict[key] += rst_dict[key] * \
                            len(batch_data['data'])
                    except:
                        loss_rst_dict[key] += rst_dict[key] * len(batch_data)

            if self.lr_scheduler:
                lr_scheduler.step()

            if verbose and epoch % self.log_interval == 0:
                output_log = f'Epoch: {epoch:03d}'
                for key in loss_rst_dict:
                    output_log += f' {key}: {loss_rst_dict[key] / data_size :.3f}'

                self.logger.info(output_log)

    def test(self, input_data, train_data=None):
        data_size = input_data.shape[0]
        theta = list()
        all_idx = torch.split(torch.arange(data_size), self.batch_size)

        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_input = input_data[idx]
                batch_theta = self.model.get_theta(batch_input)
                theta.extend(batch_theta.cpu().tolist())

        theta = np.asarray(theta)
        return theta


    def export_beta(self):
        beta = self.model.get_beta().detach().cpu().numpy()
        return beta

    def export_top_words(self, vocab, num_top_words=15):
        beta = self.export_beta()
        top_words = static_utils.print_topic_words(beta, vocab, num_top_words)
        return top_words

    def export_theta(self, dataset_handler):
        train_theta = self.test(dataset_handler.train_data)
        test_theta = self.test(dataset_handler.test_data)
        return train_theta, test_theta

    def save_beta(self, dir_path):
        beta = self.export_beta()
        np.save(os.path.join(dir_path, 'beta.npy'), beta)
        return beta

    def save_top_words(self, vocab, num_top_words, dir_path):
        top_words = self.export_top_words(vocab, num_top_words)
        with open(os.path.join(dir_path, f'top_words_{num_top_words}.txt'), 'w') as f:
            for i, words in enumerate(top_words):
                f.write(words + '\n')
        return top_words

    def save_theta(self, dataset_handler, dir_path):
        train_theta, test_theta = self.export_theta(dataset_handler)
        np.save(os.path.join(dir_path, 'train_theta.npy'), train_theta)
        np.save(os.path.join(dir_path, 'test_theta.npy'), test_theta)
        
        train_argmax_theta = np.argmax(train_theta, axis=1)
        test_argmax_theta = np.argmax(test_theta, axis=1)
        np.save(os.path.join(dir_path, 'train_argmax_theta.npy'), train_argmax_theta)
        np.save(os.path.join(dir_path, 'test_argmax_theta.npy'), test_argmax_theta)
        return train_theta, test_theta

    def save_embeddings(self, dir_path):
        if hasattr(self.model, 'word_embeddings'):
            word_embeddings = self.model.word_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'word_embeddings.npy'), word_embeddings)
            self.logger.info(f'word_embeddings size: {word_embeddings.shape}')

        if hasattr(self.model, 'topic_embeddings'):
            topic_embeddings = self.model.topic_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'topic_embeddings.npy'),
                    topic_embeddings)
            self.logger.info(
                f'topic_embeddings size: {topic_embeddings.shape}')

            topic_dist = scipy.spatial.distance.cdist(topic_embeddings, topic_embeddings)
            np.save(os.path.join(dir_path, 'topic_dist.npy'), topic_dist)

        return word_embeddings, topic_embeddings
