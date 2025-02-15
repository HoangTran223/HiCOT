import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import scipy.sparse
import scipy.io
from sentence_transformers import SentenceTransformer
from . import file_utils
import os

##
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def load_contextual_embed(texts, device, model_name="all-mpnet-base-v2", show_progress_bar=True):
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(texts, show_progress_bar=show_progress_bar)
    return embeddings


class DatasetHandler(Dataset):
    def __init__(self, data, contextual_embed=None):
        self.data = data
        self.contextual_embed = None
        if contextual_embed is not None:
            assert data.shape[0] == contextual_embed.shape[0], "Data and contextual embeddings should have the same number of samples"
            self.contextual_embed = contextual_embed

    def __len__(self):
        # Update this according to your data size
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.contextual_embed is None:
            return {
                'data': self.data[idx]
            }

        return {
            'data': self.data[idx],
            'contextual_embed': self.contextual_embed[idx]
        }


class RawDatasetHandler:
    def __init__(self, docs, preprocessing, batch_size=200, device='cpu', as_tensor=False, contextual_embed=False):

        rst = preprocessing.preprocess(docs)
        self.train_data = rst['train_bow']
        self.train_texts = rst['train_texts']
        self.vocab = rst['vocab']

        self.vocab_size = len(self.vocab)

        if contextual_embed:
            self.train_contextual_embed = load_contextual_embed(
                self.train_texts, device)
            self.contextual_embed_size = self.train_contextual_embed.shape[1]

        if as_tensor:
            if contextual_embed:
                self.train_data = np.concatenate(
                    (self.train_data, self.train_contextual_embed), axis=1)

            self.train_data = torch.from_numpy(
                self.train_data).float().to(device)
            self.train_dataloader = DataLoader(
                self.train_data, batch_size=batch_size, shuffle=True)


class BasicDatasetHandler:
    def __init__(self, dataset_dir, batch_size=200, read_labels=False, device='gpu', 
                    as_tensor=False, contextual_embed=False, doc2vec_size=384, args=None):
        self.args = args
        self.doc2vec_size = doc2vec_size    
        self.load_data(dataset_dir, read_labels)
        self.vocab_size = len(self.vocab)

        print("===>train_size: ", self.train_bow.shape[0])
        print("===>test_size: ", self.test_bow.shape[0])
        print("===>vocab_size: ", self.vocab_size)
        print("===>average length: {:.3f}".format(
            self.train_bow.sum(1).sum() / self.train_bow.shape[0]))
        
        ##
        doc2vec_dir = os.path.join(dataset_dir, 'doc2vec')
        # os.makedirs(doc2vec_dir, exist_ok=True)
        if contextual_embed == False:
            doc2vec_train_filepath = os.path.join(doc2vec_dir, f'doc_embeddings_384_.npz')
            if os.path.isfile(doc2vec_train_filepath):
                print("===> Loading train doc_embeddings...")
                self.train_doc_embeddings = np.load(doc2vec_train_filepath)['arr_0']
            else:
                raise FileNotFoundError(f"File {doc2vec_train_filepath} not found.")
            
            doc2vec_test_filepath = os.path.join(doc2vec_dir, f'doc_embeddings_test_384_.npz')
            if os.path.isfile(doc2vec_test_filepath):
                print("===> Loading test doc_embeddings...")
                self.test_doc_embeddings = np.load(doc2vec_test_filepath)['arr_0']
            else:
                raise FileNotFoundError(f"File {doc2vec_test_filepath} not found.")

        if contextual_embed:
            if os.path.isfile(os.path.join(dataset_dir, 'with_bert', 'train_bert.npz')):
                self.train_contextual_embed = np.load(os.path.join(
                    dataset_dir, 'with_bert', 'train_bert.npz'))['arr_0']
            else:
                self.train_contextual_embed = load_contextual_embed(
                    self.train_texts, device)

            if os.path.isfile(os.path.join(dataset_dir, 'with_bert', 'test_bert.npz')):
                self.test_contextual_embed = np.load(os.path.join(
                    dataset_dir, 'with_bert', 'test_bert.npz'))['arr_0']
            else:
                self.test_contextual_embed = load_contextual_embed(
                    self.test_texts, device)

            self.contextual_embed_size = self.train_contextual_embed.shape[1]

        if as_tensor:
            self.train_data = self.train_bow
            self.test_data = self.test_bow

            self.train_data = torch.from_numpy(self.train_data).to(device)
            self.test_data = torch.from_numpy(self.test_data).to(device)

            self.train_indices = torch.arange(len(self.train_data)).to(device)
            self.test_indices = torch.arange(len(self.test_data)).to(device)

            if contextual_embed:
                self.train_contextual_embed = torch.from_numpy(
                    self.train_contextual_embed).to(device)
                self.test_contextual_embed = torch.from_numpy(
                    self.test_contextual_embed).to(device)

                """train_dataset = DatasetHandler(
                    self.train_data, self.train_contextual_embed)
                test_dataset = DatasetHandler(
                    self.test_data, self.test_contextual_embed)"""
                
                train_dataset = TensorDataset(
                    self.train_data, self.train_contextual_embed, self.train_indices)
                test_dataset = TensorDataset(
                    self.test_data, self.test_contextual_embed, self.test_indices)

                self.train_dataloader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True)
                self.test_dataloader = DataLoader(
                    test_dataset, batch_size=batch_size, shuffle=False)

            else:
                """train_dataset = DatasetHandler(self.train_data)
                test_dataset = DatasetHandler(self.test_data)"""

                if args.model == "HiCOT":
                    train_dataset = TensorDataset(self.train_data, self.train_indices, 
                                            torch.tensor(self.train_doc_embeddings, dtype=torch.float))
                    test_dataset = TensorDataset(self.test_data, self.test_indices, 
                                            torch.tensor(self.test_doc_embeddings, dtype=torch.float))
                
                else:
                    train_dataset = TensorDataset(self.train_data, self.train_indices)
                    test_dataset = TensorDataset(self.test_data, self.test_indices)

                self.train_dataloader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True)
                self.test_dataloader = DataLoader(
                    test_dataset, batch_size=batch_size, shuffle=False)

    def load_data(self, path, read_labels):

        self.train_bow = scipy.sparse.load_npz(
            f'{path}/train_bow.npz').toarray().astype('float32')
        self.test_bow = scipy.sparse.load_npz(
            f'{path}/test_bow.npz').toarray().astype('float32')
        self.pretrained_WE = scipy.sparse.load_npz(
            f'{path}/word_embeddings.npz').toarray().astype('float32')

        self.train_texts = file_utils.read_text(f'{path}/train_texts.txt')
        self.test_texts = file_utils.read_text(f'{path}/test_texts.txt')

        if read_labels:
            self.train_labels = np.loadtxt(
                f'{path}/train_labels.txt', dtype=int)
            self.test_labels = np.loadtxt(f'{path}/test_labels.txt', dtype=int)

        self.vocab = file_utils.read_text(f'{path}/vocab.txt')
