import string
from collections import Counter
from typing import Dict, List, Tuple, Union, Callable
from itertools import chain

import nltk
import numpy as np
import math
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path


# Замените пути до директорий и файлов! Можете использовать для локальной отладки.
# При проверке на сервере пути будут изменены
# glue_qqp_dir = 'ml-hard/1-part/data/QQP'
# glove_path = 'ml-hard/1-part/data/glove.6B.50d.txt'
glue_qqp_dir = '1-part/data/QQP'
glove_path = '1-part/data/glove.6B.50d.txt'


nltk.download('punkt_tab')

class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        return torch.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)


class KNRM(torch.nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, freeze_embeddings: bool, kernel_num: int = 21,
                 sigma: float = 0.1, exact_sigma: float = 0.001,
                 out_layers: List[int] = [10, 5]):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=freeze_embeddings,
            padding_idx=0
        )

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()

        self.mlp = self._get_mlp()

        self.out_activation = torch.nn.Sigmoid()

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        """
        Создание списка из kernel_num гауссовских ядер.
        Генерируются kernel_num значений mu от -1 до 1.
        для mu = 5  [-.75, -.25, .25, .75, 1].
        для mu = 11 [-.9 -.7 -.5 -.3 -.1 .1 .3 .5 .7 .9 1].
        """
        def generate_symmetric_sequence(length):
            step = 1 / (length - 1)
            first_half = np.round([step + i * 2 * step for i in range(length // 2)], 2)
            symmetric_sequence = np.concatenate((-np.array(first_half[::-1]), first_half, [1]))
            for value in symmetric_sequence:
                yield value

        kernels = torch.nn.ModuleList()
        for mu in generate_symmetric_sequence(self.kernel_num):
            sigma = self.exact_sigma if mu == 1 else self.sigma
            kernels.append(GaussianKernel(mu, sigma))
        return kernels

    def _get_mlp(self) -> torch.nn.Sequential:
        layers = []
        prev_layer_size = self.kernel_num
        if not self.out_layers:
            return torch.nn.Sequential(torch.nn.Linear(prev_layer_size, 1))
        for layer_size in self.out_layers:
            layers.append(torch.nn.Linear(prev_layer_size, layer_size))
            layers.append(torch.nn.ReLU())
            prev_layer_size = layer_size
        layers.append(torch.nn.Linear(prev_layer_size, 1))
        return torch.nn.Sequential(*layers)

    def forward(self, input_1: Dict[str, torch.Tensor], input_2: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)

        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)
        return out

    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:
        # shape = [B, L, D]
        embed_query = self.embeddings(query.long())
        # shape = [B, R, D]
        embed_doc = self.embeddings(doc.long())

        # shape = [B, L, R]
        matching_matrix = torch.einsum(    
            'bld,brd->blr',
            F.normalize(embed_query, p=2, dim=-1),
            F.normalize(embed_doc, p=2, dim=-1)
        )
        return matching_matrix

    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        # shape = [Batch, Left], [Batch, Right]
        query, doc = inputs['query'], inputs['document']

        # shape = [Batch, Left, Right]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape = [Batch, Kernels]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape = [Batch]
        out = self.mlp(kernels_out)
        return out


class RankingDataset(torch.utils.data.Dataset):
    def __init__(self, index_pairs_or_triplets: List[List[Union[str, float]]],
                 idx_to_text_mapping: Dict[str, str], vocab: Dict[str, int], oov_val: int,
                 preproc_func: Callable, max_len: int = 30):
        self.index_pairs_or_triplets = index_pairs_or_triplets
        self.idx_to_text_mapping = idx_to_text_mapping
        self.vocab = vocab
        self.oov_val = oov_val
        self.preproc_func = preproc_func
        self.max_len = max_len

    def __len__(self):
        return len(self.index_pairs_or_triplets)

    def _tokenized_text_to_index(self, tokenized_text: List[str]) -> List[int]:
        return [self.vocab.get(token, self.oov_val) for token in tokenized_text]

    def _convert_text_idx_to_token_idxs(self, idx: int) -> List[int]:
        text = self.idx_to_text_mapping[idx]
        tokenized_text = self.preproc_func(text)
        return self._tokenized_text_to_index(
            tokenized_text if len(tokenized_text) < self.max_len else tokenized_text[:self.max_len]
            )
         
    def __getitem__(self, idx: int):
        raise NotImplementedError()


class TrainTripletsDataset(RankingDataset):
    def __getitem__(self, idx):
        cur_row = self.index_pairs_or_triplets[idx]
        left_idxs = self._convert_text_idx_to_token_idxs(cur_row[0])[
            :self.max_len]
        r1_idxs = self._convert_text_idx_to_token_idxs(cur_row[1])[
            :self.max_len]
        r2_idxs = self._convert_text_idx_to_token_idxs(cur_row[2])[
            :self.max_len]

        pair_1 = {'query': left_idxs, 'document': r1_idxs}
        pair_2 = {'query': left_idxs, 'document': r2_idxs}
        target = cur_row[3]
        return (pair_1, pair_2, target)                   


class ValPairsDataset(RankingDataset):
    def __getitem__(self, idx):
        id_l, id_r, t = self.index_pairs_or_triplets[idx]
        q = self._convert_text_idx_to_token_idxs(id_l)
        d = self._convert_text_idx_to_token_idxs(id_r)
        return dict(query=q, document=d), t


def collate_fn(batch_objs: List[Union[Dict[str, torch.Tensor], torch.FloatTensor]]):
    max_len_q1 = -1
    max_len_d1 = -1
    max_len_q2 = -1
    max_len_d2 = -1

    is_triplets = False
    for elem in batch_objs:
        if len(elem) == 3:
            left_elem, right_elem, label = elem
            is_triplets = True
        else:
            left_elem, label = elem

        max_len_q1 = max(len(left_elem['query']), max_len_q1)
        max_len_d1 = max(len(left_elem['document']), max_len_d1)
        if len(elem) == 3:
            max_len_q2 = max(len(right_elem['query']), max_len_q2)
            max_len_d2 = max(len(right_elem['document']), max_len_d2)

    q1s = []
    d1s = []
    q2s = []
    d2s = []
    labels = []

    for elem in batch_objs:
        if is_triplets:
            left_elem, right_elem, label = elem
        else:
            left_elem, label = elem

        pad_len1 = max_len_q1 - len(left_elem['query'])
        pad_len2 = max_len_d1 - len(left_elem['document'])
        if is_triplets:
            pad_len3 = max_len_q2 - len(right_elem['query'])
            pad_len4 = max_len_d2 - len(right_elem['document'])

        q1s.append(left_elem['query'] + [0] * pad_len1)
        d1s.append(left_elem['document'] + [0] * pad_len2)
        if is_triplets:
            q2s.append(right_elem['query'] + [0] * pad_len3)
            d2s.append(right_elem['document'] + [0] * pad_len4)
        labels.append([label])
    q1s = torch.LongTensor(q1s)
    d1s = torch.LongTensor(d1s)
    if is_triplets:
        q2s = torch.LongTensor(q2s)
        d2s = torch.LongTensor(d2s)
    labels = torch.FloatTensor(labels)

    ret_left = {'query': q1s, 'document': d1s}
    if is_triplets:
        ret_right = {'query': q2s, 'document': d2s}
        return ret_left, ret_right, labels
    else:
        return ret_left, labels


class Solution:
    def __init__(self, glue_qqp_dir: str, glove_vectors_path: str,
                 min_token_occurancies: int = 1,
                 random_seed: int = 0,
                 emb_rand_uni_bound: float = 0.2,
                 freeze_knrm_embeddings: bool = True,
                 knrm_kernel_num: int = 21,
                 knrm_out_mlp: List[int] = [],
                 dataloader_bs: int = 1024,
                 train_lr: float = 0.001,
                 change_train_loader_ep: int = 10
                 ):
        self.glue_qqp_dir = glue_qqp_dir
        self.glove_vectors_path = glove_vectors_path

        self.glue_train_df = self.get_glue_df('train')
        self.glue_dev_df = self.get_glue_df('dev')
        self.dev_pairs_for_ndcg = self.create_val_pairs(self.glue_dev_df)
        self.min_token_occurancies = min_token_occurancies
        self.all_tokens = self.get_all_tokens(
            [self.glue_train_df, self.glue_dev_df], self.min_token_occurancies)

        self.random_seed = random_seed
        self.emb_rand_uni_bound = emb_rand_uni_bound
        self.freeze_knrm_embeddings = freeze_knrm_embeddings
        self.knrm_kernel_num = knrm_kernel_num
        self.knrm_out_mlp = knrm_out_mlp
        self.dataloader_bs = dataloader_bs
        self.train_lr = train_lr
        self.change_train_loader_ep = change_train_loader_ep

        self.model, self.vocab, self.unk_words = self.build_knrm_model()
        self.idx_to_text_mapping_train = self.get_idx_to_text_mapping(
            self.glue_train_df)
        self.idx_to_text_mapping_dev = self.get_idx_to_text_mapping(
            self.glue_dev_df)

        self.val_dataset = ValPairsDataset(self.dev_pairs_for_ndcg,
              self.idx_to_text_mapping_dev,
              vocab=self.vocab, oov_val=self.vocab['OOV'],
              preproc_func=self.simple_preproc)
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.dataloader_bs, num_workers=0,
            collate_fn=collate_fn, shuffle=False)

    def get_glue_df(self, partition_type: str) -> pd.DataFrame:
        assert partition_type in ['dev', 'train']
        glue_df = pd.read_csv(
            self.glue_qqp_dir + f'/{partition_type}.tsv', sep='\t', dtype=object) # error_bad_lines=False
        glue_df = glue_df.dropna(axis=0, how='any').reset_index(drop=True)
        glue_df_fin = pd.DataFrame({
            'id_left': glue_df['qid1'],
            'id_right': glue_df['qid2'],
            'text_left': glue_df['question1'],
            'text_right': glue_df['question2'],
            'label': glue_df['is_duplicate'].astype(int)
        })
        return glue_df_fin

    def hadle_punctuation(self, inp_str: str) -> str:
        # очистить текст от пунктуации string.punctuation
        return ''.join([c for c in inp_str if c not in string.punctuation])

    def simple_preproc(self, inp_str: str) -> List[str]:
        # обработка пунктуации и приведение к нижнему регистру
        return [word.lower() for word in nltk.word_tokenize(self.hadle_punctuation(inp_str))]

    def _filter_rare_words(self, vocab: Dict[str, int], min_occurancies: int) -> Dict[str, int]:
        # очистить словарь по порогу min_occurancies
        return {k: v for k, v in vocab.items() if v >= min_occurancies}

    def get_all_tokens(self, list_of_df: List[pd.DataFrame], min_occurancies: int) -> List[str]:
        # сформировать уникальное множество токенов из list_of_df используя simple_preproc
        # и отфильтровать используя _filter_rare_words
        all_tokens = []
        sot = set()
        for df in list_of_df:
            # взять 2 и 3 столбец из df и применить к ним simple_preproc
            for col in [2,3]:
                # sot = sot.union(set(df.iloc[:, col].apply(self.simple_preproc).sum()))
                # sot.update(df.iloc[:, col].apply(self.simple_preproc).sum())
                sot.update(item for sublist in df.iloc[:, col].apply(self.simple_preproc) for item in sublist)
 
        all_tokens = list(sot)
        all_tokens = self._filter_rare_words(Counter(all_tokens), min_occurancies)
        # import pdb; pdb.set_trace()
        return list(all_tokens.keys())
    
    def _read_glove_embeddings(self, file_path: str) -> Dict[str, List[str]]:
        d = dict()
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split()  # Разделение строки на части
                word = parts[0]  # Первый элемент — слово
                vector = list(map(float, parts[1:]))  # Остальные элементы — вектор
                d[word] = vector
        return d
        
    def create_glove_emb_from_file(self, file_path: str, inner_keys: List[str],
                                   random_seed: int, rand_uni_bound: float
                                   ) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
        """
        :param file_path: путь к файлу с эмбеддингами
        :param inner_keys: список ключей(токенов), для которых нужно получить эмбеддинги
        :param random_seed: seed для torch.manual_seed
        :param rand_uni_bound: граница для torch.nn.init.uniform_
        :return: кортеж из матрицы эмбеддингов, словаря токен-индекс, списка неизвестных слов
        """
        glove_dict = self._read_glove_embeddings(file_path)
        emb_size = len(next(iter(glove_dict.values())))
        # в словарь нужно добавить токены OOV и PAD с индексами 0 и 1
        vocab = {k: i + 2 for i, k in enumerate(inner_keys)}
        vocab['OOV'] = 0
        vocab['PAD'] = 1
        unk_words = []
        emb_matrix = np.zeros((len(vocab), emb_size))
        torch.manual_seed(random_seed)
        for token, idx in vocab.items():
            if token in glove_dict:
                emb_matrix[idx] = glove_dict[token]
            else:
                unk_words.append(token)
                torch.nn.init.uniform_(torch.from_numpy(emb_matrix[idx]), -rand_uni_bound, rand_uni_bound)
        return emb_matrix, vocab, unk_words

    def build_knrm_model(self) -> Tuple[torch.nn.Module, Dict[str, int], List[str]]:
        emb_matrix, vocab, unk_words = self.create_glove_emb_from_file(
            self.glove_vectors_path, self.all_tokens, self.random_seed, self.emb_rand_uni_bound)
        torch.manual_seed(self.random_seed)
        knrm = KNRM(emb_matrix, freeze_embeddings=self.freeze_knrm_embeddings,
                    out_layers=self.knrm_out_mlp, kernel_num=self.knrm_kernel_num)
        return knrm, vocab, unk_words

    def sample_data_for_train_iter(self, inp_df: pd.DataFrame, seed: int
                                   ) -> List[List[Union[str, float]]]:
        groups = inp_df[['id_left', 'id_right', 'label']].groupby('id_left')
        pairs_w_labels = []
        np.random.seed(seed)
        all_right_ids = inp_df.id_right.values
        for id_left, group in groups:
            labels = group.label.unique()
            if len(labels) > 1:
                for label in labels:
                    same_label_samples = group[group.label ==
                                               label].id_right.values
                    if label == 0 and len(same_label_samples) > 1:
                        sample = np.random.choice(
                            same_label_samples, 2, replace=False)
                        pairs_w_labels.append(
                            [id_left, sample[0], sample[1], 0.5])
                    elif label == 1:
                        less_label_samples = group[group.label <
                                                   label].id_right.values
                        pos_sample = np.random.choice(
                            same_label_samples, 1, replace=False)
                        if len(less_label_samples) > 0:
                            neg_sample = np.random.choice(
                                less_label_samples, 1, replace=False)
                        else:
                            neg_sample = np.random.choice(
                                all_right_ids, 1, replace=False)
                        pairs_w_labels.append(
                            [id_left, pos_sample[0], neg_sample[0], 1])
        return pairs_w_labels

    def create_val_pairs(self, inp_df: pd.DataFrame, fill_top_to: int = 15,
                         min_group_size: int = 2, seed: int = 0) -> List[List[Union[str, float]]]:
        inp_df_select = inp_df[['id_left', 'id_right', 'label']]
        inf_df_group_sizes = inp_df_select.groupby('id_left').size()
        glue_dev_leftids_to_use = list(
            inf_df_group_sizes[inf_df_group_sizes >= min_group_size].index)
        groups = inp_df_select[inp_df_select.id_left.isin(
            glue_dev_leftids_to_use)].groupby('id_left')

        all_ids = set(inp_df['id_left']).union(set(inp_df['id_right']))

        out_pairs = []

        np.random.seed(seed)

        for id_left, group in groups:
            ones_ids = group[group.label > 0].id_right.values
            zeroes_ids = group[group.label == 0].id_right.values
            sum_len = len(ones_ids) + len(zeroes_ids)
            num_pad_items = max(0, fill_top_to - sum_len)
            if num_pad_items > 0:
                cur_chosen = set(ones_ids).union(
                    set(zeroes_ids)).union({id_left})
                pad_sample = np.random.choice(
                    list(all_ids - cur_chosen), num_pad_items, replace=False).tolist()
            else:
                pad_sample = []
            for i in ones_ids:
                out_pairs.append([id_left, i, 2])
            for i in zeroes_ids:
                out_pairs.append([id_left, i, 1])
            for i in pad_sample:
                out_pairs.append([id_left, i, 0])
        return out_pairs

    def get_idx_to_text_mapping(self, inp_df: pd.DataFrame) -> Dict[str, str]:
        left_dict = (
            inp_df
            [['id_left', 'text_left']]
            .drop_duplicates()
            .set_index('id_left')
            ['text_left']
            .to_dict()
        )
        right_dict = (
            inp_df
            [['id_right', 'text_right']]
            .drop_duplicates()
            .set_index('id_right')
            ['text_right']
            .to_dict()
        )
        left_dict.update(right_dict)
        return left_dict

    def ndcg_k(self, ys_true: np.array, ys_pred: np.array, ndcg_top_k: int = 10) -> float:
        def dcg(ys_true, ys_pred):
            argsort = np.argsort(ys_pred)[::-1]
            argsort = argsort[:ndcg_top_k]
            ys_true_sorted = ys_true[argsort]
            ret = 0
            for i, l in enumerate(ys_true_sorted, 1):
                ret += (2 ** l - 1) / math.log2(1 + i)
            return ret
        ideal_dcg = dcg(ys_true, ys_true)
        pred_dcg = dcg(ys_true, ys_pred)
        return (pred_dcg / ideal_dcg)

    def valid(self, model: torch.nn.Module, val_dataloader: torch.utils.data.DataLoader) -> float:
        labels_and_groups = val_dataloader.dataset.index_pairs_or_triplets
        labels_and_groups = pd.DataFrame(labels_and_groups, columns=['left_id', 'right_id', 'rel'])

        all_preds = []
        for batch in (val_dataloader):
            inp_1, y = batch
            preds = model.predict(inp_1)
            preds_np = preds.detach().numpy()
            all_preds.append(preds_np)
        all_preds = np.concatenate(all_preds, axis=0)
        labels_and_groups['preds'] = all_preds

        ndcgs = []
        for cur_id in labels_and_groups.left_id.unique():
            cur_df = labels_and_groups[labels_and_groups.left_id == cur_id]
            ndcg = self.ndcg_k(cur_df.rel.values.reshape(-1), cur_df.preds.values.reshape(-1))
            if np.isnan(ndcg):
                ndcgs.append(0)
            else:
                ndcgs.append(ndcg)
        return np.mean(ndcgs)

    def train(self, n_epochs: int):
        opt = torch.optim.SGD(self.model.parameters(), lr=self.train_lr)
        criterion = torch.nn.BCELoss()
        ndcgs = []
        for ep in range(n_epochs):
            if ep % self.change_train_loader_ep == 0:
                sampled_train_triplets = self.sample_data_for_train_iter(self.glue_train_df, seed = ep)
                train_dataset = TrainTripletsDataset(sampled_train_triplets, 
                        self.idx_to_text_mapping_train, 
                        vocab=self.vocab, oov_val=self.vocab['OOV'], 
                        preproc_func=self.simple_preproc)
                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=self.dataloader_bs, num_workers=0, 
                    collate_fn=collate_fn, shuffle=True, )

            for batch in (train_dataloader):

                inp_1, inp_2, y = batch
                preds = self.model(inp_1, inp_2)
                loss = criterion(preds, y)
                loss.backward()
                opt.step()
            ndcg = self.valid(self.model, self.val_dataloader)
            ndcgs.append(ndcg)
            if ndcg > 0.925:
                break


# sol = Solution(glue_qqp_dir, glove_path, knrm_out_mlp=[])

# sol.train(10)
# state_mlp = sol.model.mlp.state_dict()
# torch.save(state_mlp, open('../lec11/user_input/knrm_mlp.bin', 'wb'))

# state_emb = sol.model.embeddings.state_dict()
# # torch.save(state_emb, open('../lec11/user_input/knrm_emb.bin', 'wb'))
# torch.save(state_emb, open('../additional_data/lec11/knrm_emb.bin', 'wb'))

# import json
# state_vocab = sol.vocab
# json.dump(state_vocab, open('../additional_data/lec11/vocab.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)