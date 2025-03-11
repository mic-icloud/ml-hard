import math
import pickle
import random
from typing import List, Tuple

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tqdm.auto import tqdm


class Solution:
    def __init__(self, n_estimators: int = 100, lr: float = 0.5, ndcg_top_k: int = 10,
                 subsample: float = 0.6, colsample_bytree: float = 0.9,
                 max_depth: int = 5, min_samples_leaf: int = 8):
        self._prepare_data()

        self.num_input_features = self.X_train.shape[1]
        self.num_train_objects = self.X_train.shape[0]
        self.num_test_objects = self.X_test.shape[0]

        self.features_to_choice = int(
            self.num_input_features * colsample_bytree)
        self.objects_to_choice = int(self.num_train_objects * subsample)

        self.ndcg_top_k = ndcg_top_k
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        self.trees = None
        self.trees_feat_idxs = None
        self.best_ndcg = -1
        self.best_iter_idx = -1

    def _get_data(self) -> List[np.ndarray]:
        train_df, test_df = msrank_10k()

        X_train = train_df.drop([0, 1], axis=1).values
        y_train = train_df[0].values
        query_ids_train = train_df[1].values.astype(int)

        X_test = test_df.drop([0, 1], axis=1).values
        y_test = test_df[0].values
        query_ids_test = test_df[1].values.astype(int)

        return [X_train, y_train, query_ids_train, X_test, y_test, query_ids_test]

    def _prepare_data(self) -> None:
        (X_train, y_train, self.query_ids_train,
            X_test, y_test, self.query_ids_test) = self._get_data()
        X_train = self._scale_features_in_query_groups(
            X_train, self.query_ids_train)
        X_test = self._scale_features_in_query_groups(
            X_test, self.query_ids_train)

        self.X_train = torch.FloatTensor(X_train)
        self.X_test = torch.FloatTensor(X_test)

        self.ys_train = torch.FloatTensor(y_train).reshape(-1, 1)
        self.ys_test = torch.FloatTensor(y_test).reshape(-1, 1)

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        for cur_id in np.unique(inp_query_ids):
            mask = inp_query_ids == cur_id
            tmp_array = inp_feat_array[mask]
            scaler = StandardScaler()
            inp_feat_array[mask] = scaler.fit_transform(tmp_array)

        return inp_feat_array

    def _train_one_tree(self, cur_tree_idx: int,
                        train_preds: torch.FloatTensor
                        ) -> Tuple[DecisionTreeRegressor, np.ndarray]:
        lambdas = torch.zeros(self.num_train_objects, 1)
        for cur_id in np.unique(self.query_ids_train):
            train_mask = self.query_ids_train == cur_id
            lambda_update = self._compute_lambdas(
                self.ys_train[train_mask], train_preds[train_mask])
            if any(torch.isnan(lambda_update)):
                lambda_update = torch.zeros_like(lambda_update)
            lambdas[train_mask] = lambda_update

        tree = DecisionTreeRegressor(
            max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, random_state=cur_tree_idx)

        this_tree_feats = np.random.choice(
            list(range(self.num_input_features)), self.features_to_choice, replace=False)
        this_tree_objs = np.random.choice(
            list(range(self.num_train_objects)), self.objects_to_choice, replace=False)

        tree.fit(
            self.X_train[this_tree_objs.reshape(-1)
                         ][:, this_tree_feats].numpy(),
            -lambdas[this_tree_objs.reshape(-1), :].numpy()
        )

        return tree, this_tree_feats

    def _calc_data_ndcg(self, queries_list: np.ndarray,
                        true_labels: torch.FloatTensor, preds: torch.FloatTensor) -> float:
        ndsgs = []
        for query in np.unique(queries_list):
            mask = queries_list == query
            cur_true_labels = true_labels[mask]
            cur_preds = preds[mask]
            ndsgs.append(self._ndcg_k(cur_true_labels, cur_preds, self.ndcg_top_k))
        return np.mean(ndsgs)

    def fit(self):
        np.random.seed(0)
        # допишите ваш код здесь
        pass

    def predict(self, data: torch.FloatTensor) -> torch.FloatTensor:
        preds = torch.zeros(data.shape[0], 1)
        for cur_tree_idx in range(len(self.trees)):
            tree = self.trees[cur_tree_idx]
            features_idx = self.trees_feat_idxs[cur_tree_idx]
            tmp_preds = tree.predict(data[:, features_idx].numpy())
            preds += self.lr * torch.FloatTensor(tmp_preds).reshape(-1, 1)
        
        return preds

    def _compute_lambdas(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        # допишите ваш код здесь
        pass

    def _ndcg_k(self, ys_true, ys_pred, ndcg_top_k) -> float:
        def dcg(ys_true, ys_pred):
            _, argsort = torch.sort(ys_pred, descending=True, dim=0)
            argsort = argsort[:ndcg_top_k]
            ys_true_sorted = ys_true[argsort]
            ret = 0
            for i, l in enumerate(ys_true_sorted, 1):
                ret += (2 ** l - 1) / math.log2(1 + i)
            return ret
        ideal_dcg = dcg(ys_true, ys_true)
        pred_dcg = dcg(ys_true, ys_pred)
        return (pred_dcg / ideal_dcg).item()

    def save_model(self, path: str):
        # сохранить self.model с тремя полями: n_estimators, max_depth, min_samples_leaf
        state = {
            'trees': self.trees,
            'trees_feat_idxs': self.trees_feat_idxs,
            'best_ndcg': self.best_ndcg,
            'lr': self.lr
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)


    def load_model(self, path: str):
        # загрузить self.model с тремя полями: n_estimators, max_depth, min_samples_leaf
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.trees = state['trees']
        self.trees_feat_idxs = state['trees_feat_idxs']
        self.best_ndcg = state['best_ndcg']
        self.lr = state['lr']


net = Solution()
net.fit()