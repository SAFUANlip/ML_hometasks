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
from numpy.random import default_rng
from IPython.display import clear_output
import pandas as pd 
import matplotlib.pyplot as plt

from ndcg import ndcg, dcg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Solution:
    def __init__(self, n_estimators: int = 100, lr: float = 0.5, ndcg_top_k: int = 10,
                 subsample: float = 0.6, colsample_bytree: float = 0.9,
                 max_depth: int = 5, min_samples_leaf: int = 8):
        self.X_train = None
        self.ys_train = None
        self.X_test = None
        self.ys_test = None
        self._prepare_data()

        self.ndcg_top_k = ndcg_top_k
        self.n_estimators = n_estimators  # количество деревьев
        self.lr = lr  # Learning Rate, коэффициент, на который умножаются предсказания каждого нового дерева
        self.max_depth = max_depth  # максимальная глубина
        self.min_samples_leaf = min_samples_leaf  # минимальное количество термальных листьев

        self.subsample = subsample  # доля объектов от выборки
        self.colsample_bytree = colsample_bytree  # доля признаков от выборки

        self.trees: List[DecisionTreeRegressor] = []  # все деревья
        self.trees_columns = [] # columns which use tree, my update
        self.all_ndcg: List[float] = []
        self.best_ndcg = float(0.0)
        self.k_best_trees = 0

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
        
        self.X_train = torch.FloatTensor(self._scale_features_in_query_groups(X_train, self.query_ids_train))
        self.ys_train = torch.FloatTensor(y_train).reshape(-1, 1)

        self.X_test = torch.FloatTensor(self._scale_features_in_query_groups(X_test, self.query_ids_test))
        self.ys_test = torch.FloatTensor(y_test).reshape(-1, 1)


    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        for id in np.unique(inp_query_ids):
            scaler = StandardScaler()
            idxs = inp_query_ids == id
            inp_feat_array[idxs] = scaler.fit_transform(inp_feat_array[idxs])

        return inp_feat_array

    def _train_one_tree(self, cur_tree_idx: int,
                        train_preds: torch.FloatTensor
                        ) -> Tuple[DecisionTreeRegressor, np.ndarray]:
        """
        Метод для тренировки одного дерева.

        @cur_tree_idx: номер текущего дерева, который предлагается использовать в качестве random_seed для того,
        чтобы алгоритм был детерминирован.
        @train_preds: суммарные предсказания всех предыдущих деревьев (для расчёта лямбд).
        @return: это само дерево и индексы признаков, на которых обучалось дерево
        """
        # допишите ваш код здесь
        _, _, _, _, lambda_update = self._compute_lambdas(self.ys_train, train_preds)
        
        tree_train = self.trees[cur_tree_idx]
        rng = default_rng()
        indices = rng.choice(self.X_train.shape[0], size=int(self.X_train.shape[0]*self.subsample), replace=False)
        columns = rng.choice(self.X_train.shape[1], size=int(self.X_train.shape[1]*self.colsample_bytree), replace=False)
        
        X_bott = self.X_train[indices]
        X_bott = X_bott[:, columns]
        y_boot = lambda_update[indices] 
        
        tree_train.fit(X_bott, y_boot)
        return tree_train, columns

    def _calc_data_ndcg(self, queries_list: np.ndarray,
                        true_labels: torch.FloatTensor, preds: torch.FloatTensor) -> float:
        """ Расчёт метрики по набору данных """
        for id in np.unique(queries_list):
            idxs = queries_list == id
            true_labels_i = true_labels[idxs]
            preds_i = preds[idxs]

            score = self._ndcg_k(true_labels_i, preds_i, self.ndcg_top_k)
            self.all_ndcg.append(score)
        return np.mean(self.all_ndcg)

    def fit(self):
        """
        генеральный метод обучения K деревьев, каждое из которых тренируется
        с использованием метода _train_one_tree
        """
        set_seed(0)
        y_pred = torch.zeros_like(self.ys_train)
        y_pred_validation = torch.zeros_like(self.ys_train)
        self.train_history = []
        self.val_history = []
        for i in range(self.n_estimators):
            self.trees.append(DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf))
            tree, columns = self._train_one_tree(i, y_pred)
            self.trees_columns.append(columns)
            lambda_i = torch.FloatTensor(tree.predict(self.X_train[:, columns])).reshape(-1, 1)
            lambda_i_validation = torch.FloatTensor(tree.predict(self.X_test[:, columns])).reshape(-1, 1)
            y_pred -= self.lr * lambda_i
            y_pred_validation -= self.lr * lambda_i_validation
            
            self.train_history.append(self._calc_data_ndcg(self.query_ids_train, self.ys_train, y_pred))
            self.val_history.append(self._calc_data_ndcg(self.query_ids_test, self.ys_test, y_pred_validation))
            if self.best_ndcg < self.train_history[-1]:
                self.best_ndcg = self.train_history[-1]
                self.k_best_trees = i
            
            clear_output(True)
            plt.plot(np.arange(i+1), self.train_history, label = 'train')
            plt.plot(np.arange(i+1), self.val_history, label = 'val')
            plt.xlabel("number of trees")
            plt.ylabel("ndcg")
            plt.legend()
            plt.show()
            print(f"tree: {i}\nndcg on train: {self.train_history[-1]}")
            print(f"ndcg on validation: {self.val_history[-1]}")
            

    def predict(self, data: pd.DataFrame) -> torch.FloatTensor:
        X_test = data.drop([1], axis=1).values
        query_ids_test = data[1].values.astype(int)
        
        X_test = torch.FloatTensor(self._scale_features_in_query_groups(X_test, query_ids_test))
        
        predict = torch.tensor((), dtype=torch.float64)
        predict = predict.new_zeros((X_test.shape[0], 1))
        for i in range(self.k_best_trees):
            tree = self.trees[i]
            predict -= self.lr * torch.FloatTensor(tree.predict(X_test[:, self.trees_columns[i]])).reshape(-1, 1)
        return predict
            

    def _compute_lambdas(self, y_true, y_pred, ndcg_scheme='exp2'):
        # рассчитаем нормировку, IdealDCG
        ideal_dcg = dcg(y_true, y_true, ndcg_scheme)
        N = 1 / ideal_dcg

        # рассчитаем порядок документов согласно оценкам релевантности
        _, rank_order = torch.sort(y_true, descending=True, axis=0)
        rank_order += 1

        with torch.no_grad():
            # получаем все попарные разницы скоров в батче
            pos_pairs_score_diff = 1.0 + torch.exp((y_pred - y_pred.t()))

            # поставим разметку для пар, 1 если первый документ релевантнее
            # -1 если второй документ релевантнее
            Sij = self._compute_labels_in_batch(y_true)
            # посчитаем изменение gain из-за перестановок
            gain_diff = self._compute_gain_diff(y_true, ndcg_scheme)

            # посчитаем изменение знаменателей-дискаунтеров
            decay_diff = (1.0 / torch.log2(rank_order + 1.0)) - (1.0 / torch.log2(rank_order.t() + 1.0))
            # посчитаем непосредственное изменение nDCG
            delta_ndcg = torch.abs(N * gain_diff * decay_diff)
            # посчитаем лямбды
            lambda_update =  (0.5 * (1 - Sij) - 1 / pos_pairs_score_diff) * delta_ndcg
            lambda_update = torch.sum(lambda_update, dim=1, keepdim=True)

            return Sij, gain_diff, decay_diff, delta_ndcg, lambda_update


    def _compute_labels_in_batch(self, y_true):
        # разница релевантностей каждого с каждым объектом
        rel_diff = y_true - y_true.t()

        # 1 в этой матрице - объект более релевантен
        pos_pairs = (rel_diff > 0).type(torch.float32)

        # 1 тут - объект менее релевантен
        neg_pairs = (rel_diff < 0).type(torch.float32)
        Sij = pos_pairs - neg_pairs
        return Sij


    def _compute_gain_diff(self, y_true, gain_scheme):
        if gain_scheme == "exp2":
            gain_diff = torch.pow(2.0, y_true) - torch.pow(2.0, y_true.t())
        elif gain_scheme == "diff":
            gain_diff = y_true - y_true.t()
        else:
            raise ValueError(f"{gain_scheme} method not supported")
        return gain_diff

    def _ndcg_k(self, ys_true, ys_pred, ndcg_top_k) -> float:
        try:
            return ndcg(ys_true, ys_pred, gain_scheme='exp2', top_k=ndcg_top_k)
        except ZeroDivisionError:
            return float(0)

    def save_model(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"trees": self.trees, "trees_col": self.trees_columns, "k_best_tr": self.k_best_trees, "lr": self.lr}, f)

    def load_model(self, path: str):
        with open(path, "rb") as f:
            fields = pickle.load(f)
        self.trees, self.trees_columns, self.k_best_trees, self.lr = fields["trees"], fields["trees_col"], fields["k_best_tr"], fields["lr"]