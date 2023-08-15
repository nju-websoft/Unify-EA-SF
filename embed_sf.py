import argparse

import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
import scipy.sparse as ss
import cupy as cp
from pathlib import Path


parser = argparse.ArgumentParser(description='Embed_SF')
parser.add_argument('--input', type=str, default="./dataset/dbp15k/zh_en/")
# parser.add_argument('--input', type=str, default="./dataset/openea/D_W_15K_V1/")
parser.add_argument('--model', type=str, default="TransE")


class KnowledgeGraph():
    def __init__(self, triples) -> None:
        super().__init__()
        for col in ['h', 'r', 't']:
            triples[col] = triples[col].astype(str)
        ents = np.sort(pd.concat([triples['h'], triples['t']]).unique())
        self.entities = CategoricalDtype(ents, ordered=True)
        self.triples = triples.copy()
        self.triples['h'] = self.triples['h'].astype(self.entities)
        self.triples['t'] = self.triples['t'].astype(self.entities)
        self.relations = CategoricalDtype(
            np.sort(self.triples['r'].unique()), ordered=True)
        self.triples['r'] = self.triples['r'].astype(self.relations)
        assert not self.triples.isna().values.any()

    def __call__(self, triples: pd.DataFrame, ent_cols=None, rel_cols=None):
        triples = triples.copy()

        for [presets, cols, dtype] in ([['h', 't'], ent_cols, self.entities], [['r'], rel_cols, self.relations]):
            for col in (presets if cols is None else cols):
                if col in triples.columns:
                    raw = triples[col]
                    triples[col] = triples[col].astype(str).astype(dtype)
                    if triples[col].isna().any():
                        print('kg encode error %s' % col)
                        print(raw[triples[col].isna()])

        return triples.dropna()


def counting(df, idx_col, val_col, dims):
    cnt = df.groupby(by=idx_col)[val_col].count().reset_index()
    dim = (dims[idx_col[0]], dims[idx_col[1]])
    return ss.csr_matrix((
        cnt[val_col].values, 
        (cnt[idx_col[0]].values, cnt[idx_col[1]].values)
    ), dim)


def load_data_dbp15k(task_dir):
    print("load dbp15k data...")
    task_dir = task_dir / 'mtranse' / '0_3'
    kg1, kg2 = [
        KnowledgeGraph(
            pd.read_csv(task_dir / part, names=['h', 'r', 't'], sep='\t')
        ) for part in ['triples_1', 'triples_2']
    ]

    train, test = [
        kg1(kg2(
            pd.read_csv(task_dir / part, names=['e1', 'e2'], sep='\t'),
            ent_cols=['e2']),
            ent_cols=['e1'])
        for part in ['sup_pairs', 'ref_pairs']
    ]

    train, test = [
        pd.DataFrame(dict((k, part[k].cat.codes) for k in ['e1', 'e2'])).values
        for part in [train, test]
    ]

    return kg1, kg2, train, test


def load_data_openea(task_dir, splition_id):
    print("load openea data...")
    kg1, kg2 = [
        KnowledgeGraph(
            pd.read_csv(task_dir / part, names=['h', 'r', 't'], sep='\t')
        ) for part in ['rel_triples_1', 'rel_triples_2']
    ]

    train, test = [
        kg1(kg2(
            pd.read_csv(task_dir / '721_5fold' / str(splition_id) / part, names=['e1', 'e2'], sep='\t'),
            ent_cols=['e2']),
            ent_cols=['e1'])
        for part in ['train_links', 'test_links']
    ]

    train, test = [
        pd.DataFrame(dict((k, part[k].cat.codes) for k in ['e1', 'e2'])).values
        for part in [train, test]
    ]

    return kg1, kg2, train, test


def transe_lambda_weight(triples):
    print("compute lambda values based on TransE...")
    ent_dim = max(triples['h'].max(), triples['t'].max()) + 1
    rel_dim = triples['r'].max() + 1
    dims = dict(h=ent_dim, t=ent_dim, r=rel_dim)
    rh = counting(triples, ['r', 'h'], 't', dims)
    rt = counting(triples, ['r', 't'], 'h', dims)
    R = (rt - rh).multiply(ss.csr_matrix(1. / np.maximum(rh.sum(1), 1)))

    et = counting(triples, ['h', 't'], 'r', dims)
    eh = counting(triples, ['t', 'h'], 'r', dims)
    e_c = et.sum(1) + eh.sum(1)
    e = eh + et

    et = -counting(triples, ['h', 'r'], 't', dims) @ R
    eh = counting(triples, ['t', 'r'], 'h', dims) @ R
    e += et + eh

    return e.multiply(ss.csr_matrix(1. / np.maximum(e_c, 1))), (e, e_c)


def gcn_lambda_weight(triples):
    print("compute lambda values based on GCN...")
    ent_dim = max(triples['h'].max(), triples['t'].max()) + 1
    ent_dim = max(triples['h'].max(), triples['t'].max()) + 1
    rel_dim = triples['r'].max() + 1
    dims = dict(h=ent_dim, t=ent_dim, r=rel_dim)

    et = (counting(triples, ['h', 't'], 'r', dims) >= 1).astype(int)
    eh = (counting(triples, ['t', 'h'], 'r', dims) >= 1).astype(int)
    e_c = et.sum(1) + eh.sum(1)
    e = eh + et

    return e.multiply(1. / e_c), (e, e_c)


def run_train(kg1, kg2, train, test, initial_sim=None, lambda_fn=transe_lambda_weight, max_step=3):

    t1, t2 = (pd.DataFrame(dict((col, kg.triples[col].cat.codes) for col in 'hrt')) for kg in (kg1, kg2))

    H1, _ = lambda_fn(t1)
    H2, _ = lambda_fn(t2)
    T = ss.csr_matrix((np.ones(train.shape[0]), (train[:, 0], train[:, 1])), shape=(H1.shape[0], H2.shape[0]))
    H1 = H1.toarray()
    H2 = H2.toarray()
    H1 = cp.array(H1 / np.sqrt(np.square(H1).sum(1, keepdims=True)))
    H2 = cp.array(H2 / np.sqrt(np.square(H2).sum(1, keepdims=True)))
    T = T.toarray()

    T = cp.array(T)
    T1 = T.copy()
    T1 = np.clip(T, 0, 1)
    for i in range(0, max_step):
        T1 = cp.array(T1)
        T1 = cp.matmul(cp.matmul(H1, T1), H2.T)

        T1 = cp.clip(T1 + T * 2, -1, 1)
        # csls
        T1 = T1 - T1.mean(1, keepdims=True) * 0.5 - T1.mean(0, keepdims=True) * 0.5
        T1 = cp.clip(T1 + T * 2, -1, 1)

        # add penalty to duplicated matches of each entity
        scale = cp.maximum(
            (T1 * (T1 > 0.0)).sum(1, keepdims=True),
            (T1 * (T1 > 0.0)).sum(0, keepdims=True)
        )

        T1 = T1 / cp.maximum(scale, 1)


        T1 = cp.clip(T1 + T * 2, -1, 1)
        T1 = cp.asnumpy(T1)

        if initial_sim is not None:
            T1 = T1 * 0.9 + initial_sim* 0.1

        out = T1[np.ix_(test[:, 0], test[:, 1])]
        # evaluation
        rank = (out >= np.diag(out).reshape((-1, 1))).sum(1)
        print("epoch {}: H@1={}, H@10={}, MRR={}".format(i+1, (rank <= 1).mean(), (rank <= 10).mean(), (1. / rank).mean()))
        # yield (rank <= 1).mean(), (rank <= 10).mean(), (1. / rank).mean()


if __name__ == '__main__':
    args = parser.parse_args()

    if "openea" in args.input:
        kg1, kg2, train, test = load_data_openea(Path(args.input), 3)
    elif "dbp15k" in args.input:
        kg1, kg2, train, test = load_data_dbp15k(Path(args.input))
    
    if args.model == "TransE":
        lambda_fn = transe_lambda_weight
    elif args.model == "GCN":
        lambda_fn = gcn_lambda_weight
    
    run_train(kg1, kg2, train, test, initial_sim=None, lambda_fn=lambda_fn, max_step=5)
