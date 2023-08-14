import pandas as pd
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
import scipy.sparse as ss
import cupy as cp
from pathlib import Path


class KnowledgeGraph():
    def __init__(self, triples, type_rel) -> None:
        super().__init__()
        for col in ['h', 'r', 't']:
            triples[col] = triples[col].astype(str)
        ents = np.sort(pd.concat([triples['h'], triples['t']]).unique())
        self.entities = CategoricalDtype(ents, ordered=True)
        ht = triples[triples['r'] == type_rel][['h', 't']]
        classes = np.sort(ht['t'].unique())
        self.classes = CategoricalDtype(classes, ordered=True)
        self.triples = triples.drop(index=ht.index).drop_duplicates()
        self.triples['h'] = self.triples['h'].astype(self.entities)
        self.triples['t'] = self.triples['t'].astype(self.entities)
        self.relations = CategoricalDtype(np.sort(self.triples['r'].unique()), ordered=True)
        self.triples['r'] = self.triples['r'].astype(self.relations)
        assert not self.triples.isna().values.any()
        # assert not self.type_triples.isna().values.any()
    
    def entity_count(self):
        return len(self.entities.categories)
    
    def relation_count(self):
        return len(self.relations.categories)
    
    def class_count(self):
        return len(self.classes.categories)
    
    def __call__(self, triples : pd.DataFrame, ent_cols=None, rel_cols=None, cls_cols=None):
        triples = triples.copy()

        for col in (['h', 't'] if ent_cols is None else ent_cols):
            if col in triples.columns:
                raw = triples[col]
                triples[col] = triples[col].astype(str).astype(self.entities)
                if triples[col].isna().any():
                    print('kg encode error %s' % col)
                    print(raw[triples[col].isna()])

        for col in (['r'] if rel_cols is None else rel_cols):
            if col in triples.columns:
                raw = triples[col]
                triples[col] = triples[col].astype(self.relations)
                if triples[col].isna().any():
                    print('kg encode error %s' % col)
                    print(raw[triples[col].isna()])
        
        for col in ([] if cls_cols is None else cls_cols):
            if col in triples.columns:
                raw = triples[col]
                triples[col] = triples[col].astype(self.classes)
                if triples[col].isna().any():
                    print('kg encode error %s' % col)
                    print(raw[triples[col].isna()])
        
        return triples.dropna()


def train(task_name, max_step=2):
    base_dir = Path('./dbp15k/') / task_name / 'mtranse' / '0_3'

    train = np.load(base_dir / 'train.npy')
    test = np.load(base_dir / 'test.npy')
    # valid = np.load(base_dir / 'valid.npy')

    H1 = ss.load_npz(base_dir / 'kg1.npz').toarray()
    H2 = ss.load_npz(base_dir / 'kg2.npz').toarray()
    T = ss.csr_matrix((np.ones(train.shape[0]), (train[:, 0], train[:, 1])), shape=H1.shape).toarray()

    H1 = H1 / np.maximum(np.sqrt(np.square(H1).sum(1, keepdims=True)), 1e-8)
    H2 = H2 / (np.sqrt(np.square(H2).sum(1, keepdims=True)) + 1e-8)
    H1 = H1 / (H1.sum(1, keepdims=True) + 1e-8)
    H2 = H2 / (H2.sum(1, keepdims=True) + 1e-8)
    H1 = H1 / np.sqrt(np.square(H1).sum(1, keepdims=True))
    H2 = H2 / np.sqrt(np.square(H2).sum(1, keepdims=True))
    H1[np.isnan(H1)] = 0.0
    H2[np.isnan(H2)] = 0.0

    T1 = T.copy()
    T1 = np.clip(T, 0, 1)
    for i in range(0, 2):
        T1 = cp.asnumpy(cp.matmul(cp.matmul(cp.array(H1), cp.array(T1)), cp.array(H2.T)))
        cp.get_default_memory_pool().free_all_blocks()
        
        T1 = np.clip(T1 + T * 2, 0, 1)

        T1 = T1 / (np.maximum((T1 * (T1 > 0)).sum(1, keepdims=True), 1))
        T1 = T1 / (np.maximum((T1 * (T1 > 0)).sum(0, keepdims=True), 1))

        # T1 = T1 - T1.mean(1, keepdims=True) * 0.5 - T1.mean(0, keepdims=True) * 0.5
        T1 = np.clip(T1 + T * 2, 0, 1)
        out = T1[np.ix_(test[:, 0], test[:, 1])]
        rank = (out >= np.diag(out).reshape((-1, 1))).sum(1)
        print("H@1/5/10 and MR = ", (rank <= 1).mean(), (rank <= 5).mean(), (rank <= 10).mean(), rank.mean())

def preprocess():
    base_dir = Path('./dbp15k/')
    dim = 20000
    for task_name in ['fr_en', 'ja_en', 'zh_en']:
        print(task_name)
        task_dir = base_dir / task_name / 'mtranse' / '0_3'
        kg1 = KnowledgeGraph(pd.read_csv(task_dir / 'triples_1', names=['h', 'r', 't'], sep='\t'), type_rel='ttt')
        kg2 = KnowledgeGraph(pd.read_csv(task_dir / 'triples_2', names=['h', 'r', 't'], sep='\t'), type_rel='ttt')

        train, test = [
            kg1(kg2(
                pd.read_csv(task_dir / part, names=['e1', 'e2'], sep='\t'), 
                ent_cols=['e2']),
            ent_cols=['e1'])
            for part in ['sup_pairs', 'ref_pairs']
        ]

        np.save(task_dir / 'train', pd.DataFrame(dict(e1=train['e1'].cat.codes, e2=train['e2'].cat.codes)).values)
        np.save(task_dir / 'test', pd.DataFrame(dict(e1=test['e1'].cat.codes, e2=test['e2'].cat.codes)).values)

        for name, kg in (('kg1', kg1), ('kg2', kg2)):
            r_vecs = {}
            for r, g in kg.triples.assign(r=kg.triples['r'].cat.codes).groupby(by='r'):
                gl = g.shape[0]
                vec = np.zeros(dim)
                for t in g['t'].cat.codes:
                    vec[t] += 1
                for t in g['h'].cat.codes:
                    vec[t] -= 1
                idx = np.arange(dim)[vec != 0]
                vec = vec / gl

                r_vecs[r] = ss.csr_matrix((vec[idx], (np.zeros(len(idx)), idx)), shape=(1, dim))
            print(len(r_vecs))

            h_vecs = {}
            h_cnt = {}
            for h, g in kg.triples.assign(h=kg.triples['h'].cat.codes).groupby(by='h'):
                gl = g.shape[0]
                t_cnt = {}
                for t in g['t'].cat.codes:
                    t_cnt[t] = t_cnt.get(t, 0) + 1
                k = list(t_cnt.keys())
                v = [t_cnt[_] for _ in k]
                vec = ss.csr_matrix((v, (np.zeros(len(k)), k)), shape=(1, dim))
                for r in g['r'].cat.codes:
                    vec -= r_vecs[r]
                h_vecs[h] = vec
                h_cnt[h] = gl
            
            for t, g in kg.triples.assign(t=kg.triples['t'].cat.codes).groupby(by='t'):
                gl = g.shape[0]
                t_cnt = {}
                for h in g['h'].cat.codes:
                    t_cnt[h] = t_cnt.get(h, 0) + 1
                k = list(t_cnt.keys())
                v = [t_cnt[_] for _ in k]
                vec = ss.csr_matrix((v, (np.zeros(len(k)), k)), shape=(1, dim))
                for r in g['r'].cat.codes:
                    vec += r_vecs[r]
                if t in h_vecs:
                    h_vecs[t] += vec
                else:
                    h_vecs[t] = vec
                    h_cnt[t] = 0
                h_vecs[t] = h_vecs[t] / (h_cnt[t] + gl)

            h_vecs_k = list(h_vecs.keys())
            data = np.concatenate([h_vecs[i].data for i in h_vecs_k])
            x = np.concatenate([[i] * h_vecs[i].data.shape[0] for i in h_vecs_k])
            y = np.concatenate([h_vecs[i].indices for i in h_vecs_k])
            H = ss.csr_matrix((data, (x, y)), shape=(dim, dim))
            ss.save_npz(task_dir / (name + '.npz'), H)


if __name__ == '__main__':
    # preprocess()
    train('zh_en')
