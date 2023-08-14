# [What Makes Entities Similar? A Similarity Flooding Perspective for Multi-sourced Knowledge Graph Embeddings](https://proceedings.mlr.press/v202/sun23d/sun23d.pdf)

> Joint representation learning over multi-sourced knowledge graphs (KGs) yields transferable and expressive embeddings that improve downstream tasks. Entity alignment (EA) is a critical step in this process. Despite recent considerable research progress in embedding-based EA, how it works remains to be explored. In this paper, we provide a similarity flooding perspective to explain existing translation-based and aggregation-based EA models. We prove that the embedding learning process of these models actually seeks a fixpoint of pairwise similarities between entities. We also provide experimental evidence to support our theoretical analysis. We propose two simple but effective methods inspired by the fixpoint computation in similarity flooding, and demonstrate their effectiveness on benchmark datasets. Our work bridges the gap between recent embedding-based models and the conventional similarity flooding algorithm. It would improve our understanding of and increase our faith in embedding-based EA.

## Dependencies

The code for *embedding-based similarity flooding* is dependent on:

* python3 (*tested with v3.10.12*)
* pandas (*tested with v1.5.3*)
* numpy (*tested with v1.22.3*)
* scipy (*tested with v1.7.3*)
* cudatoolkit (*tested with v10.2.89*)
* cupy (*tested with v12.1.0*)

## Running code

First, we need to execute the *preprocess()* function to process the DBP15K dataset.

Then, for example, to run TransFlood on *ZH-EN*, please execute function *train('zh_en')*.

## Citation
If you find the work helpful, please kindly cite the following paper:
```bibtex
@inproceedings{pmlr-v202-sun23d,
  author       = {Zequn Sun and
                  Jiacheng Huang and
                  Xiaozhou Xu and
                  Qijin Chen and
                  Weijun Ren and
                  Wei Hu},
  title        = {What Makes Entities Similar? A Similarity Flooding Perspective for Multi-sourced Knowledge Graph Embeddings},
  booktitle    = {Proceedings of the 40th International Conference on Machine Learning},
  pages        = {32875--32885},
  year         = {2023},
}
```
