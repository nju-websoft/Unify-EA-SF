# [What Makes Entities Similar? A Similarity Flooding Perspective for Multi-sourced Knowledge Graph Embeddings](https://proceedings.mlr.press/v202/sun23d/sun23d.pdf)

> Joint representation learning over multi-sourced knowledge graphs (KGs) yields transferable and expressive embeddings that improve downstream tasks. Entity alignment (EA) is a critical step in this process. Despite recent considerable research progress in embedding-based EA, how it works remains to be explored. In this paper, we provide a similarity flooding perspective to explain existing translation-based and aggregation-based EA models. We prove that the embedding learning process of these models actually seeks a fixpoint of pairwise similarities between entities. We also provide experimental evidence to support our theoretical analysis. We propose two simple but effective methods inspired by the fixpoint computation in similarity flooding, and demonstrate their effectiveness on benchmark datasets. Our work bridges the gap between recent embedding-based models and the conventional similarity flooding algorithm. It would improve our understanding of and increase our faith in embedding-based EA.

*** UPDATE *** 

* Aug. 15, 2023: We improve and update the implementation for *similarity flooding via entity compositions*, which can achieve slightly better results than those reported in our ICML paper.

## Dependencies

The implementation for *similarity flooding via entity compositions* is dependent on:

* **python3** (*tested with v3.10.12*)
* **pandas** (*tested with v1.5.3*)
* **numpy** (*tested with v1.22.3*)
* **scipy** (*tested with v1.7.3*)
* **cudatoolkit** (*tested with v10.2.89*)
* **cupy** (*tested with v12.1.0*)

The implementation for *AliNet + SPA* is dependent on [AliNet](https://github.com/nju-websoft/AliNet).

## Running code

### Similarity flooding via entity compositions

To run TransFlood on DBP15K ZH-EN, please execute the following script:

```bash
python embed_sf.py --input ./dataset/dbp15k/zh_en/ --model TransE
```

To run GCNFlood on DBP15K ZH-EN, please execute:

```bash
python embed_sf.py --input ./dataset/dbp15k/zh_en/ --model GCN
```

### AliNet with self-propagation

To run AliNet + SPA on DBP15K ZH-EN, please enter the folder "alinet_spa" and execute the following script:

```bash
python main.py --input ../dataset/dbp15k/zh_en/mtranse/0_3/ --alpha 0.1
```

To disable self-propagation, i.e., to run the original AliNet, please execute:

```bash
python main.py --input ../dataset/dbp15k/zh_en/mtranse/0_3/ --alpha 0.0
```

> If you have any difficulty or question in running code and reproducing experimental results, please email to zqsun.nju@gmail.com or jchuang.nju@gmail.com.

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
