# Negative Sampling for NER

*Unlabeled entity problem* is prevalent in many NER scenarios (e.g., weakly supervised NER). 
Our [paper](https://openreview.net/forum?id=5jRVa89sZk) in ICLR-2021 proposes using negative sampling for solving this issue.
This repo. contains the implementation of our approach.

Note that this is not an officially supported Tencent product.

## Preparation

Two steps. Firstly, reformulating the NER data and move it into a new folder named "dataset". 
The folder contains {train, dev, test}.json. 
Each JSON file is a list of dicts. See the following case:
```
[ 
 {
  "sentence": "['Somerset', '83', 'and', '174', '(', 'P.', 'Simmons', '4-38', ')', ',', 'Leicestershire', '296', '.']",
  "labeled entities": "[(0, 0, 'ORG'), (5, 6, 'PER'), (10, 10, 'ORG')]",
 },
 {
  "sentence": "['Leicestershire', '22', 'points', ',', 'Somerset', '4', '.']",
  "labeled entities": "[(0, 0, 'ORG'), (4, 4, 'ORG')]",
 }
]
```

Secondly, pretrained LM (i.e., [BERT](https://www.aclweb.org/anthology/N19-1423/)) and [eval. script](https://www.clips.uantwerpen.be/conll2000/chunking/conlleval.txt). 
Create a dir. named "resource" and arrange them as
- resource
    - bert-base-cased
        - model.pt
        - vocab.txt
    - conlleval.pl

Note that the files in BERT need to be renamed as above.

## Training and Test
```
CUDA_VISIBLE_DEVICES=0 python main.py -dd dataset -cd save -rd resource
```

## Citation
```
@inproceedings{li2021empirical,
    title={Empirical Analysis of Unlabeled Entity Problem in Named Entity Recognition},
    author={Yangming Li and lemao liu and Shuming Shi},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=5jRVa89sZk}
}
```
