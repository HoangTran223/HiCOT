# HiCOT: Improving Neural Topic Models via Optimal Transport and Contrastive Learning


## Setup
1. Install the following libraries
    ```bash
    numpy==1.26.4
    torch==2.4.0
    torchvision==0.19.0
    torchaudio==2.4.0
    torch_kmeans==0.2.0
    pytorch==2.2.0
    scipy==1.10
    sentence_transformers==2.2.2
    gensim==4.2.0
    ```
2. Install java
3. Download [this java jar](https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/palmetto-0.1.0-jar-with-dependencies.jar) to `./evaluations/palmetto.jar`.
4. Download and extract [this processed Wikipedia corpus](https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/Wikipedia_bd.zip) to `./datasets/wikipedia/` as an external reference corpus.


## Usage
To run HiCOT, execute the following command:

```bash
python main.py --model HiCOT --dataset AGNews --num_topics 50 --dropout 0.2 --seed 1 --device cuda --lr_step_size 125 --batch_size 128 --beta_temp 0.1 --epochs 400 --lr 0.002 --use_pretrainWE 
--weight_ECR 10 --alpha_ECR 20 --weight_loss_TP 2 --alpha_TP 2 --weight_loss_DT 1 --weight_loss_CLC 1 --weight_loss_CLT 1 --threshold_epoch 100 --threshold_cluster 100 --method_CL HAC --metric_CL euclidean
```


## Options:

- **Datasets**: `20NG`, `AGNews`, `IMDB`, `SearchSnippets`, `GoogleNews`
- **Hierarchical Clustering Algorithms:**  
  - Set `method_CL` to one of the following: `HAC`, `HDBSCAN` 
**Distance Metrics for Contrastive Learning:**  
  - Set `metric_CL` to one of the following: `euclidean`, `cosine` 

## Acknowledgement
Some part of this implementation is based on [TopMost](https://github.com/BobXWu/TopMost). We also utilizes [Palmetto](https://github.com/dice-group/Palmetto) for evaluating topic coherence.
