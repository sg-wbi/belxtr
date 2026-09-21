# Biomedical Entity Linking via Contextualized Token Retrieval

## Results

We store all predictions and gold identifiers in `data`.

To reproduce the results from the paper create a virtual env and install:

```bash
pip install pandas
pip install bioc
```

Performance on BELB:

```bash
python evaluate.py --target belb
```

Performance on partial BELB with refined test set:

```bash
python evaluate.py --target belb-refined-retrieve
```

Performance on partial BELB with LLM-reranking:

```bash
python evaluate.py --target belb-refined-rerank
```

Performance on BioRED (end-to-end document-level):

```bash
python evaluate.py --target biored
```

## Train

Install [`torch`](https://pytorch.org/get-started)) (hardware-dependent)

Install other deps:


```bash
# datasets
pip install git+https://github.com/sg-wbi/belb.git

pip install -e .
```

Run (see `./config/` for hyperparameters):

```bash
# for multi-GPU
# accelerate launch train.py 
python train.py \
    run=ncbi-disease \
    hps.train=ncbi-disease.ctd-diseases \
    hps.test=ncbi-disease.ctd-diseases \
    hps.kb_entity_based=false \
#   hps.qe=false \ # add [MASK]
#   hps.qe_train=false  \ train [MASK]

```
