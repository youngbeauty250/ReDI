# ReDI Retrieval Experiments

This repository contains code and data for running retrieval experiments with **query decomposition + interpretation (ReDI)**.

## Code Overview

- **`retrievers.py`**  
  Provides retrieval functions using decomposed and interpreted queries:  
  - `retrieval_bm25_fusion_desc` — sparse retrieval with BM25 fusion.  
  - `retrieval_sbert_bge_fusion_desc` — dense retrieval with SBERT/BGE fusion.  

## Data

- **`ReDI_bm25_reason.tar.gz`**  
  Contains queries for sparse retrieval (BM25-based).  

- **`ReDI_dense_reason.tar.gz`**  
  Contains queries for dense retrieval (SBERT/BGE-based).  

## Example Command

You can run retrieval with the following command:

```bash
python run.py \
  --task biology \
  --model bm25_fusion_desc \
  --reasoning ReDI_bm25 \
  --output_dir /path/to/output
