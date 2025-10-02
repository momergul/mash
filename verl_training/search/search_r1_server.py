# File: search_r1_server.py
# -------------------------
# Server separate from the main script for responding to retrieval requests

import json
import os
import warnings
from typing import List, Dict, Optional
import argparse

import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm
import datasets

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from verl_training.search.search_r1_searcher import SearchR1Config, DenseRetriever

parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
parser.add_argument("--index_path", type=str,
                    default="knowledge_sources/e5_Flat.index",
                    help="Corpus indexing file.")
parser.add_argument("--corpus_path", type=str,
                    default="knowledge_sources/wiki-18.jsonl",
                    help="Local corpus file.")
parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Name of the retriever model.")
parser.add_argument("--retrieval_batch_size", type=int, default=128, help="Batch size for retrieval.")
parser.add_argument("--faiss_gpu", action='store_true', help="Whether to use FAISS GPU or not.")
parser.add_argument("--listen_port", type=int, default=8000)
args = parser.parse_args()

class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False

app = FastAPI()

config = SearchR1Config(
    retrieval_method = "e5",  # or "dense"
    index_path=args.index_path,
    corpus_path=args.corpus_path,
    retrieval_topk=args.topk,
    faiss_gpu=args.faiss_gpu,
    retrieval_model_path=args.retriever_model,
    retrieval_pooling_method="mean",
    retrieval_query_max_length=256,
    retrieval_use_fp16=True,
    retrieval_batch_size=args.retrieval_batch_size,
)

retriever = DenseRetriever(config)

@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }
    """
    if not request.topk:
        request.topk = config.retrieval_topk  # fallback to default

    # Perform batch retrieval
    results, scores = retriever.batch_search(
        query_list=request.queries,
        num=request.topk,
        return_score=request.return_scores
    )
    
    # Format response
    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            # If scores are returned, combine them with results
            combined = []
            for doc, score in zip(single_result, scores[i]):
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append(single_result)
    return {"result": resp}

if __name__ == "__main__":
    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=args.listen_port)


