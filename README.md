# Fashion Retrieval System using OpenCLIP, FAISS and Qwen2.5-VL

## Overview

This project implements a multimodal fashion retrieval system capable of retrieving visually similar fashion products from a natural language query.

The system combines contrastive vision-language embeddings for efficient retrieval with a Vision Language Model (Qwen2.5-VL) for semantic reranking.

The retrieval pipeline is designed to be scalable and follows a two-stage architecture commonly used in production search systems.

---

## Features

- Text-to-image fashion retrieval
- OpenCLIP-based multimodal embeddings
- FAISS vector similarity search
- SQLite metadata filtering
- Qwen2.5-VL semantic reranking
- Interactive Gradio interface
- Hybrid retrieval pipeline

---

## Architecture

```

User Query
│
▼
OpenCLIP Text Encoder
│
▼
Metadata Filter (SQLite)
│
▼
FAISS Vector Search
│
▼
Top-N Candidates
│
▼
Qwen2.5-VL Reranker
│
▼
Top-K Results

```

---

## Project Structure

```

project/
│
├── dataset/
│
├── demo/
│   └── app.py
│
├── models/
│   ├── openclip.py
│   └── qwen_vl.py
│
├── retriever/
│   ├── search.py
│   └── rerank.py
│
├── outputs/
│   ├── embeddings/
│   ├── faiss/
│   └── metadata/
│
├── requirements.txt
└── README.md

```

---

## Dataset

DeepFashion Multimodal Dataset

Contains approximately **42,500** fashion products with:

- Product image
- Category hierarchy
- Product description
- Metadata

---

## Technologies Used

| Component | Technology |
|----------|-------------|
| Vision Encoder | OpenCLIP ViT-H-14 |
| Vision Language Model | Qwen2.5-VL-7B-Instruct |
| Vector Database | FAISS |
| Metadata Store | SQLite |
| Dataset | Hugging Face Datasets |
| Interface | Gradio |
| Language | Python |

---

## Retrieval Pipeline

### Stage 1: Metadata Filtering

The user query is analyzed to identify simple attributes such as:

- Gender
- Garment type
- Clothing category

Relevant candidates are selected using SQLite.

---

### Stage 2: Vector Retrieval

The text query is encoded using OpenCLIP.

Similarity search is performed using FAISS to retrieve the most relevant candidate images.

---

### Stage 3: Vision-Language Reranking

The top retrieved candidates are passed to Qwen2.5-VL.

Qwen analyzes each image together with the user query and assigns a semantic relevance score.

The final ranking is based on these scores.

---

## Example Queries

- women tops
- blue denim jeans
- black formal shirt
- red floral dress
- white sneakers
- floral summer dress

---

## Running the Project

### Create environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Launch application

```bash
python -m demo.app
```

---

## Sample Workflow

Input:

```

women blue denim jacket

```

Pipeline:

```

Query
↓

OpenCLIP embedding
↓

Metadata filtering
↓

FAISS search
↓

Top candidates
↓

Qwen reranking
↓

Final ranked results

```

---

## Performance

Dataset size:

- 42,537 products

Embedding model:

- OpenCLIP ViT-H-14

Similarity search:

- FAISS Inner Product Search

Semantic reranking:

- Qwen2.5-VL-7B

---

## Limitations

- Qwen reranking introduces additional inference latency because each retrieved candidate is analyzed individually.
- Metadata filtering currently relies on rule-based keyword matching.
- The reranking stage is sequential and can be optimized using batching or asynchronous inference.

---

## Future Improvements

- Batch inference for Qwen reranking
- Attribute extraction using LLM-generated structured queries
- Color and material-aware filtering
- User feedback-based relevance learning
- Distributed FAISS index for larger datasets
- Deployment with FastAPI and Docker
- GPU-optimized inference using TensorRT or vLLM

---

## Results

The hybrid retrieval system combines fast vector search with semantic reranking, producing more contextually relevant fashion recommendations than vector similarity search alone while maintaining a modular and extensible architecture.

---

## Author

**Mitali Raj**
