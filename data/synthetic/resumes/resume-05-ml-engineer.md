# Wei Chen

## Summary

ML engineer with 5 years of experience taking models from notebook to production, with a focus on serving infrastructure, retrieval-augmented generation, and MLOps tooling. Comfortable owning a system end to end, from training pipeline to low-latency inference.

## Experience

**ML Engineer, Fenwick AI** (2022 - Present)
Built and operate a retrieval-augmented generation pipeline for a customer-support copilot, using a hybrid dense/BM25 retriever over a ChromaDB vector store and a fine-tuned Llama model served via vLLM. Reduced end-to-end response latency from 6.2s to 2.4s through embedding caching and batched inference. Set up an evaluation harness using RAGAS to track faithfulness and context precision on every model release, gating deploys on a regression threshold.

**Machine Learning Engineer, Solstice Robotics** (2020 - 2022)
Deployed computer-vision defect-detection models (PyTorch, ONNX Runtime) to edge devices, building the model-versioning and rollback system used across the fleet. Built a feature-flagged canary deployment process for gradually rolling out new model versions.

**Software Engineer (ML infra), Solstice Robotics** (2019 - 2020)
Built the training-data labeling pipeline and internal annotation tool used by the computer vision team.

## Skills

Python, PyTorch, ONNX, vLLM, LangChain, LangGraph, ChromaDB, RAGAS, Docker, Kubernetes, MLflow, feature engineering, model serving, prompt engineering, hybrid retrieval (BM25 + dense embeddings).

## Education

M.S. in Computer Science (Machine Learning track), Georgia Institute of Technology, 2019.

## Projects

Open-sourced a small library for reciprocal-rank-fusion of BM25 and dense retriever results, used internally at two former employers.

## Certifications

NVIDIA Deep Learning Institute — Building Transformer-Based NLP Applications (2023).
