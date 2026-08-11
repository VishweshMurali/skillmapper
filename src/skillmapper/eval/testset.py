from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QAItem:
    question: str
    doc_id: str  # scope retrieval to this one resume/job for a clean signal
    reference: str


TESTSET: list[QAItem] = [
    QAItem(
        question="What certifications does the candidate hold?",
        doc_id="resume-01-senior-swe-backend",
        reference="AWS Certified Solutions Architect - Associate (2022) and Certified Kubernetes Administrator (CKA, 2023).",
    ),
    QAItem(
        question="What is the candidate's educational background?",
        doc_id="resume-01-senior-swe-backend",
        reference="A B.S. in Computer Science from the University of Illinois Urbana-Champaign, 2017.",
    ),
    QAItem(
        question="What programming languages and technologies does the candidate know?",
        doc_id="resume-01-senior-swe-backend",
        reference="Python, Go, PostgreSQL, Kafka, Redis, gRPC, Docker, Kubernetes, Terraform, AWS, CI/CD, and distributed tracing.",
    ),
    QAItem(
        question="What did the candidate do at Northwind Payments?",
        doc_id="resume-01-senior-swe-backend",
        reference="Led the redesign of the core ledger service into Go microservices, reduced p99 checkout latency from 900ms to 210ms, and owns the on-call rotation for the payments platform.",
    ),
    QAItem(
        question="What degree does the candidate have and from where?",
        doc_id="resume-03-data-scientist-senior",
        reference="A Ph.D. in Statistics from the University of Wisconsin-Madison (2018) and a B.S. in Applied Mathematics from the University of Michigan (2013).",
    ),
    QAItem(
        question="What machine learning skills does the candidate list?",
        doc_id="resume-03-data-scientist-senior",
        reference="Python, R, SQL, PyTorch, LightGBM, scikit-learn, spaCy, Spark, Snowflake, Airflow, MLflow, A/B testing, statistical modeling, NLP, and time-series forecasting.",
    ),
    QAItem(
        question="What certification does the candidate hold?",
        doc_id="resume-03-data-scientist-senior",
        reference="Google Cloud Professional Machine Learning Engineer (2021).",
    ),
    QAItem(
        question="What did the candidate build at Fenwick AI?",
        doc_id="resume-05-ml-engineer",
        reference="A retrieval-augmented generation pipeline for a customer-support copilot using hybrid dense/BM25 retrieval over ChromaDB and a fine-tuned Llama model served via vLLM, with a RAGAS-based evaluation harness gating releases.",
    ),
    QAItem(
        question="What is the candidate's educational background?",
        doc_id="resume-05-ml-engineer",
        reference="An M.S. in Computer Science with a Machine Learning track from Georgia Institute of Technology, 2019.",
    ),
    QAItem(
        question="What are the minimum requirements for this backend engineering role?",
        doc_id="job-01-swe-backend",
        reference="At least 5 years of backend software engineering experience, strong proficiency in Python or Go, distributed systems experience (Kafka, Redis, gRPC/REST), PostgreSQL at scale, and containerized deployment experience.",
    ),
    QAItem(
        question="What responsibilities does this role involve?",
        doc_id="job-02-data-scientist",
        reference="Building, validating, and shipping ML models into production, designing and analyzing A/B tests, productionizing models with engineering, and communicating results to non-technical stakeholders.",
    ),
    QAItem(
        question="What qualifications are considered a plus for this role?",
        doc_id="job-03-ml-engineer",
        reference="Experience with RAGAS-style evaluation frameworks, model-serving infrastructure like vLLM or ONNX Runtime, MLOps tooling, and open-source ML infrastructure contributions.",
    ),
]
