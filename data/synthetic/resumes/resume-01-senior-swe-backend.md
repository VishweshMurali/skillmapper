# Priya Natarajan

## Summary

Senior backend software engineer with 8 years of experience building distributed systems that process high-throughput transactional data. Specializes in Python and Go microservices, event-driven architectures, and reliability engineering. Enjoys mentoring junior engineers and driving on-call maturity.

## Experience

**Senior Software Engineer, Northwind Payments** (2021 - Present)
Led the redesign of the core ledger service, migrating a monolithic Django application into a set of Go microservices communicating over gRPC and Kafka. Reduced p99 checkout latency from 900ms to 210ms by introducing read-replica routing and a Redis-backed idempotency cache. Own the on-call rotation for the payments platform and authored the incident response runbook adopted team-wide.

**Software Engineer II, Northwind Payments** (2019 - 2021)
Built the fraud-signal ingestion pipeline using Kafka Streams and PostgreSQL, processing roughly 40 million events per day. Introduced contract testing between services using Pact, cutting integration-bug escape rate by half. Partnered with the data science team to expose real-time features for the fraud-scoring model via a low-latency gRPC API.

**Software Engineer, Alderbrook Logistics** (2017 - 2019)
Developed REST APIs in Python/Flask for warehouse routing optimization. Containerized the deployment pipeline with Docker and Kubernetes, reducing deployment time from 45 minutes to under 5 minutes. Wrote the initial Terraform modules that became the standard for provisioning the team's AWS infrastructure.

## Skills

Python, Go, PostgreSQL, Kafka, Redis, gRPC, Docker, Kubernetes, Terraform, AWS (ECS, RDS, S3, Lambda), CI/CD (GitHub Actions), system design, distributed tracing (OpenTelemetry), on-call/incident management.

## Education

B.S. in Computer Science, University of Illinois Urbana-Champaign, 2017.

## Certifications

AWS Certified Solutions Architect - Associate (2022). Certified Kubernetes Administrator (CKA, 2023).
