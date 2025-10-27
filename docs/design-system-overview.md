# Global E-commerce Policy Intelligence Platform - Architecture Design

## Real-World Scenario

**Platform Purpose**: A government agency needs to monitor e-commerce competitiveness across 100+ countries, forecast labor market trends, analyze ESG compliance, and simulate policy impacts using AI-powered insights.

**Business Requirements**:

- Ingest data from 50+ heterogeneous sources (APIs, CSVs, PDFs, real-time feeds)
- Calculate composite indices with versioned methodologies
- AI-powered document analysis and policy recommendations
- Interactive dashboards for policymakers
- Reproducible, auditable results
- Multi-tenant support (different government agencies)
- 99.9% uptime SLA

----

## High-Level Architecture Overview

```mermaid
---
title: Global E-commerce Policy Intelligence Platform - High Level Design
---
graph TD
    subgraph A [CLIENT LAYER]
        A1[Next.js Web App ISR]
        A2[Mobile App]
        A3[Third-party API Consumers]
    end
    subgraph B [EDGE & API GATEWAY LAYER]
        B1[Azure Front Door/Cloud CDN]
        B2[API Management: Rate Limiting, Auth]
    end
    subgraph C [APPLICATION LAYER]
        C1[Scoring API: Azure Functions/App Service, .NET Core]
        C2[Analytics API: Azure Functions/App Service, Python]
        C3[AI/RAG API: Azure Functions, Azure OpenAPI, Python]
    end
    subgraph D [ORCHESTRATION LAYER]
        D1[n8n Workflows]
        D2[Event-driven: Azure Functions]
    end
    subgraph E [DATA LAYER]
        E1[Data Ingestion: Azure Data Factory, Logic Apps]
        E2[Data Storage: Azure Data Lake Gen2, Blob Storage]
        E3[Data Processing: Azure Databricks, Synapse Analytics, PySpark]
        E4[Metadata & Versioning: Azure SQL Database, PostgreSQL, MSSQL]
    end

    A --> B --> C --> D --> E
```

----

## Detailed Component Breakdown

### Client / Frontend Layer

#### React 18 + Next.js 14 (App Router with ISR)

- **Purpose**: Provide an interactive web interface for policymakers to view dashboards, reports, and AI-generated insights.
  
- **Rationale**: React is widely adopted with a rich ecosystem. Next.js provides server-side rendering and Incremental Static Regeneration (ISR) for performance and SEO.
  
- **Key Features**:
  - ISR for up-to-date dashboards
  - Responsive design for desktop and mobile
  - Integration with third-party visualization libraries (e.g., D3.js, Chart.js)
  - Reusable components for charts, tables, and forms
  - Reduce client-side bundle size by leveraging Next.js features
  
- **Responsibilities**:
  - Render dashboards and reports
  - Handle user authentication and authorization
  - Consume APIs from the Application Layer
  - Provide user settings and preferences

Architecture Patterns: Feature based folder structure + shared components library

```text
/app
  /(dashboard)      # Main policy dashboards
    /[country]      # Dynamic routes
  /(admin)          # Admin tools
  /api              # Next.js API routes (BFF pattern)
/components
  /ui               # Reusable buttons, cards, tables
  /charts           # Chart.js/Recharts wrappers
/lib
  /api-client       # Typed API calls to backend
  /auth             # NextAuth.js integration
/styles
  /globals.css      # Global styles
/tests
  /unit             # Unit tests
  /integration      # Integration tests
```

**Key Technologies**:

- **TypeScript**: Type safety
- **Tailwind CSS**: Utility-first styling
- **shadcn/ui**: Component library (Radix UI primitives)
- **TanStack Query (React Query)**: Server state management
- **Zustand**: Client state management (lightweight)
- **Recharts/Chart.js**: Data visualization
**NextAuth.js**: Authentication integration with Azure AD/Google

**ISR Strategy**:

- Index pages (e.g., /countries) revalidate every 5 minutes
- Dynamic pages (e.g., /countries/indonesia) revalidate on-demand via webhook
- Static export for public reports

**Sample Flow**:

```typescript
// app/countries/[code]/page.tsx
export const revalidate = 300; // ISR: 5 minutes

export default async function CountryPage({ params }) {
  const data = await fetchCountryScore(params.code); // Server-side fetch
  return <ScoreDetails data={data} />;
}
```

### Backend / API Layer

#### C# / .NET Core 8.0 - High-Performance APIs

**Purpose**: Provide robust, scalable APIs for scoring calculations, analytics, and AI-powered insights.

**Rationale**: .NET Core offers high performance, strong typing, enterprise pattern, and seamless integration with Azure services.

**Key Features**:

- RESTful API design with versioning
- Swagger/OpenAPI documentation
- JWT-based authentication and role-based access control
- Integration with Azure Functions for serverless compute
- Comprehensive logging and monitoring with Application Insights

**Responsibilities**:

- **Scoring Service**: Calculate composite ecommerce competitiveness indices
- **Authentication/Authorization**: OAuth2, JWT validation, multi-tenant context
- **Rate Limiting & Circuit Breakers**: Polly for resilience
- **Data Contracts Validation**: FluentValidation for input schemas

**Architecture Patterns**: Clean Architecture + CQRS + Mediator Pattern

```text
/src
  /Application
    /Commands
    /Queries
  /Domain
    /Entities
    /ValueObjects
  /Infrastructure
    /Persistence
    /Messaging
  /WebApi
    /Controllers
    /DTOs
    /Middlewares
/tests
  /Unit
  /Integration
```

**Key Technologies**:

- **ASP.NET Core Minimal APIs**: Lightweight endpoints
- **Entity Framework Core**: ORM for MSSQL/Postgres
- **Dapper**: High-performance queries for read-heavy operations
- **MediatR**: Command/Query separation
- **Polly**: Retry, circuit breaker, timeout policies
- **Serilog**: Structured logging to Azure Monitor/Stackdriver
- **OpenTelemetry**: Distributed tracing

**Sample Use Case**:

- `/v1/score/{country}?year=2025` endpoint
- Fetches indicators from Postgres, applies YAML config weights
- Returns JSON with reproducibility hash
- Caches in Redis for 10 minutes

#### Python 3.11+ - ML/AI & Data Processing

**Purpose**: Data science, ML pipelines, AI orchestration, and flexible scripting.

**Rationale**: Python's rich ecosystem for data science (Pandas, Scikit-learn, TensorFlow) and ease of prototyping complex algorithms.

**Key Features**:

- **FastAPI**: Async API framework (alternative to Flask)
- **Pandas/Polars**: Data manipulation
- **scikit-learn**: ML models (regression, classification)
- **Prophet/statsmodels**: Time-series forecasting
- **Great Expectations**: Data validation
- **SQLAlchemy**: ORM for Postgres/MSSQL
- **Pydantic**: Data validation and serialization

**Sample Use Case**:

- `/v1/forecast/labor-gap` endpoint
- Loads historical data from Postgres
- Runs ARIMA forecast via scikit-learn
- Returns prediction intervals + confidence scores

### **3. DATABASE LAYER**

#### **MSSQL Server 2022 (Azure SQL / Cloud SQL)**

**Purpose**: Transactional data requiring ACID compliance and enterprise features.

**Use Cases**:

- **User accounts & tenants**: Authentication, RBAC, audit logs
- **Configuration data**: Methodology versions, feature flags
- **Transactional reports**: Immutable snapshot tables for compliance

**Schema Design**:

- Normalized tables (3NF) for consistency
- Temporal tables for change tracking
- Columnstore indexes for analytical queries
- Always Encrypted for PII (e.g., user emails)

**Sample Tables**:

```sql
CREATE TABLE [dbo].[Users] (
    [Id] UNIQUEIDENTIFIER PRIMARY KEY,
    [Email] NVARCHAR(255) NOT NULL,
    [TenantId] UNIQUEIDENTIFIER,
    [CreatedAt] DATETIME2 DEFAULT GETUTCDATE()
);

CREATE TABLE [dbo].[ScoreSnapshots] (
    [Id] BIGINT IDENTITY PRIMARY KEY,
    [CountryCode] CHAR(3),
    [Year] INT,
    [Score] DECIMAL(5,2),
    [MethodologyVersion] VARCHAR(20),
    [ComputedAt] DATETIME2
);
```

#### **PostgreSQL 16 (Azure Flexible Server / Cloud SQL)**

**Purpose**: Primary analytical database for time-series, panel data, and vector embeddings.

**Use Cases**:

- **Indicator data**: Country-year panel data (GDP, trade, internet penetration)
- **ML feature store**: Pre-computed features for models
- **Vector embeddings**: pgvector extension for RAG retrieval
- **Time-series data**: TimescaleDB extension for hyper-optimized queries

**Schema Design**:

- Partitioning by year for large tables
- Materialized views for aggregate queries
- Indexes on (country_code, year) for fast lookups

**Sample Schema**:

```sql
CREATE TABLE indicators (
    id SERIAL PRIMARY KEY,
    country_code CHAR(3),
    year INT,
    indicator_key VARCHAR(50),
    value NUMERIC,
    source VARCHAR(100),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(country_code, year, indicator_key)
);

-- pgvector for RAG
CREATE EXTENSION vector;
CREATE TABLE document_embeddings (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(100),
    chunk_text TEXT,
    embedding vector(1536),  -- OpenAI embedding size
    metadata JSONB
);
CREATE INDEX ON document_embeddings USING ivfflat (embedding vector_cosine_ops);
```

#### **Elasticsearch 8.x (Azure Managed / GCP Managed)**

**Purpose**: Full-text search, log aggregation, and hybrid retrieval (BM25 + vector).

**Use Cases**:

- **Policy document search**: Index PDFs, Word docs, web pages
- **Hybrid RAG retrieval**: Combine keyword (BM25) + semantic (vector) search
- **Observability**: Centralized logging (ELK stack)
- **Real-time analytics**: Dashboard metrics aggregation

**Index Design**:

```json
{
  "mappings": {
    "properties": {
      "document_id": { "type": "keyword" },
      "title": { "type": "text", "analyzer": "standard" },
      "content": { "type": "text", "analyzer": "english" },
      "embedding": { "type": "dense_vector", "dims": 1536 },
      "country": { "type": "keyword" },
      "published_date": { "type": "date" },
      "metadata": { "type": "object" }
    }
  }
}
```

**Query Strategy**:

- BM25 for keyword precision (e.g., "labor regulation Indonesia")
- kNN for semantic similarity (user question → embedding)
- RRF (Reciprocal Rank Fusion) to combine both scores

#### **MongoDB 7.x (Azure Cosmos DB / MongoDB Atlas)**

**Purpose**: Semi-structured data, rapid prototyping, event sourcing.

**Use Cases**:

- **Data ingestion staging**: Raw API responses before normalization
- **Configuration store**: YAML methodology configs, feature flags
- **Event logs**: Workflow execution history (N8n audit trail)
- **Cache layer**: Session data, user preferences

**Schema Design**:

```json
{
  "_id": "esg_score_amazon_2025",
  "company_id": "amazon",
  "score_date": "2025-01-15",
  "pillars": {
    "environmental": 78.5,
    "social": 82.0,
    "governance": 90.2
  },
  "sources": ["bloomberg_esg", "cdp_report"],
  "methodology_version": "2025.10",
  "metadata": {
    "computed_by": "scoring-service-v2",
    "hash": "a3f5d9..."
  }
}
```

#### Blob Storage / Data Lake

**Purpose**: Raw and processed data storage for large files.

**Use Cases**:

- Raw data dumps (CSV, JSON, XML)
- Processed Parquet files for analytics
- Document storage (PDFs, Word docs for RAG)
- Model artifacts (ML models, vector indexes)
- Audit logs and compliance records

**Technologies**:

- **Azure Data Lake Gen2**: Hierarchical namespace, optimized for big data
- **Blob Storage**: Cost-effective object storage
- **GCP Cloud Storage**: Scalable object storage with lifecycle policies
- **Data Formats**: Parquet for analytics, JSON/CSV for interchange, PDF/DOCX for documents

### **4. CLOUD INFRASTRUCTURE**

#### **Azure (Primary)**

**Services Used**:

|Service|Purpose|
|---|---|
|**Azure Kubernetes Service (AKS)**|Container orchestration for .NET/Python microservices|
|**Azure API Management**|API gateway, rate limiting, OAuth2 validation|
|**Azure Front Door**|Global CDN, WAF, SSL termination|
|**Azure SQL Database**|Managed MSSQL with auto-failover|
|**Azure Database for PostgreSQL Flexible**|Managed Postgres with pgvector|
|**Azure Cosmos DB (MongoDB API)**|Globally distributed MongoDB|
|**Azure Cognitive Search**|Managed Elasticsearch alternative|
|**Azure Container Registry**|Private Docker image registry|
|**Azure Key Vault**|Secret management, certificate rotation|
|**Azure Monitor + App Insights**|Observability, tracing, metrics|
|**Azure Service Bus**|Event-driven messaging (pub/sub)|
|**Azure Functions**|Serverless compute for event handlers|
|**Azure Blob Storage**|Data lake (raw CSVs, PDFs)|

**Network Architecture**:

- VNet with subnets (web, app, data)
- Private endpoints for databases (no public IPs)
- Network Security Groups (NSGs) for firewall rules
- Azure Bastion for secure SSH/RDP

#### **Google Cloud Platform (Alternative/Hybrid)**

**Services Used**:

|Service|Purpose|
|---|---|
|**Google Kubernetes Engine (GKE)**|Container orchestration|
|**Cloud Run**|Serverless container deployment|
|**Cloud API Gateway**|API management|
|**Cloud CDN**|Content delivery|
|**Cloud SQL (PostgreSQL/MSSQL)**|Managed databases|
|**BigQuery**|Data warehouse for large-scale analytics|
|**Vertex AI**|Managed ML platform (scikit-learn, custom models)|
|**Cloud Storage**|Object storage (data lake)|
|**Secret Manager**|Secret management|
|**Cloud Monitoring (Stackdriver)**|Observability|

### **5. DEV TOOLS & CI/CD**

#### **Git (GitHub/Azure DevOps)**

**Branching Strategy**: GitFlow

- `main` → production
- `develop` → staging
- `feature/*` → development branches
- `hotfix/*` → emergency patches

**Monorepo Structure**:

```text
/services
  /scoring-api          # .NET project
  /analytics-api        # Python FastAPI
  /rag-service          # Python LangChain
/frontend
  /web                  # Next.js app
/infrastructure
  /terraform            # IaC modules
  /ansible              # Configuration management
/pipelines              # CI/CD workflows
```

#### **CI/CD Pipelines (GitHub Actions / Azure Pipelines)**

**Pipeline Stages**:

1. **Build**:
    - Restore dependencies (NuGet, pip, npm)
    - Compile code
    - Run linters (ESLint, Pylint, dotnet format)
2. **Test**:
    - Unit tests (xUnit, pytest, Jest)
    - Integration tests (Testcontainers for databases)
    - Code coverage > 80%
3. **Security Scan**:
    - SAST (Snyk, SonarQube)
    - Dependency scanning (Dependabot)
    - Container scanning (Trivy)
4. **Build & Push Images**:
    - Docker build multi-stage (optimize layers)
    - Push to Azure Container Registry / Artifact Registry
    - Tag with Git SHA + semantic version
5. **Deploy**:
    - **Dev**: Auto-deploy on merge to `develop`
    - **Staging**: Manual approval
    - **Production**: Blue-green deployment via Terraform
6. **Post-Deploy**:
    - Smoke tests
    - Synthetic monitoring
    - Rollback on failure

**Sample GitHub Actions Workflow**:

```yaml
name: Deploy Scoring API

on:
  push:
    branches: [main]
    paths:
      - 'services/scoring-api/**'

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up .NET
        uses: actions/setup-dotnet@v3
        with:
          dotnet-version: '8.0.x'
      
      - name: Run tests
        run: dotnet test --configuration Release
      
      - name: Build Docker image
        run: docker build -t scoring-api:${{ github.sha }} .
      
      - name: Push to ACR
        run: |
          az acr login --name myregistry
          docker push myregistry.azurecr.io/scoring-api:${{ github.sha }}
      
      - name: Deploy to AKS
        run: |
          kubectl set image deployment/scoring-api \
            scoring-api=myregistry.azurecr.io/scoring-api:${{ github.sha }}
```

#### **Docker**

**Container Strategy**:

- **Multi-stage builds** to minimize image size
- **Non-root users** for security
- **Health checks** for orchestration
- **Environment-specific configs** via env vars

**Sample Dockerfile (.NET)**:

```dockerfile
# Build stage
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY ["ScoringApi.csproj", "./"]
RUN dotnet restore
COPY . .
RUN dotnet publish -c Release -o /app/publish

# Runtime stage
FROM mcr.microsoft.com/dotnet/aspnet:8.0
WORKDIR /app
COPY --from=build /app/publish .
RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app
USER appuser
EXPOSE 8080
HEALTHCHECK CMD curl --fail http://localhost:8080/health || exit 1
ENTRYPOINT ["dotnet", "ScoringApi.dll"]
```

#### **Terraform (Infrastructure as Code)**

**Purpose**: Declarative infrastructure provisioning across Azure/GCP.

**Module Structure**:

```text
/infrastructure/terraform
  /modules
    /networking        # VNet, subnets, NSGs
    /aks               # Kubernetes cluster
    /databases         # SQL, Postgres, MongoDB
    /storage           # Blob/Cloud Storage
    /monitoring        # Log Analytics, alerts
  /environments
    /dev
    /staging
    /production
```

**Sample Module (Azure Postgres)**:

```hcl
resource "azurerm_postgresql_flexible_server" "main" {
  name                = "${var.prefix}-pg"
  resource_group_name = var.resource_group_name
  location            = var.location
  administrator_login = var.admin_username
  administrator_password = var.admin_password
  sku_name            = "GP_Standard_D4s_v3"
  storage_mb          = 524288
  version             = "16"
  
  high_availability {
    mode = "ZoneRedundant"
  }
  
  tags = var.tags
}

output "fqdn" {
  value = azurerm_postgresql_flexible_server.main.fqdn
}
```

#### **Ansible (Configuration Management)**

**Purpose**: Post-provisioning configuration, secrets injection, application deployment.

**Use Cases**:

- Install monitoring agents (Datadog, Prometheus exporters)
- Configure OS-level security (firewall, SELinux)
- Deploy Docker Compose for edge/dev environments
- Rotate credentials

**Sample Playbook**:

```yaml
- name: Configure AKS Nodes
  hosts: aks_nodes
  become: yes
  tasks:
    - name: Install monitoring agent
      apt:
        name: datadog-agent
        state: present
    
    - name: Configure log rotation
      copy:
        src: logrotate.conf
        dest: /etc/logrotate.d/app
    
    - name: Restart services
      systemd:
        name: datadog-agent
        state: restarted
```

### **6. ML/AI LAYER**

#### **LangChain (Python Library)**

**Purpose**: Build LLM-powered applications with retrieval, chains, and agents.

**Use Cases**:

- **RAG (Retrieval Augmented Generation)**: Policy document Q&A
- **Document summarization**: Condense 100-page ESG reports
- **Classification**: Tag policy documents by topic

**Architecture**:

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import PGVector
from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI

# Load and chunk documents
loader = PyPDFLoader("policy_doc.pdf")
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(pages)

# Embed and store
embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002")
vectorstore = PGVector.from_documents(
    chunks, 
    embeddings,
    connection_string="postgresql://user:pass@host/db",
    collection_name="policies"
)

# Query with LLM
llm = AzureOpenAI(deployment_name="gpt-4", temperature=0.2)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True
)

result = qa_chain({"query": "What are Indonesia's ecommerce regulations?"})
print(result["result"])
print(result["source_documents"])
```

#### **LangGraph (Workflow Orchestration)**

**Purpose**: Build stateful, multi-step AI workflows (like agents with loops, conditions).

**Use Cases**:

- **Multi-step research agent**: Search → Retrieve → Synthesize → Validate
- **Data quality agent**: Ingest → Validate → Flag errors → Human review
- **Policy recommendation pipeline**: Analyze gaps → Generate options → Score feasibility

**Sample Workflow**:

```python
from langgraph.graph import Graph

graph = Graph()

# Define nodes (steps)
graph.add_node("search", search_documents)
graph.add_node("retrieve", retrieve_context)
graph.add_node("generate", generate_answer)
graph.add_node("validate", validate_citations)

# Define edges (flow)
graph.add_edge("search", "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "validate")
graph.add_conditional_edge(
    "validate",
    lambda x: "search" if x["has_errors"] else "END"
)

# Execute
result = graph.run({"query": "Labor policy trends"})
```

#### **N8n (Workflow Automation)**

**Purpose**: Low-code workflow orchestration for rapid prototyping and integration.

**Use Cases**:

- **Data ingestion pipelines**: Schedule CSV downloads → validate → load to Postgres
- **Alert workflows**: Anomaly detected → send Slack message → create Jira ticket
- **API orchestration**: Call 5 external APIs → aggregate → cache in Redis

**Architecture**:

- Self-hosted on AKS/GKE (Docker container)
- PostgreSQL for workflow metadata
- Webhook triggers for event-driven flows

**Sample Workflow**:

1. HTTP trigger: Receive new trade data via webhook
2. Transform: Normalize CSV to JSON schema
3. Postgres node: Upsert into `indicators` table
4. Conditional: If error > 5%, send alert
5. Response: Return success status

**When to Replace N8n**:

- Mature workflows → migrate to Azure Functions/Cloud Functions
- Complex logic → replace with Python orchestration (Prefect, Airflow)

#### **scikit-learn (ML Library)**

**Purpose**: Classical ML models for forecasting, classification, clustering.

**Use Cases**:

- **Time-series forecasting**: ARIMA, Prophet for labor market predictions
- **Regression models**: Predict ecommerce growth from macro indicators
- **Clustering**: Group countries by economic profiles
- **Anomaly detection**: Flag unusual data points

**Sample Pipeline**:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import mlflow

# Define pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100))
])

# Train with cross-validation
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    pipeline.fit(X[train_idx], y[train_idx])
    score = pipeline.score(X[val_idx], y[val_idx])
    
    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_metric("r2_score", score)
        mlflow.sklearn.log_model(pipeline, "model")
```

**Integration**:

- Models stored in Azure ML / Vertex AI
- Served via FastAPI endpoint
- Monitored for drift (Evidently AI)

#### **Ollama (Local LLM Runtime)**

**Purpose**: Run open-source LLMs locally for cost optimization, data privacy, and offline scenarios.

**Use Cases**:

- **Prototype RAG**: Test LLaMA 3, Mistral before using Azure OpenAI
- **Data privacy**: Sensitive policy documents never leave infrastructure
- **Cost optimization**: Free inference for high-volume tasks (classification, extraction)

**Deployment**:

- Run as Docker container on GPU nodes (AKS with NVIDIA GPUs)
- Load quantized models (GGUF format) for efficiency
- Expose OpenAI-compatible API

**Sample Docker Compose**:

```yaml
version: '3.8'
services:
  ollama:
    image: ollama/ollama:latest
    volumes:
      - ./models:/root/.ollama
    ports:
      - "11434:11434"
    environment:
      - OLLAMA_MODELS=/root/.ollama/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**Integration with LangChain**:

```python
from langchain.llms import Ollama

llm = Ollama(model="llama3:8b", base_url="http://ollama:11434")
response = llm("Summarize ecommerce policy trends in 2025")
```

## **Complete Data Flow Example**

**Scenario**: Calculate ecommerce competitiveness score for Indonesia

1. **Ingestion (N8n)**:
    - Cron trigger: Daily at 2 AM UTC
    - Fetch data: UN Comtrade API (trade volumes)
    - Fetch data: World Bank API (internet penetration)
    - Validate schema: Great Expectations
    - Load: Upsert into Postgres `indicators` table
2. **Scoring (.NET API)**:
    - Request: `GET /v1/score/IDN?year=2025`
    - Load methodology config from MongoDB (YAML → JSON)
    - Query Postgres for indicators (trade, internet, platform adoption)
    - Apply weights: `score = 0.4*trade + 0.3*internet + 0.3*platform`
    - Calculate hash: SHA256(indicators + config version)
    - Cache result in Redis (TTL 10 min)
    - Return JSON:
  
 ```json
 {
   "country": "IDN",
   "year": 2025,
   "score": 78.5,
   "components": {"trade": 82, "internet": 75, "platform": 79},
   "methodology_version": "2025.10",
   "hash": "a3f5d9...<hash>"
 }
 ```

3. **RAG Analysis (Python + LangChain)**:
    - Request: `POST /v1/insights` with query "Why did Indonesia's score improve?"
    - Embed query via Azure OpenAI (text-embedding-ada-002)
    - Hybrid search:
        - Elasticsearch: BM25 keyword match on policy docs
        - Postgres pgvector: Semantic search on embeddings
    - Retrieve top 4 chunks
    - Generate answer via GPT-4 with sources
    - Validate citations (check if sources mentioned in response)
    - Return JSON with answer + source URLs

4. **Visualization (Next.js)**:
    - User visits `/countries/indonesia`
    - Server component fetches data from .NET API (ISR, revalidate=300)
    - Render chart: Recharts line graph showing score trend 2020-2025
    - Client component: Interactive sliders to adjust weights (what-if analysis)
    - On slider change: Client-side API call to scoring API
    - Real-time score update without page reload

5. **Monitoring**:
    - All services emit OpenTelemetry traces
    - Logs sent to Azure Monitor / Cloud Logging
    - Dashboards: Grafana showing:
        - API latency (P50, P95, P99)
        - Data freshness (hours since last update)
        - RAG quality (hallucination rate, citation coverage)
        - Infrastructure: CPU, memory, DB connections

----

## **Key Design Decisions & Rationale**

| Decision                     | Rationale                                                                             |
| ---------------------------- | ------------------------------------------------------------------------------------- |
| **.NET for Scoring API**     | Strong typing, low latency (<50ms), mature enterprise patterns (DI, CQRS)             |
| **Python for ML/AI**         | Rich ecosystem (scikit-learn, LangChain), data science team familiarity               |
| **Postgres as Primary DB**   | JSONB for flexibility, pgvector for RAG, TimescaleDB for time-series, ACID compliance |
| **Elasticsearch for Search** | Hybrid retrieval (BM25 + kNN), full-text search, log aggregation                      |
| **MongoDB for Staging**      | Schema-less for raw API responses, fast writes for event logs                         |
| **Next.js ISR**              | Fast page loads (static), fresh data (background revalidation), SEO-friendly          |
| **N8n for Prototyping**      | Visual workflows, rapid iteration, client demos; migrate to code later                |
| **Ollama for Privacy**       | Keep sensitive data on-premise, avoid API costs for high-volume tasks                 |
| **Terraform + Ansible**      | Terraform for infra provisioning, Ansible for runtime config (separation of concerns) |
| **AKS/GKE over Serverless**  | ML/AI workloads need GPUs, long-running processes, cost predictability                |

## **Scalability & Resilience**

**Horizontal Scaling**:

- AKS/GKE: Autoscale pods based on CPU/memory (HPA)
- API Management: Global distribution with Azure Front Door
- Databases: Read replicas for Postgres/MSSQL, sharding for MongoDB

**Resilience Patterns**:

- **Circuit Breaker**: Polly in .NET, tenacity in Python
- **Retry with Exponential Backoff**: Transient failures (network glitches)
- **Bulkhead**: Isolate thread pools per external dependency
- **Health Checks**: Kubernetes liveness/readiness probes
- **Graceful Degradation**: If LLM fails, return cached or extractive summaries

**Disaster Recovery**:

- Geo-redundant backups (MSSQL, Postgres)
- Multi-region deployment (active-passive)
- RTO < 1 hour, RPO < 5 minutes

## **Security**

**Identity & Access**:

- Azure AD / Google Identity for SSO
- OAuth2 + JWT for API authentication
- RBAC: Role-based access to APIs and data

**Secrets Management**:

- Azure Key Vault / Secret Manager
- No secrets in code or environment variables
- Automatic rotation every 90 days

**Network Security**:

- Private endpoints for databases (no public IPs)
- NSGs/Firewall rules (whitelist IPs)
- WAF on Azure Front Door (SQL injection, XSS protection)

**Data Security**:

- Encryption at rest: Azure Storage Service Encryption, Postgres TDE
- Encryption in transit: TLS 1.3 for all APIs
- PII anonymization: Mask sensitive fields in logs

**Compliance**:

- GDPR: Data residency (EU regions), right to erasure
- SOC 2: Audit logs, access controls
- PCI-DSS (if handling payments): Tokenization, no plaintext storage

## **Observability**

**Metrics** (Prometheus/Azure Monitor):

- API: Request rate, latency, error rate (RED metrics)
- Database: Connection pool usage, query time
- ML: Model inference latency, prediction drift

**Logs** (Elasticsearch/Cloud Logging):

- Structured JSON logs (Serilog, Python logging)
- Correlation IDs across services
- Log levels: DEBUG, INFO, WARN, ERROR

**Traces** (OpenTelemetry → Jaeger/Azure Monitor):

- Distributed tracing across .NET → Python → Database
- Identify bottlenecks (slow DB queries, external API calls)

**Dashboards** (Grafana):

- Golden signals: Latency, Traffic, Errors, Saturation
- Business metrics: Scores generated/hour, RAG accuracy
- Infrastructure: Node health, pod restarts

**Alerts** (Azure Monitor / Cloud Monitoring):

- P1: API error rate > 5% → PagerDuty
- P2: Data freshness > 24h → Slack
- P3: High latency (P95 > 500ms) → Email
