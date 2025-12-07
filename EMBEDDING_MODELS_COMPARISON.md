# Embedding Models Size Comparison

Comprehensive list of popular embedding models organized by size, from small to very large.

## Small Models (< 200 MB)

| Model Name | Size | Parameters | Embedding Dim | Speed | MTEB Score | Notes |
|------------|------|------------|--------------|-------|------------|-------|
| **all-MiniLM-L6-v2** | ~80 MB | 22.7M | 384 | ~14,000 sent/sec | 56.5-58.8 | ⭐ **Currently in use** - Fast, efficient |
| **all-MiniLM-L12-v2** | ~130 MB | 33.4M | 384 | ~7,500 sent/sec | 59.76 | Better quality than L6, slower |
| **all-distilroberta-v1** | ~290 MB | 82M | 768 | ~4,000 sent/sec | 59.84 | Good balance |
| **e5-small-v2** | ~130 MB | 33M | 384 | ~10,000 sent/sec | ~58 | Salesforce model |

## Medium Models (200 MB - 1 GB)

| Model Name | Size | Parameters | Embedding Dim | Speed | MTEB Score | Notes |
|------------|------|------------|--------------|-------|------------|-------|
| **all-mpnet-base-v2** | ~420 MB | 110M | 768 | ~2,800-4,000 sent/sec | 57.8-63.3 | Strong general-purpose |
| **bge-small-en-v1.5** | ~130 MB | 33M | 384 | ~10,000 sent/sec | ~62 | BAAI model |
| **bge-base-en-v1.5** | ~420 MB | 110M | 768 | ~4,000 sent/sec | ~63 | BAAI model |
| **e5-base-v2** | ~440 MB | 110M | 768 | ~3,500 sent/sec | ~62 | Salesforce model |
| **gte-small** | ~130 MB | 33M | 384 | ~10,000 sent/sec | ~61 | Alibaba model |
| **gte-base** | ~440 MB | 110M | 768 | ~3,500 sent/sec | ~63 | Alibaba model |
| **instructor-base** | ~440 MB | 110M | 768 | ~3,500 sent/sec | ~61 | Instruction-tuned |
| **instructor-large** | ~1.2 GB | 330M | 768 | ~1,500 sent/sec | ~64 | Instruction-tuned |

## Large Models (1 GB - 5 GB)

| Model Name | Size | Parameters | Embedding Dim | Speed | MTEB Score | Notes |
|------------|------|------------|--------------|-------|------------|-------|
| **bge-large-en-v1.5** | ~1.2 GB | 330M | 1024 | ~1,500 sent/sec | ~64-65 | ⭐ Top MTEB performer |
| **e5-large-v2** | ~1.2 GB | 330M | 1024 | ~1,500 sent/sec | ~64 | Salesforce model |
| **gte-large** | ~1.2 GB | 330M | 1024 | ~1,500 sent/sec | ~64 | Alibaba model |
| **instructor-xl** | ~4.5 GB | 1.2B | 768 | ~400 sent/sec | ~66 | Very high quality |
| **gtr-t5-large** | ~1.2 GB | 330M | 768 | ~1,500 sent/sec | ~63 | Google T5-based |
| **gtr-t5-xl** | ~4.5 GB | 1.2B | 768 | ~400 sent/sec | ~66 | Google T5-based |
| **sentence-t5-xl** | ~4.8 GB | 1.2B | 768 | ~400 sent/sec | ~66 | Multilingual capable |

## Very Large Models (5 GB - 12 GB)

| Model Name | Size | Parameters | Embedding Dim | Speed | MTEB Score | Notes |
|------------|------|------------|--------------|-------|------------|-------|
| **gtr-t5-xxl** | ~18 GB | 4.9B | 768 | ~100 sent/sec | ~68 | State-of-the-art |
| **sentence-t5-xxl** | ~19.6 GB | 4.9B | 768 | ~100 sent/sec | ~68 | Very high quality |
| **bge-m3** | ~2.2 GB | 560M | 1024 | ~800 sent/sec | ~66 | Multilingual |

## Notes

- **Size**: Approximate disk size (can vary based on quantization/format)
- **Speed**: Approximate sentences per second (varies by hardware)
- **MTEB Score**: Average score on Massive Text Embedding Benchmark (higher = better)
- **Embedding Dim**: Dimensionality of output vectors

## Recommendations by Use Case

- **Fast, lightweight**: `all-MiniLM-L6-v2` (current) or `e5-small-v2`
- **Best quality/size ratio**: `bge-large-en-v1.5` or `e5-large-v2`
- **State-of-the-art**: `gtr-t5-xxl` or `sentence-t5-xxl` (requires significant resources)
- **Multilingual**: `sentence-t5-xl` or `bge-m3`
- **Instruction-tuned**: `instructor-large` or `instructor-xl`

## Model Families

- **MiniLM**: Microsoft's efficient models
- **MPNet**: Microsoft's general-purpose models
- **E5**: Salesforce's embedding models
- **BGE**: BAAI's (Beijing Academy of AI) models
- **GTE**: Alibaba's general text embedding models
- **GTR-T5**: Google's T5-based retrieval models
- **Instructor**: HKUNLP's instruction-tuned models

---

# Benchmark Metrics Table

Comprehensive benchmark performance metrics for embedding models. All scores are from MTEB (Massive Text Embedding Benchmark) unless otherwise noted.

## Complete Benchmark Metrics

| Model Name | Size | MTEB Avg | Retrieval | Classification | Clustering | Rerank | STS | R@10 | R@100 | MRR | NDCG | Notes |
|------------|------|----------|-----------|---------------|------------|--------|-----|------|-------|-----|------|-------|
| **all-MiniLM-L6-v2** | 80 MB | 56.5-58.8 | 41.9 | 62.1 | 41.8 | - | - | 0.81 | - | 0.69 | 0.74 | ⭐ **Currently in use** |
| **all-MiniLM-L12-v2** | 130 MB | 59.76 | - | - | - | - | - | 0.60 | - | 0.60 | 0.68 | Better than L6 |
| **all-mpnet-base-v2** | 420 MB | 57.8-63.3 | - | - | - | - | - | - | - | - | - | Strong general-purpose |
| **all-distilroberta-v1** | 290 MB | 59.84 | - | - | - | - | - | - | - | - | - | Good balance |
| **e5-small-v2** | 130 MB | ~58 | - | - | - | - | - | 0.68 | - | 0.68 | 0.76 | Salesforce |
| **e5-base-v2** | 440 MB | ~62 | - | - | - | - | - | - | - | - | - | Salesforce |
| **e5-large-v2** | 1.2 GB | 64.23 | - | - | - | - | - | 0.92 | - | 0.79 | 0.83 | Salesforce |
| **bge-small-en-v1.5** | 130 MB | ~62 | - | - | - | - | - | - | - | - | - | BAAI |
| **bge-base-en-v1.5** | 420 MB | ~63 | - | - | - | - | - | 0.81 | - | 0.68 | 0.68 | BAAI |
| **bge-large-en-v1.5** | 1.2 GB | 64.23-89.2 | - | - | - | - | - | 0.91 | - | 0.80 | 0.84 | ⭐ Top performer |
| **bge-m3** | 2.2 GB | ~66 | - | - | - | - | - | 0.87 | - | 0.68 | 0.76 | Multilingual |
| **gte-small** | 130 MB | ~61 | - | - | - | - | - | - | - | - | - | Alibaba |
| **gte-base** | 440 MB | ~63 | - | - | - | - | - | - | - | - | - | Alibaba |
| **gte-large** | 1.2 GB | 63.13-83.8 | - | - | - | - | - | - | - | - | - | Alibaba |
| **instructor-base** | 440 MB | ~61 | - | - | - | - | - | - | - | - | - | Instruction-tuned |
| **instructor-large** | 1.2 GB | ~64 | - | - | - | - | - | - | - | - | - | Instruction-tuned |
| **gtr-t5-large** | 1.2 GB | ~63 | - | - | - | - | - | - | - | - | - | Google T5 |
| **gtr-t5-xl** | 4.5 GB | ~66 | - | - | - | - | - | - | - | - | - | Google T5 |
| **gtr-t5-xxl** | 18 GB | ~68 | - | - | - | - | - | - | - | - | - | State-of-the-art |
| **sentence-t5-xl** | 4.8 GB | ~66 | - | - | - | - | - | - | - | - | - | Multilingual |
| **sentence-t5-xxl** | 19.6 GB | ~68 | - | - | - | - | - | - | - | - | Very high quality |

## Detailed Retrieval Metrics

| Model Name | Size | Top-1 Hit Rate | Top-5 Hit Rate | R@5 | MRR@5 | NDCG@5 | MAP | Notes |
|------------|------|---------------|---------------|-----|-------|--------|-----|-------|
| **all-MiniLM-L6-v2** | 80 MB | 0.68 | 0.87 | 0.81 | 0.69 | 0.74 | 0.70 | ⭐ **Currently in use** |
| **all-MiniLM-L12-v2** | 130 MB | 0.60 | 0.81 | 0.60 | 0.60 | 0.68 | 0.70 | Better quality than L6 |
| **bge-base-en-v1.5** | 420 MB | 0.60 | 0.81 | 0.81 | 0.68 | 0.68 | 0.70 | BAAI model |
| **bge-large-en-v1.5** | 1.2 GB | 0.91 | 0.94 | 0.91 | 0.80 | 0.84 | 0.81 | ⭐ Top performer |
| **e5-large-v2** | 1.2 GB | 0.92 | 0.93 | 0.92 | 0.79 | 0.83 | 0.80 | Salesforce |
| **bge-m3** | 2.2 GB | 0.68 | 0.87 | 0.87 | 0.68 | 0.76 | 0.70 | Multilingual |

## Metric Definitions

- **MTEB Average**: Average score across all MTEB tasks (retrieval, classification, clustering, rerank, STS, etc.)
- **Retrieval**: Performance on information retrieval tasks
- **Classification**: Text classification accuracy
- **Clustering**: Clustering quality metrics
- **Rerank**: Reranking performance
- **STS**: Semantic Textual Similarity score
- **R@10 / R@100**: Recall at 10/100 (proportion of relevant docs in top 10/100)
- **R@5**: Recall at 5
- **MRR**: Mean Reciprocal Rank (average of reciprocal ranks of first relevant result)
- **MRR@5**: Mean Reciprocal Rank at 5
- **NDCG**: Normalized Discounted Cumulative Gain (ranking quality metric)
- **NDCG@5**: NDCG at 5
- **MAP**: Mean Average Precision
- **Top-1 Hit Rate**: Proportion of queries where top result is relevant
- **Top-5 Hit Rate**: Proportion of queries where at least one of top 5 results is relevant

## Notes on Benchmarks

- **MTEB scores** can vary slightly between different evaluation runs and versions
- **Retrieval metrics** (R@10, MRR, NDCG) are most relevant for search/RAG applications
- **Classification/Clustering** metrics matter more for categorization tasks
- **Speed** (sentences/second) is not included in MTEB but is important for production use
- Some models have **multiple reported scores** due to different evaluation setups or versions

## Performance Summary

**Best Overall Quality:**
- `bge-large-en-v1.5` (64.23 MTEB avg, 0.91 R@5, 0.80 MRR)
- `e5-large-v2` (64.23 MTEB avg, 0.92 R@5, 0.79 MRR)
- `gtr-t5-xxl` (~68 MTEB avg, requires 18GB)

**Best Speed/Quality Balance:**
- `all-MiniLM-L6-v2` (current) - Fastest, decent quality
- `bge-base-en-v1.5` - Good quality, reasonable speed
- `e5-base-v2` - Good quality, reasonable speed

**Best for Retrieval:**
- `bge-large-en-v1.5` - Highest R@5 (0.91) and MRR (0.80)
- `e5-large-v2` - Very close second (0.92 R@5, 0.79 MRR)

