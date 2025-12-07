# Embedding Model Migration Guide

## Why You Need to Regenerate Embeddings

When switching embedding models, **you MUST regenerate all existing embeddings** in your vector store. Here's why:

### 1. Different Embedding Dimensions

- **Old Model (all-MiniLM-L6-v2)**: 384-dimensional embeddings
- **New Model (BGE-large-en-v1.5)**: 1024-dimensional embeddings

Vector stores are dimension-specific. You cannot mix embeddings of different dimensions.

### 2. Different Embedding Spaces

Even if dimensions matched, embeddings from different models exist in different semantic spaces:
- Models learn different representations
- Cosine similarity between embeddings from different models is meaningless
- A document that's "similar" in one model's space may not be in another's

### 3. Quality and Accuracy

- BGE-large-en-v1.5 produces higher quality embeddings
- Old embeddings won't benefit from the new model's improved semantic understanding
- Search/retrieval quality will be inconsistent

## Migration Strategy

### Option 1: Complete Regeneration (Recommended)

**Best for**: Fresh start, maximum quality

1. **Backup your current vector store** (if you want to keep old data)
2. **Regenerate all embeddings** with the new model
3. **Re-index your vector store**

**Pros:**
- Clean, consistent embeddings
- Maximum quality
- No compatibility issues

**Cons:**
- Requires full re-indexing
- Takes time for large datasets

### Option 2: Gradual Migration

**Best for**: Large datasets, zero-downtime requirements

1. **Create a new vector store** with new embeddings
2. **Dual-write**: Generate embeddings with both models temporarily
3. **Query both stores** and merge results (weighted)
4. **Gradually migrate** documents to new store
5. **Retire old store** once migration complete

**Pros:**
- No downtime
- Can compare old vs new quality
- Gradual transition

**Cons:**
- More complex
- Temporary storage overhead
- Requires dual query logic

### Option 3: Hybrid Approach

**Best for**: Testing new model quality

1. **Keep old vector store** running
2. **Create new vector store** with new model
3. **A/B test** queries on both
4. **Compare results** and measure quality improvements
5. **Switch to new store** once validated

## Migration Checklist

- [ ] **Identify your vector store** (Chroma, Pinecone, Qdrant, FAISS, Weaviate, etc.)
- [ ] **Backup current embeddings** (export if possible)
- [ ] **Document current embedding model** (dimension, model name)
- [ ] **Update embedding generation code** to use BGE-large-en-v1.5
- [ ] **Regenerate all embeddings** with new model
- [ ] **Update vector store schema** (if dimension changed)
- [ ] **Re-index all documents**
- [ ] **Test retrieval quality** on sample queries
- [ ] **Validate results** match or exceed old model
- [ ] **Deploy new vector store**
- [ ] **Monitor performance** and accuracy

## Code Examples

### Example: ChromaDB Migration

```python
import chromadb
from sentence_transformers import SentenceTransformer

# Old model (if you still have it)
# old_model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dims

# New model
new_model = SentenceTransformer("BAAI/bge-large-en-v1.5")  # 1024 dims

# Connect to ChromaDB
client = chromadb.Client()
collection = client.get_or_create_collection(
    name="documents",
    metadata={"embedding_dimension": 1024}  # Update dimension!
)

# Regenerate embeddings for all documents
documents = collection.get()  # Get all documents
for doc_id, text in zip(documents['ids'], documents['documents']):
    # Generate new embedding
    new_embedding = new_model.encode(text).tolist()
    
    # Update in vector store
    collection.update(
        ids=[doc_id],
        embeddings=[new_embedding]
    )
```

### Example: Pinecone Migration

```python
import pinecone
from sentence_transformers import SentenceTransformer

# Initialize
pinecone.init(api_key="your-api-key")
new_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Create new index with correct dimension
index_name = "documents-v2"  # New index
pinecone.create_index(
    index_name,
    dimension=1024,  # New dimension
    metric="cosine"
)

index = pinecone.Index(index_name)

# Fetch all vectors from old index
old_index = pinecone.Index("documents-v1")
vectors = old_index.fetch(ids=all_ids)

# Regenerate and upsert
for vector_id, metadata in vectors['vectors'].items():
    text = metadata['text']  # Assuming text is stored in metadata
    new_embedding = new_model.encode(text).tolist()
    
    index.upsert(
        vectors=[(vector_id, new_embedding, metadata)]
    )
```

### Example: FAISS Migration

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# New model
new_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Load old index and texts
# (Assuming you have the original texts stored)
texts = load_texts()  # Your function to load texts

# Generate new embeddings
embeddings = new_model.encode(texts, show_progress_bar=True)

# Create new FAISS index
dimension = 1024
index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

# Normalize embeddings for cosine similarity
faiss.normalize_L2(embeddings)

# Add to index
index.add(embeddings.astype('float32'))

# Save new index
faiss.write_index(index, "embeddings_new.faiss")
```

## Batch Processing Script

For large datasets, process in batches:

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

def regenerate_embeddings_batch(
    texts,
    model_name="BAAI/bge-large-en-v1.5",
    batch_size=32,
    output_file="embeddings_new.npy"
):
    """Regenerate embeddings in batches."""
    
    model = SentenceTransformer(model_name)
    all_embeddings = []
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        embeddings = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        all_embeddings.append(embeddings)
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)
    
    # Save
    np.save(output_file, all_embeddings)
    print(f"Saved {len(all_embeddings)} embeddings to {output_file}")
    print(f"Embedding dimension: {all_embeddings.shape[1]}")
    
    return all_embeddings
```

## Validation Steps

After migration, validate the new embeddings:

### 1. Dimension Check
```python
assert embeddings.shape[1] == 1024, "Wrong dimension!"
```

### 2. Quality Check
```python
# Test on known similar documents
text1 = "Machine learning is a subset of AI"
text2 = "ML is part of artificial intelligence"

emb1 = new_model.encode(text1)
emb2 = new_model.encode(text2)

similarity = cosine_similarity([emb1], [emb2])[0][0]
print(f"Similarity: {similarity:.4f}")  # Should be high (>0.8)
```

### 3. Retrieval Test
```python
# Test retrieval on sample queries
query = "What is machine learning?"
results = vector_store.query(
    query_embeddings=[new_model.encode(query)],
    n_results=5
)

# Verify results are relevant
```

## Performance Considerations

### Memory Usage
- **Old model**: ~80 MB, 384-dim embeddings
- **New model**: ~1.2 GB, 1024-dim embeddings
- **Storage**: ~2.67x larger (1024/384) per embedding

### Regeneration Time
- Depends on dataset size and hardware
- BGE-large is slower than MiniLM (~1,500 vs 14,000 sent/sec)
- Use batch processing for efficiency

### Cost (if using cloud services)
- More storage needed (larger embeddings)
- Potentially more compute for regeneration
- Consider quantization (FP16/INT8) to reduce costs

## Rollback Plan

If migration has issues:

1. **Keep old vector store** until new one is validated
2. **Version your vector stores** (e.g., `documents-v1`, `documents-v2`)
3. **Test thoroughly** before switching production
4. **Monitor metrics** after switch (retrieval quality, latency)
5. **Have rollback procedure** documented

## Summary

✅ **Yes, you MUST regenerate all embeddings** when switching models

**Key Points:**
- Different dimensions (384 → 1024)
- Different embedding spaces (not comparable)
- New model = better quality (worth the migration)
- Plan for 2-3x storage increase
- Test thoroughly before production switch

**Next Steps:**
1. Identify your vector store
2. Backup current data
3. Use the migration script appropriate for your store
4. Regenerate all embeddings
5. Validate quality
6. Deploy new vector store

