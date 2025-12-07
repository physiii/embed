# Embedding Service

A FastAPI service that provides text embeddings using **BAAI/bge-large-en-v1.5** model - a state-of-the-art BERT-based embedding model with 335M parameters producing 1024-dimensional embeddings.

## Running with Docker

### Prerequisites
- Docker
- Docker Compose

### Start the service

```bash
docker-compose up -d
```

This will:
1. Build the Docker image
2. Start the embedding service on port 8000

### Test the service

```bash
# Using curl
curl -X POST -H "Content-Type: application/json" -d '{"text":"Hello world"}' http://localhost:8000/embed

# Using the test script (outside the container)
./test.py
```

### Stop the service

```bash
docker-compose down
```

## API Usage

Send a POST request to `/embed` with a JSON payload containing a `text` field:

```json
{
  "text": "Your text to be embedded"
}
```

The response will be a JSON object with an `embedding` field containing the vector representation.

## Resource Management

The Docker Compose file is configured with memory limits. Adjust the values in `docker-compose.yml` based on your model's requirements and available system resources.

**Note**: BGE-large-en-v1.5 is a larger model (~1.2 GB) compared to previous models. Ensure you have sufficient memory (recommended: 4GB+ RAM/VRAM).

## ⚠️ Important: Vector Store Migration Required

**If you're switching from a different embedding model (e.g., all-MiniLM-L6-v2), you MUST regenerate all embeddings in your vector store.**

### Why?
- **Different dimensions**: Old model (384-dim) vs New model (1024-dim)
- **Different embedding spaces**: Embeddings from different models aren't comparable
- **Quality improvement**: New embeddings will be higher quality

### Quick Migration

```bash
# Regenerate embeddings from your text files
python migrate_embeddings.py --input your_texts.json --output embeddings_new.npy

# Or export for specific vector stores
python migrate_embeddings.py --input your_texts.json --format chroma
python migrate_embeddings.py --input your_texts.json --format pinecone
```

See [EMBEDDING_MIGRATION_GUIDE.md](EMBEDDING_MIGRATION_GUIDE.md) for detailed migration instructions.

## Model Architecture

For detailed information about the BGE-large-en-v1.5 architecture, training methodology, and fine-tuning results, see [BGE_ARCHITECTURE.md](BGE_ARCHITECTURE.md).

## Converting Model to ONNX for Visualization

To visualize the model architecture in Netron:

1. **Install dependencies** (if not already installed):
   ```bash
   pip install onnx onnxruntime
   ```

2. **Convert the model to ONNX**:
   ```bash
   python convert_to_onnx.py
   ```

3. **Visualize in Netron**:
   ```bash
   # Install Netron (if not already installed)
   pip install netron
   
   # Open the ONNX file
   netron bge-large-en-v1.5.onnx
   ```

   Or download Netron from: https://github.com/lutzroeder/netron

The ONNX file will be saved as `bge-large-en-v1.5.onnx` (approximately 1.3 GB).

## Model Quantization

The model supports quantization to reduce memory usage and improve inference speed. See [QUANTIZATION_GUIDE.md](QUANTIZATION_GUIDE.md) for detailed information.

### Quick Summary

| Precision | Model Size | Memory | Speed (GPU) | Speed (CPU) | Accuracy |
|-----------|------------|--------|-------------|-------------|----------|
| **FP32** (default) | 1.34 GB | 1.2-1.5 GB | Baseline | Baseline | 100% |
| **FP16** | 0.67 GB | 0.7-0.9 GB | 1.5-2x faster | Slower | 99.5-99.9% |
| **INT8** | 0.34 GB | 0.4-0.5 GB | 2-4x faster | 2-3x faster | 97-99% |

### Using Quantization

**Via Environment Variable:**
```bash
# FP16 (recommended for GPU)
QUANTIZATION=fp16 python main.py

# INT8 (recommended for CPU)
QUANTIZATION=int8 python main.py

# FP32 (default, maximum accuracy)
QUANTIZATION=fp32 python main.py
```

**In Docker Compose:**
Edit `docker-compose.yml` and set:
```yaml
environment:
  - QUANTIZATION=fp16  # or int8, fp32
```

**Compare Quantizations:**
```bash
python quantize_model.py --precision compare
```

This will benchmark all quantization levels and compare accuracy. 