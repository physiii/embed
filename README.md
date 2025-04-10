# Embedding Service

A FastAPI service that provides text embeddings using Alibaba-NLP/gte-Qwen2-1.5B-instruct model.

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