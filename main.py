from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn
import numpy as np
import logging
import os
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Load the model - BGE-large-en-v1.5 for high-quality embeddings
# Quantization can be controlled via QUANTIZATION environment variable: fp32, fp16, or int8
quantization = os.getenv("QUANTIZATION", "fp32").lower()

logger.info(f"Loading model with quantization: {quantization.upper()}")

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Apply quantization if requested
if quantization == "fp16":
    if torch.cuda.is_available():
        model = model.half()
        logger.info("Model converted to FP16 (half precision)")
    else:
        logger.warning("FP16 on CPU may be slower. Consider INT8 or keeping FP32.")
        model = model.half()
elif quantization == "int8":
    try:
        # Apply dynamic quantization
        transformer = model[0].auto_model
        quantized_model = torch.quantization.quantize_dynamic(
            transformer,
            {torch.nn.Linear, torch.nn.LayerNorm},
            dtype=torch.qint8
        )
        model[0].auto_model = quantized_model
        logger.info("Model quantized to INT8")
    except Exception as e:
        logger.error(f"Failed to quantize to INT8: {e}")
        logger.info("Falling back to FP32")
elif quantization != "fp32":
    logger.warning(f"Unknown quantization type: {quantization}. Using FP32.")

model.eval()

class EmbedRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    embedding: list[float]

@app.post("/embed", response_model=EmbedResponse)
async def create_embedding(request: EmbedRequest):
    try:
        # Log the incoming request text
        logger.info(f"Received embedding request for text: {request.text[:100]}...")  # Log first 100 characters

        # Generate the embedding
        embedding = model.encode(request.text)
        
        # Convert numpy array to list for JSON serialization
        embedding_list = embedding.tolist()
        
        logger.info(f"Successfully generated embedding of length {len(embedding_list)}")
        
        return {"embedding": embedding_list}
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    logger.info("Embedding server is starting up...")
    try:
        model_info = f"Model loaded: {model.__class__.__name__}"
        if hasattr(model, '_model_card_vars'):
            if 'name' in model._model_card_vars:
                model_info += f" - {model._model_card_vars['name']}"
            elif 'modelId' in model._model_card_vars:
                model_info += f" - {model._model_card_vars['modelId']}"
        logger.info(model_info)
        logger.info(f"Quantization: {quantization.upper()}")
        logger.info(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")
        if torch.cuda.is_available():
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Running on CPU")
    except Exception as e:
        logger.error(f"Error retrieving model information: {str(e)}")
        logger.info("Model loaded, but unable to retrieve detailed information.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)