from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn
import numpy as np
import logging
import os
import torch
import subprocess
from typing import List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# ---- GPU / memory diagnostics ----
def _bytes_to_mib(num_bytes: int) -> float:
    return float(num_bytes) / (1024.0 * 1024.0)


def _torch_module_bytes(module: torch.nn.Module) -> dict:
    param_bytes = 0
    for p in module.parameters(recurse=True):
        param_bytes += p.numel() * p.element_size()

    buffer_bytes = 0
    for b in module.buffers(recurse=True):
        buffer_bytes += b.numel() * b.element_size()

    return {
        "param_bytes": int(param_bytes),
        "buffer_bytes": int(buffer_bytes),
        "param_mib": _bytes_to_mib(int(param_bytes)),
        "buffer_mib": _bytes_to_mib(int(buffer_bytes)),
    }


def _nvidia_smi_process_vram_mib(pid: int) -> Optional[int]:
    """
    Best-effort: query per-process VRAM from nvidia-smi (works across processes).
    Returns MiB or None if unavailable.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            if int(parts[0]) == int(pid):
                return int(parts[1])
    except Exception:
        return None
    return None


def _cuda_report() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}

    dev = torch.cuda.current_device()
    name = torch.cuda.get_device_name(dev)
    alloc = torch.cuda.memory_allocated(dev)
    reserved = torch.cuda.memory_reserved(dev)
    max_alloc = torch.cuda.max_memory_allocated(dev)
    max_reserved = torch.cuda.max_memory_reserved(dev)
    free_b, total_b = torch.cuda.mem_get_info(dev)

    # nvidia-smi per-process VRAM (includes non-torch allocations too)
    smi_mib = _nvidia_smi_process_vram_mib(os.getpid())

    return {
        "available": True,
        "device_index": int(dev),
        "device_name": name,
        "mem_free_mib": _bytes_to_mib(int(free_b)),
        "mem_total_mib": _bytes_to_mib(int(total_b)),
        "torch_allocated_mib": _bytes_to_mib(int(alloc)),
        "torch_reserved_mib": _bytes_to_mib(int(reserved)),
        "torch_max_allocated_mib": _bytes_to_mib(int(max_alloc)),
        "torch_max_reserved_mib": _bytes_to_mib(int(max_reserved)),
        "nvidia_smi_process_used_mib": smi_mib,
    }


# Load the model - BGE-large-en-v1.5 for high-quality embeddings
# Quantization can be controlled via QUANTIZATION environment variable: fp32, fp16, or int8
quantization = os.getenv("QUANTIZATION", "fp32").lower()

logger.info(f"Loading model with quantization: {quantization.upper()}")

# Determine device (GPU if available, otherwise CPU).
# You can pin to a specific GPU with EMBED_CUDA_DEVICE (e.g. "0" or "1").
embed_cuda_device = os.getenv("EMBED_CUDA_DEVICE")
if torch.cuda.is_available():
    device = f"cuda:{embed_cuda_device}" if embed_cuda_device is not None else "cuda"
else:
    device = "cpu"
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    try:
        logger.info(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    except Exception:
        pass

model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)

# Apply quantization if requested
if quantization == "fp16":
    if torch.cuda.is_available():
        model = model.half()
        logger.info("Model converted to FP16 (half precision)")
    else:
        logger.warning("FP16 on CPU may be slower. Consider INT8 or keeping FP32.")
        model = model.half()
elif quantization == "int8":
    # NOTE: torch dynamic quantization is CPU-oriented. If you need GPU INT8, use a GPU quantization backend.
    # We keep this best-effort, but call it out loudly because it may not run on GPU as expected.
    logger.warning("INT8 quantization via torch.quantization is CPU-oriented; verify GPU usage via /gpu_report.")
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

class BatchEmbedRequest(BaseModel):
    texts: List[str]

class BatchEmbedResponse(BaseModel):
    embeddings: List[List[float]]

@app.post("/embed", response_model=EmbedResponse)
async def create_embedding(request: EmbedRequest):
    try:
        # Log the incoming request text
        logger.info(f"Received embedding request for text: {request.text[:100]}...")  # Log first 100 characters

        # Generate the embedding
        with torch.no_grad():
            embedding = model.encode(request.text)
        
        # Convert numpy array to list for JSON serialization
        embedding_list = embedding.tolist()
        
        logger.info(f"Successfully generated embedding of length {len(embedding_list)}")
        
        return {"embedding": embedding_list}
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed_batch", response_model=BatchEmbedResponse)
async def create_embedding_batch(request: BatchEmbedRequest):
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    try:
        preview = request.texts[0][:100] if request.texts else ""
        logger.info(f"Received batch embedding request for {len(request.texts)} texts. First text preview: {preview}...")

        # Batch encode all texts in one forward pass
        with torch.no_grad():
            embeddings = model.encode(
                request.texts,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

        embeddings_list = embeddings.tolist()
        logger.info(f"Successfully generated {len(embeddings_list)} embeddings of length {len(embeddings_list[0]) if embeddings_list else 0}")

        return {"embeddings": embeddings_list}
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gpu_report")
async def gpu_report():
    """
    Self-report what this embedding service is doing w.r.t GPU usage.

    Includes:
      - which CUDA device this process is using (if any)
      - VRAM usage (torch allocator + nvidia-smi per-process, best-effort)
      - model parameter/buffer bytes (i.e. weights/coefficients)
    """
    model_id = "BAAI/bge-large-en-v1.5"
    try:
        if hasattr(model, "_model_card_vars"):
            if "modelId" in model._model_card_vars:
                model_id = model._model_card_vars["modelId"]
            elif "name" in model._model_card_vars:
                model_id = model._model_card_vars["name"]
    except Exception:
        pass

    module_bytes = _torch_module_bytes(model)
    cuda = _cuda_report()

    non_weight_torch_alloc_mib = None
    if cuda.get("available"):
        non_weight_torch_alloc_mib = cuda["torch_allocated_mib"] - module_bytes["param_mib"] - module_bytes["buffer_mib"]

    return {
        "pid": os.getpid(),
        "model_id": model_id,
        "quantization": quantization,
        "device": str(next(model.parameters()).device) if any(True for _ in model.parameters()) else "unknown",
        "weights": {
            "param_bytes": module_bytes["param_bytes"],
            "buffer_bytes": module_bytes["buffer_bytes"],
            "param_mib": module_bytes["param_mib"],
            "buffer_mib": module_bytes["buffer_mib"],
        },
        "cuda": cuda,
        "derived": {
            "non_weight_torch_alloc_mib_estimate": non_weight_torch_alloc_mib,
        },
        "env": {
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
            "EMBED_CUDA_DEVICE": os.getenv("EMBED_CUDA_DEVICE"),
        },
    }

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
        logger.info(f"Model device: {next(model.parameters()).device}")
        logger.info(f"GPU report: {_cuda_report()}")
    except Exception as e:
        logger.error(f"Error retrieving model information: {str(e)}")
        logger.info("Model loaded, but unable to retrieve detailed information.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)