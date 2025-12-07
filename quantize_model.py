"""
Quantization utility for BGE-large-en-v1.5 model.

Supports FP16, INT8, and INT4 quantization with performance comparisons.
"""

import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import os
import time
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_model_size_mb(model_path_or_name: str) -> float:
    """Get model size in MB."""
    if os.path.exists(model_path_or_name):
        return os.path.getsize(model_path_or_name) / (1024 * 1024)
    return 0.0

def benchmark_model(model, texts: list, num_iterations: int = 10):
    """Benchmark model inference speed."""
    # Warmup
    _ = model.encode(texts[:1])
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        _ = model.encode(texts)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    throughput = len(texts) / avg_time
    return avg_time, throughput

def compare_embeddings(embeddings1, embeddings2):
    """Compare two embedding sets using cosine similarity."""
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Normalize embeddings
    embeddings1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
    
    # Compute cosine similarity
    similarity = cosine_similarity(embeddings1_norm, embeddings2_norm)
    # Get diagonal (comparing same texts)
    diagonal = np.diag(similarity)
    
    return {
        'mean_similarity': float(np.mean(diagonal)),
        'min_similarity': float(np.min(diagonal)),
        'max_similarity': float(np.max(diagonal)),
        'std_similarity': float(np.std(diagonal))
    }

def load_fp32_model(model_name: str = "BAAI/bge-large-en-v1.5"):
    """Load model in FP32 precision."""
    logger.info("Loading FP32 model...")
    model = SentenceTransformer(model_name)
    model.eval()
    return model

def load_fp16_model(model_name: str = "BAAI/bge-large-en-v1.5"):
    """Load model in FP16 precision."""
    logger.info("Loading FP16 model...")
    model = SentenceTransformer(model_name)
    
    # Convert to FP16
    if torch.cuda.is_available():
        model = model.half()  # Convert to FP16
        logger.info("Model converted to FP16 (GPU)")
    else:
        logger.warning("FP16 on CPU may be slower. Consider INT8 instead.")
        # Still convert but may not be optimal
        model = model.half()
    
    model.eval()
    return model

def load_int8_model(model_name: str = "BAAI/bge-large-en-v1.5"):
    """Load model with INT8 quantization (dynamic quantization)."""
    logger.info("Loading INT8 quantized model...")
    model = SentenceTransformer(model_name)
    
    # Get the underlying transformer model
    transformer = model[0].auto_model
    
    # Apply dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        transformer,
        {torch.nn.Linear, torch.nn.LayerNorm},  # Quantize these layers
        dtype=torch.qint8
    )
    
    # Replace the model
    model[0].auto_model = quantized_model
    model.eval()
    
    logger.info("Model quantized to INT8")
    return model

def load_int8_bitsandbytes(model_name: str = "BAAI/bge-large-en-v1.5"):
    """Load model with 8-bit quantization using bitsandbytes (requires transformers)."""
    try:
        from transformers import AutoModel, AutoTokenizer
        import torch
        
        logger.info("Loading INT8 model with bitsandbytes...")
        
        # Load with 8-bit quantization
        model = AutoModel.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        logger.info("Model loaded with 8-bit quantization (bitsandbytes)")
        return model
    except ImportError:
        logger.error("bitsandbytes not installed. Install with: pip install bitsandbytes")
        raise

def analyze_model(model, model_name: str, test_texts: Optional[list] = None):
    """Analyze model properties and performance."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Analyzing: {model_name}")
    logger.info(f"{'='*60}")
    
    # Get model info
    embedding_dim = model.get_sentence_embedding_dimension()
    logger.info(f"Embedding dimension: {embedding_dim}")
    
    # Estimate model size (rough)
    if hasattr(model, 'get_memory_footprint'):
        memory_mb = model.get_memory_footprint() / (1024 * 1024)
        logger.info(f"Estimated memory footprint: {memory_mb:.2f} MB")
    
    # Benchmark if test texts provided
    if test_texts:
        logger.info(f"Benchmarking with {len(test_texts)} texts...")
        avg_time, throughput = benchmark_model(model, test_texts, num_iterations=5)
        logger.info(f"Average inference time: {avg_time*1000:.2f} ms")
        logger.info(f"Throughput: {throughput:.2f} sentences/second")
    
    return {
        'embedding_dim': embedding_dim,
        'throughput': throughput if test_texts else None
    }

def compare_quantizations(
    model_name: str = "BAAI/bge-large-en-v1.5",
    test_texts: Optional[list] = None,
    compare_accuracy: bool = True
):
    """Compare different quantization levels."""
    
    if test_texts is None:
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing enables computers to understand human language.",
            "Embeddings represent text as dense vectors in high-dimensional space.",
            "Semantic search finds documents based on meaning rather than keywords."
        ]
    
    results = {}
    
    # Load FP32 baseline
    logger.info("\n" + "="*60)
    logger.info("Loading FP32 baseline model...")
    logger.info("="*60)
    model_fp32 = load_fp32_model(model_name)
    results['fp32'] = analyze_model(model_fp32, "FP32", test_texts)
    
    if compare_accuracy:
        embeddings_fp32 = model_fp32.encode(test_texts)
    
    # Test FP16
    try:
        logger.info("\n" + "="*60)
        logger.info("Testing FP16 quantization...")
        logger.info("="*60)
        model_fp16 = load_fp16_model(model_name)
        results['fp16'] = analyze_model(model_fp16, "FP16", test_texts)
        
        if compare_accuracy:
            embeddings_fp16 = model_fp16.encode(test_texts)
            comparison = compare_embeddings(embeddings_fp32, embeddings_fp16)
            logger.info(f"\nFP16 vs FP32 Accuracy:")
            logger.info(f"  Mean cosine similarity: {comparison['mean_similarity']:.6f}")
            logger.info(f"  Min similarity: {comparison['min_similarity']:.6f}")
            logger.info(f"  Max similarity: {comparison['max_similarity']:.6f}")
            results['fp16']['accuracy'] = comparison
    except Exception as e:
        logger.error(f"Failed to load FP16 model: {e}")
    
    # Test INT8
    try:
        logger.info("\n" + "="*60)
        logger.info("Testing INT8 quantization...")
        logger.info("="*60)
        model_int8 = load_int8_model(model_name)
        results['int8'] = analyze_model(model_int8, "INT8", test_texts)
        
        if compare_accuracy:
            embeddings_int8 = model_int8.encode(test_texts)
            comparison = compare_embeddings(embeddings_fp32, embeddings_int8)
            logger.info(f"\nINT8 vs FP32 Accuracy:")
            logger.info(f"  Mean cosine similarity: {comparison['mean_similarity']:.6f}")
            logger.info(f"  Min similarity: {comparison['min_similarity']:.6f}")
            logger.info(f"  Max similarity: {comparison['max_similarity']:.6f}")
            results['int8']['accuracy'] = comparison
    except Exception as e:
        logger.error(f"Failed to load INT8 model: {e}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("QUANTIZATION COMPARISON SUMMARY")
    logger.info("="*60)
    
    if 'fp32' in results:
        logger.info(f"FP32:  {results['fp32'].get('throughput', 'N/A'):.2f} sent/sec")
    if 'fp16' in results:
        speedup = results['fp16'].get('throughput', 0) / results['fp32'].get('throughput', 1) if 'fp32' in results else 1
        logger.info(f"FP16:  {results['fp16'].get('throughput', 'N/A'):.2f} sent/sec ({speedup:.2f}x speedup)")
        if 'accuracy' in results['fp16']:
            logger.info(f"       Accuracy: {results['fp16']['accuracy']['mean_similarity']:.6f} cosine similarity")
    if 'int8' in results:
        speedup = results['int8'].get('throughput', 0) / results['fp32'].get('throughput', 1) if 'fp32' in results else 1
        logger.info(f"INT8:  {results['int8'].get('throughput', 'N/A'):.2f} sent/sec ({speedup:.2f}x speedup)")
        if 'accuracy' in results['int8']:
            logger.info(f"       Accuracy: {results['int8']['accuracy']['mean_similarity']:.6f} cosine similarity")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quantize and compare BGE-large-en-v1.5 model")
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Model name or path"
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "int8", "compare"],
        default="compare",
        help="Precision to use (default: compare all)"
    )
    parser.add_argument(
        "--no-accuracy",
        action="store_true",
        help="Skip accuracy comparison"
    )
    
    args = parser.parse_args()
    
    if args.precision == "compare":
        compare_quantizations(
            model_name=args.model,
            compare_accuracy=not args.no_accuracy
        )
    else:
        logger.info(f"Loading model in {args.precision.upper()} precision...")
        if args.precision == "fp32":
            model = load_fp32_model(args.model)
        elif args.precision == "fp16":
            model = load_fp16_model(args.model)
        elif args.precision == "int8":
            model = load_int8_model(args.model)
        
        logger.info(f"Model loaded successfully in {args.precision.upper()} precision")

