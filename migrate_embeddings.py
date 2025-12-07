"""
Utility script to migrate embeddings from one model to another.

This script helps regenerate embeddings when switching embedding models.
Supports various vector stores and batch processing for large datasets.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging
from typing import List, Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_texts_from_file(file_path: str) -> List[str]:
    """Load texts from various file formats."""
    file_path = Path(file_path)
    
    if file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return [item if isinstance(item, str) else item.get('text', '') for item in data]
            elif isinstance(data, dict):
                return [data.get('text', '')] if 'text' in data else list(data.values())
    elif file_path.suffix == '.jsonl':
        texts = []
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                texts.append(data.get('text', ''))
        return texts
    elif file_path.suffix == '.txt':
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

def regenerate_embeddings(
    texts: List[str],
    model_name: str = "BAAI/bge-large-en-v1.5",
    batch_size: int = 32,
    show_progress: bool = True,
    output_file: Optional[str] = None
) -> np.ndarray:
    """
    Regenerate embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed
        model_name: Name of the embedding model
        batch_size: Batch size for processing
        show_progress: Show progress bar
        output_file: Optional file to save embeddings
    
    Returns:
        numpy array of embeddings
    """
    logger.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    embedding_dim = model.get_sentence_embedding_dimension()
    logger.info(f"Model embedding dimension: {embedding_dim}")
    logger.info(f"Processing {len(texts)} texts in batches of {batch_size}")
    
    all_embeddings = []
    
    # Process in batches
    iterator = tqdm(range(0, len(texts), batch_size), desc="Generating embeddings") if show_progress else range(0, len(texts), batch_size)
    
    for i in iterator:
        batch = texts[i:i+batch_size]
        embeddings = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        all_embeddings.append(embeddings)
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)
    
    logger.info(f"Generated {len(all_embeddings)} embeddings")
    logger.info(f"Embedding shape: {all_embeddings.shape}")
    
    # Save if output file specified
    if output_file:
        output_path = Path(output_file)
        np.save(output_path, all_embeddings)
        logger.info(f"Saved embeddings to {output_path}")
        
        # Also save metadata
        metadata = {
            'model': model_name,
            'dimension': embedding_dim,
            'num_embeddings': len(all_embeddings),
            'texts_file': None  # Could store reference to texts
        }
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
    
    return all_embeddings

def compare_embeddings(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
    texts: List[str],
    top_k: int = 5
):
    """Compare two sets of embeddings to validate migration."""
    from sklearn.metrics.pairwise import cosine_similarity
    
    logger.info("Comparing embeddings...")
    
    # Compute pairwise similarities
    similarities = cosine_similarity(embeddings1, embeddings2)
    
    # Get diagonal (same text, different models)
    diagonal = np.diag(similarities)
    
    logger.info(f"Mean similarity: {np.mean(diagonal):.4f}")
    logger.info(f"Min similarity: {np.min(diagonal):.4f}")
    logger.info(f"Max similarity: {np.max(diagonal):.4f}")
    logger.info(f"Std similarity: {np.std(diagonal):.4f}")
    
    # Show top-k most similar pairs
    if top_k > 0:
        logger.info(f"\nTop {top_k} most similar embeddings:")
        top_indices = np.argsort(diagonal)[-top_k:][::-1]
        for idx in top_indices:
            logger.info(f"  Text {idx}: {texts[idx][:50]}...} - Similarity: {diagonal[idx]:.4f}")

def export_for_chroma(
    texts: List[str],
    embeddings: np.ndarray,
    ids: Optional[List[str]] = None,
    metadata: Optional[List[Dict]] = None,
    output_dir: str = "chroma_export"
):
    """Export embeddings in ChromaDB format."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if ids is None:
        ids = [f"doc_{i}" for i in range(len(texts))]
    
    if metadata is None:
        metadata = [{}] * len(texts)
    
    # Save in ChromaDB-compatible format
    data = {
        'ids': ids,
        'documents': texts,
        'embeddings': embeddings.tolist(),
        'metadatas': metadata
    }
    
    output_file = output_path / "chroma_data.json"
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Exported ChromaDB data to {output_file}")
    logger.info(f"  {len(texts)} documents")
    logger.info(f"  {embeddings.shape[1]}-dimensional embeddings")

def export_for_pinecone(
    texts: List[str],
    embeddings: np.ndarray,
    ids: Optional[List[str]] = None,
    metadata: Optional[List[Dict]] = None,
    output_file: str = "pinecone_vectors.json"
):
    """Export embeddings in Pinecone format."""
    if ids is None:
        ids = [f"doc_{i}" for i in range(len(texts))]
    
    if metadata is None:
        metadata = [{}] * len(texts)
    
    # Pinecone format: list of (id, vector, metadata) tuples
    vectors = []
    for i, (text_id, embedding, meta) in enumerate(zip(ids, embeddings, metadata)):
        vectors.append({
            'id': text_id,
            'values': embedding.tolist(),
            'metadata': {**meta, 'text': texts[i]}  # Include text in metadata
        })
    
    with open(output_file, 'w') as f:
        json.dump(vectors, f, indent=2)
    
    logger.info(f"Exported Pinecone vectors to {output_file}")
    logger.info(f"  {len(vectors)} vectors")
    logger.info(f"  {embeddings.shape[1]}-dimensional embeddings")

def main():
    parser = argparse.ArgumentParser(description="Migrate embeddings to new model")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file with texts (JSON, JSONL, or TXT)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Embedding model name (default: BAAI/bge-large-en-v1.5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for embeddings (.npy)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["numpy", "chroma", "pinecone"],
        default="numpy",
        help="Output format (default: numpy)"
    )
    parser.add_argument(
        "--ids-file",
        type=str,
        help="Optional file with document IDs (one per line)"
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        help="Optional JSON file with metadata for each document"
    )
    
    args = parser.parse_args()
    
    # Load texts
    logger.info(f"Loading texts from {args.input}")
    texts = load_texts_from_file(args.input)
    logger.info(f"Loaded {len(texts)} texts")
    
    # Load IDs if provided
    ids = None
    if args.ids_file:
        with open(args.ids_file, 'r') as f:
            ids = [line.strip() for line in f]
        if len(ids) != len(texts):
            logger.warning(f"IDs count ({len(ids)}) doesn't match texts count ({len(texts)})")
            ids = None
    
    # Load metadata if provided
    metadata = None
    if args.metadata_file:
        with open(args.metadata_file, 'r') as f:
            metadata = json.load(f)
        if len(metadata) != len(texts):
            logger.warning(f"Metadata count ({len(metadata)}) doesn't match texts count ({len(texts)})")
            metadata = None
    
    # Generate embeddings
    output_file = args.output or f"embeddings_{Path(args.input).stem}.npy"
    embeddings = regenerate_embeddings(
        texts=texts,
        model_name=args.model,
        batch_size=args.batch_size,
        output_file=output_file if args.format == "numpy" else None
    )
    
    # Export in requested format
    if args.format == "chroma":
        export_for_chroma(texts, embeddings, ids, metadata)
    elif args.format == "pinecone":
        export_for_pinecone(texts, embeddings, ids, metadata, args.output or "pinecone_vectors.json")
    
    logger.info("Migration complete!")

if __name__ == "__main__":
    main()



