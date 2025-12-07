"""
Convert BGE-large-en-v1.5 model to ONNX format for visualization in Netron.

This script exports the SentenceTransformer model to ONNX format, which can be
visualized using Netron (https://github.com/lutzroeder/netron).
"""

import torch
from sentence_transformers import SentenceTransformer
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_to_onnx(model_name="BAAI/bge-large-en-v1.5", output_path="bge-large-en-v1.5.onnx", max_seq_length=512):
    """
    Convert a SentenceTransformer model to ONNX format.
    
    Args:
        model_name: HuggingFace model identifier
        output_path: Path to save the ONNX model
        max_seq_length: Maximum sequence length for the model
    """
    logger.info(f"Loading model: {model_name}")
    
    # Load the SentenceTransformer model
    model = SentenceTransformer(model_name)
    model.eval()
    
    logger.info("Model loaded successfully")
    logger.info(f"Model architecture: {model.get_sentence_embedding_dimension()} dimensions")
    
    # Get the underlying transformer model
    # SentenceTransformer wraps the model, we need to access the underlying transformer
    transformer_model = model[0].auto_model
    
    logger.info("Preparing dummy input...")
    
    # Create dummy input matching the model's expected format
    # Input shape: (batch_size, sequence_length)
    dummy_input_ids = torch.randint(0, 30522, (1, max_seq_length), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, max_seq_length), dtype=torch.long)
    
    logger.info(f"Input shape: {dummy_input_ids.shape}")
    
    # Define input names
    input_names = ["input_ids", "attention_mask"]
    output_names = ["last_hidden_state", "pooler_output"]
    
    # Define dynamic axes for variable batch size and sequence length
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        "pooler_output": {0: "batch_size"}
    }
    
    logger.info("Exporting model to ONNX format...")
    logger.info("This may take a few minutes for large models...")
    
    try:
        # Export to ONNX
        torch.onnx.export(
            transformer_model,
            (dummy_input_ids, dummy_attention_mask),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,  # Use opset 14 for better transformer support
            do_constant_folding=True,
            export_params=True,
            verbose=False
        )
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
        logger.info(f"✅ Model successfully exported to: {output_path}")
        logger.info(f"   File size: {file_size:.2f} MB")
        logger.info(f"\nTo visualize in Netron, run:")
        logger.info(f"   netron {output_path}")
        logger.info(f"\nOr open it directly in Netron app/web interface")
        
    except Exception as e:
        logger.error(f"Error exporting model: {str(e)}")
        logger.error("Make sure you have onnx installed: pip install onnx")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert BGE-large-en-v1.5 to ONNX format")
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Model name or path (default: BAAI/bge-large-en-v1.5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="bge-large-en-v1.5.onnx",
        help="Output ONNX file path (default: bge-large-en-v1.5.onnx)"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )
    
    args = parser.parse_args()
    
    convert_to_onnx(
        model_name=args.model,
        output_path=args.output,
        max_seq_length=args.max_seq_length
    )

