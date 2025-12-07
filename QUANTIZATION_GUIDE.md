# Model Quantization Guide: FP32 → FP16 → INT8

## Current Model Precision

**BGE-large-en-v1.5 by default uses FP32 (32-bit floating point)** when loaded with standard PyTorch/SentenceTransformers.

- **Model Size**: ~1.2-1.34 GB (FP32)
- **Memory Usage**: ~1.2-1.5 GB VRAM/RAM (inference)
- **Parameters**: 335 million × 4 bytes = ~1.34 GB

## Quantization Options

### 1. FP16 (16-bit Floating Point) - Half Precision

**Size Reduction:**
- **Model Size**: ~0.67 GB (50% reduction)
- **Memory Usage**: ~0.7-0.9 GB VRAM/RAM
- **Speed**: 1.5-2x faster inference (on GPU with Tensor Cores)

**Performance Impact:**
- ✅ **Accuracy**: Typically maintains 99.5-99.9% of FP32 accuracy
- ✅ **Speed**: Significant speedup on modern GPUs (NVIDIA with Tensor Cores)
- ✅ **Memory**: 50% reduction in memory footprint
- ⚠️ **CPU**: May be slower on CPU (no hardware acceleration)
- ⚠️ **Overflow Risk**: Slight risk of numerical overflow in extreme cases

**Best For:**
- GPU inference (especially NVIDIA GPUs with Tensor Cores)
- When you need speed + quality balance
- Production deployments with GPU resources

**Accuracy Loss:**
- Typically < 0.5% on embedding quality metrics
- Cosine similarity scores remain very close to FP32
- Retrieval performance (R@10, MRR) typically unchanged

### 2. INT8 (8-bit Integer) - Quantized

**Size Reduction:**
- **Model Size**: ~0.34 GB (75% reduction)
- **Memory Usage**: ~0.4-0.5 GB VRAM/RAM
- **Speed**: 2-4x faster inference (on supported hardware)

**Performance Impact:**
- ⚠️ **Accuracy**: May lose 1-3% accuracy depending on quantization method
- ✅ **Speed**: Significant speedup on both GPU and CPU
- ✅ **Memory**: 75% reduction in memory footprint
- ✅ **CPU**: Works well on CPU (integer operations are fast)
- ⚠️ **Calibration**: Requires calibration dataset for best results

**Quantization Methods:**

#### a) Dynamic Quantization (PyTorch)
- Quantizes weights to INT8, activations remain FP32
- Easy to implement, minimal accuracy loss
- Good for CPU inference

#### b) Static Quantization (PyTorch)
- Quantizes both weights and activations to INT8
- Requires calibration dataset
- Better compression, slightly more accuracy loss

#### c) QAT (Quantization-Aware Training)
- Model trained with quantization in mind
- Best accuracy retention
- Requires retraining

**Best For:**
- CPU inference
- Memory-constrained environments
- Edge devices
- When slight accuracy loss is acceptable

**Accuracy Loss:**
- Typically 1-3% on embedding quality
- May see 2-5% degradation in retrieval metrics
- Still very usable for most applications

### 3. INT4 / 4-bit Quantization

**Size Reduction:**
- **Model Size**: ~0.17 GB (87.5% reduction)
- **Memory Usage**: ~0.2-0.3 GB VRAM/RAM

**Performance Impact:**
- ⚠️ **Accuracy**: 3-5% accuracy loss typical
- ✅ **Memory**: Massive reduction
- ⚠️ **Speed**: May be slower due to dequantization overhead
- ⚠️ **Compatibility**: Limited framework support

**Best For:**
- Extreme memory constraints
- Experimental use
- Not recommended for production without testing

## Performance Comparison Table

| Precision | Model Size | Memory (Inference) | Speed (GPU) | Speed (CPU) | Accuracy | Best Use Case |
|-----------|------------|---------------------|-------------|-------------|----------|--------------|
| **FP32** | 1.34 GB | 1.2-1.5 GB | 1.0x (baseline) | 1.0x (baseline) | 100% | Maximum accuracy |
| **FP16** | 0.67 GB | 0.7-0.9 GB | 1.5-2x faster | 0.8-1.2x | 99.5-99.9% | GPU inference |
| **INT8** | 0.34 GB | 0.4-0.5 GB | 2-4x faster | 2-3x faster | 97-99% | CPU/Edge devices |
| **INT4** | 0.17 GB | 0.2-0.3 GB | 1-2x faster | 1.5-2x faster | 95-97% | Extreme constraints |

## Memory Breakdown

### FP32 (Current)
```
Model Weights:     335M params × 4 bytes = 1.34 GB
Activations:       ~100-200 MB (batch-dependent)
Total:            ~1.5 GB
```

### FP16
```
Model Weights:     335M params × 2 bytes = 0.67 GB
Activations:       ~50-100 MB
Total:            ~0.8 GB
```

### INT8
```
Model Weights:     335M params × 1 byte = 0.34 GB
Activations:       ~25-50 MB (if quantized)
Total:            ~0.4 GB
```

## Speed Implications

### GPU (NVIDIA with Tensor Cores)
- **FP32**: Baseline speed
- **FP16**: 1.5-2x faster (Tensor Cores optimized for FP16)
- **INT8**: 2-4x faster (on newer GPUs with INT8 support)
- **Note**: Older GPUs may not see INT8 speedup

### CPU
- **FP32**: Baseline speed
- **FP16**: May be slower (no hardware acceleration)
- **INT8**: 2-3x faster (integer operations are fast)
- **Note**: INT8 is often best choice for CPU

## Accuracy Impact on Embedding Quality

### Cosine Similarity Preservation
- **FP16**: Typically maintains >99.5% correlation with FP32 embeddings
- **INT8**: Typically maintains >97% correlation
- **INT4**: Typically maintains >95% correlation

### Retrieval Performance (R@10, MRR)
- **FP16**: Usually unchanged or <0.5% degradation
- **INT8**: May see 1-3% degradation
- **INT4**: May see 3-5% degradation

### Embedding Distance Metrics
- **FP16**: Embeddings are very close (L2 distance < 0.01 typically)
- **INT8**: Embeddings may differ more (L2 distance < 0.05 typically)
- **INT4**: Larger differences (L2 distance < 0.1 typically)

## Implementation Options

### 1. SentenceTransformers Native FP16
```python
from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer("BAAI/bge-large-en-v1.5")
# Automatically uses FP16 if GPU available and CUDA supports it
```

### 2. PyTorch Dynamic Quantization (INT8)
```python
import torch
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")
# Quantize the underlying transformer
model[0].auto_model = torch.quantization.quantize_dynamic(
    model[0].auto_model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### 3. BitsAndBytes (8-bit/4-bit)
```python
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
import torch

# Load with 8-bit quantization
model = AutoModel.from_pretrained(
    "BAAI/bge-large-en-v1.5",
    load_in_8bit=True,
    device_map="auto"
)
```

### 4. ONNX Quantization
```python
# Convert to ONNX then quantize
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# After ONNX conversion
quantize_dynamic(
    "bge-large-en-v1.5.onnx",
    "bge-large-en-v1.5-int8.onnx",
    weight_type=QuantType.QUInt8
)
```

## Recommendations

### For GPU Inference (Recommended: FP16)
- ✅ Best speed/accuracy tradeoff
- ✅ Minimal accuracy loss
- ✅ Significant memory savings
- ✅ Native support in modern frameworks

### For CPU Inference (Recommended: INT8)
- ✅ Best speed on CPU
- ✅ Good memory savings
- ✅ Acceptable accuracy loss
- ✅ Works well with ONNX Runtime

### For Maximum Accuracy (Recommended: FP32)
- ✅ No quantization errors
- ✅ Best embedding quality
- ⚠️ Highest memory usage
- ⚠️ Slower inference

### For Memory-Constrained (Recommended: INT8)
- ✅ 75% memory reduction
- ✅ Still good accuracy
- ✅ Fast inference
- ⚠️ Requires quantization setup

## Testing Quantization Impact

Before deploying quantized models, test:

1. **Embedding Quality**:
   ```python
   # Compare cosine similarity between FP32 and quantized
   embeddings_fp32 = model_fp32.encode(texts)
   embeddings_quant = model_quant.encode(texts)
   similarity = cosine_similarity(embeddings_fp32, embeddings_quant)
   ```

2. **Retrieval Performance**:
   - Test on your specific retrieval tasks
   - Compare R@10, R@100, MRR metrics
   - Ensure degradation is acceptable

3. **Speed Benchmarking**:
   - Measure inference time
   - Test with realistic batch sizes
   - Compare GPU vs CPU performance

## Summary

**Current Setup (FP32):**
- Model: 1.34 GB
- Memory: ~1.5 GB
- Accuracy: 100% (baseline)
- Speed: Baseline

**Recommended for Most Use Cases:**
- **GPU**: FP16 (0.67 GB, 1.5-2x faster, 99.5%+ accuracy)
- **CPU**: INT8 (0.34 GB, 2-3x faster, 97-99% accuracy)

**Trade-offs:**
- Lower precision = Less memory + Faster inference
- Lower precision = Potential accuracy loss (usually minimal for FP16, small for INT8)
- Choose based on your constraints: memory, speed, or accuracy priority

