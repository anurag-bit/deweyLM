# Dewey Decimal Classification RAG System
## High-Performance Implementation for AMD MI300X GPU Server

This project implements a complete RAG (Retrieval-Augmented Generation) system for the Dewey Decimal Classification using state-of-the-art language models, optimized for high-performance GPU servers.

### Server Specifications
- **GPU**: AMD MI300X (192GB VRAM)
- **RAM**: 240GB
- **CPU**: 20 vCPUs
- **Storage**: 720GB NVMe SSD + 5TB Scratch Disk

### Models Used
- **Embedding Model**: `google/embeddinggemma-300m`
- **Language Model**: `google/gemma-3-270m-it`

## Quick Start

### 🚀 One-Command Launch (Recommended)
```bash
# Run the quick start script
./start.sh
```

### 📖 Step-by-Step Setup

#### 1. Environment Setup

```bash
# Clone the repository and navigate to it
git clone <your-repo-url>
cd dewey

# Create conda environment
conda env create -f environment_server.yml
conda activate dewey-rag-server

# Alternative: Use requirements file
pip install -r requirements_server.txt
```

#### 2. System Setup and Launch

```bash
# Launch the interactive launcher
python launcher.py

# Or run complete pipeline directly
python deploy_server.py --step full
```

### 🎯 Interface Options

#### Option 1: Streamlit GUI (Recommended for beginners)
- **Visual web interface** with charts and visualizations
- **Interactive controls** for search parameters
- **Source highlighting** and detailed results
- **Export capabilities** for results

```bash
# From launcher menu (option 1) or directly:
streamlit run streamlit_gui.py
```

#### Option 2: Command Line Interface
- **Fast terminal interface** for power users
- **Minimal resource usage**
- **Scriptable and automatable**
- **Quick testing and debugging**

```bash
# From launcher menu (option 2) or directly:
python cli_interface.py
```

#### Option 3: Python API
- **Direct programmatic access**
- **Integration into other applications**
- **Batch processing capabilities**
- **Custom analysis and workflows**

```python
from rag_system_server import HighPerformanceRAGSystem

# Initialize and use
rag = HighPerformanceRAGSystem()
rag.load_vector_index()
result = rag.query_rag_system("What is computer science classification?")
```

## Project Structure

```
dewey/
├── dewey-decimal-classification-*.pdf    # Source PDF
├── start.sh                              # Quick start script
├── launcher.py                           # Interactive interface launcher
├── setup_server.sh                       # Server setup script
├── environment_server.yml                # Conda environment
├── requirements_server.txt               # Pip requirements
├── process_pdf_server.py                 # PDF processing
├── rag_system_server.py                  # RAG implementation
├── create_finetuning_dataset.py          # Fine-tuning data prep
├── deploy_server.py                      # Main deployment script
├── streamlit_gui.py                      # Web GUI interface
├── cli_interface.py                      # Command line interface
├── USER_GUIDE.md                         # Comprehensive user guide
├── processed_data/                       # Extracted text data
│   ├── extracted_pages.json
│   ├── text_chunks.json
│   ├── dewey_classification_full_text.txt
│   └── processing_stats.json
├── vector_db/                            # Vector database
│   ├── embeddings.npy
│   ├── faiss_index.bin
│   └── chunks_metadata.json
├── fine_tuning_data/                     # Training datasets
│   ├── instruction_dataset.json
│   ├── conversational_dataset.json
│   ├── completion_dataset.json
│   └── combined_training_dataset.json
└── models_cache/                         # Downloaded models
```

## Key Features

### High-Performance PDF Processing
- **Multi-threaded**: Utilizes all CPU cores for parallel page processing
- **Multiple engines**: PyMuPDF, pdfplumber, and PyPDF2 fallback
- **Smart chunking**: Context-aware text segmentation for optimal embeddings
- **Memory efficient**: Handles large PDFs without memory issues

### Advanced RAG System
- **Vector Search**: FAISS with cosine similarity for fast retrieval
- **Batch Processing**: Optimized embedding generation
- **GPU Acceleration**: Full utilization of AMD MI300X capabilities
- **Contextual Responses**: Retrieves relevant chunks for accurate answers

### Fine-tuning Ready
- **Multiple Formats**: Instruction, conversational, and completion datasets
- **Quality Control**: Filtered and validated training examples
- **LoRA/QLoRA Ready**: Optimized for parameter-efficient fine-tuning

## Performance Optimizations

### For AMD MI300X GPU
```python
# Optimal batch sizes for 192GB VRAM
EMBEDDING_BATCH_SIZE = 128
LLM_BATCH_SIZE = 32
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
```

### Memory Management
- **4-bit quantization** for LLM inference
- **Float16** precision for embeddings
- **Gradient checkpointing** for fine-tuning
- **Dynamic batching** based on available memory

## Usage Examples

### 🖥️ Streamlit GUI Examples

The Streamlit interface provides an intuitive way to interact with the system:

**Features:**
- **Interactive Query Box**: Type questions naturally
- **Real-time Parameters**: Adjust search depth and response length
- **Visual Results**: Charts showing source distribution and relevance scores
- **Source Explorer**: Detailed view of retrieved document chunks
- **Export Options**: Save results as JSON or text files

**Sample Queries in GUI:**
1. Type: "What is the classification for artificial intelligence?"
2. Adjust "Chunks to retrieve" slider (3-7 recommended)
3. Click "Search" and explore the tabbed results

### 💻 CLI Examples

Fast command-line interaction for power users:

```bash
$ python cli_interface.py
🔍 > What is the classification for computer science?

💬 Answer: Computer science is classified under 004 in the Dewey Decimal 
Classification system. This falls within the 000-099 range (Computer 
science, information, and general works)...

📚 Sources used:
  1. Page 247 (similarity: 0.891)
     004 Computer science, programming, programs, data...
  2. Page 156 (similarity: 0.834)  
     The 000s are used for computer science, knowledge...

⏱️ Response time: 2.31 seconds
```

### 🐍 Python API Examples
```python
# Example queries
queries = [
    "What is the Dewey classification for computer science?",
    "How are philosophy books organized?",
    "What numbers are used for mathematics?",
    "Explain the 400s section of Dewey classification"
]

for query in queries:
    result = rag.query_rag_system(query, k=5)
    print(f"Q: {query}")
    print(f"A: {result['response']}")
    print("-" * 50)
```

### 2. Fine-tuning Setup
```python
# Load training dataset
with open('fine_tuning_data/combined_training_dataset.json', 'r') as f:
    training_data = json.load(f)

# Use with Hugging Face Trainer or similar
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# Configure LoRA for efficient fine-tuning
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)
```

## Advanced Configuration

### Vector Database Tuning
```python
# For large-scale deployment
INDEX_TYPE = "IndexIVFFlat"  # For faster search
NLIST = 100  # Number of clusters
NPROBE = 10  # Search clusters
```

### Model Quantization
```python
# 4-bit quantization for maximum efficiency
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
```

## Monitoring and Logging

The system includes comprehensive logging and monitoring:

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.INFO)

# Monitor GPU memory usage
import torch
print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
```

## Interface Comparison

| Feature | Streamlit GUI | CLI Interface | Python API |
|---------|---------------|---------------|------------|
| **Ease of Use** | ⭐⭐⭐⭐⭐ Beginner-friendly | ⭐⭐⭐⭐ Power users | ⭐⭐⭐ Developers |
| **Visual Features** | ⭐⭐⭐⭐⭐ Charts, graphs, tables | ⭐⭐ Text-only | ⭐ Programmatic |
| **Performance** | ⭐⭐⭐ Web overhead | ⭐⭐⭐⭐⭐ Fastest | ⭐⭐⭐⭐ Direct access |
| **Customization** | ⭐⭐⭐ UI controls | ⭐⭐⭐⭐ Command options | ⭐⭐⭐⭐⭐ Full control |
| **Batch Processing** | ❌ Single queries | ⭐⭐ Script-friendly | ⭐⭐⭐⭐⭐ Automated |
| **Export/Save** | ⭐⭐⭐⭐⭐ JSON/Text download | ⭐⭐ Copy/paste | ⭐⭐⭐⭐⭐ Programmatic |
| **Learning Curve** | Low | Medium | High |

**Choose Streamlit GUI if:**
- You want visual exploration of results
- You're new to the system
- You need to share results with others
- You prefer point-and-click interfaces

**Choose CLI if:**
- You want fast, lightweight interaction
- You're comfortable with terminal/command line
- You need to script or automate queries
- You have limited system resources

**Choose Python API if:**
- You're integrating into existing applications
- You need custom processing of results
- You want to build your own interface
- You need batch processing capabilities

## Deployment Options

### 1. FastAPI Web Service
```python
# Create API endpoint
from fastapi import FastAPI
app = FastAPI()

@app.post("/query")
async def query_endpoint(query: str):
    result = rag.query_rag_system(query)
    return result
```

### 2. Streamlit Interface
```python
# Interactive web interface
import streamlit as st

st.title("Dewey Classification RAG System")
query = st.text_input("Ask about Dewey classifications:")
if query:
    result = rag.query_rag_system(query)
    st.write(result['response'])
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch sizes in configuration
2. **Model Loading**: Ensure sufficient disk space for model cache
3. **CUDA Errors**: Verify ROCm installation for AMD GPU
4. **Slow Processing**: Check if GPU acceleration is enabled

### Performance Monitoring
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi  # or rocm-smi for AMD

# Monitor system resources
htop
iotop
```

## Future Enhancements

1. **Multi-GPU Support**: Distribute processing across multiple GPUs
2. **Streaming Responses**: Real-time response generation
3. **Advanced Retrieval**: Hybrid search with BM25 + vector similarity
4. **Knowledge Graph**: Integrate classification hierarchies
5. **Multi-modal**: Support for images and tables in classifications

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for the transformer models
- Facebook AI Research for FAISS
- The library science community for the Dewey Decimal Classification system

---

For questions or support, please open an issue on the GitHub repository.