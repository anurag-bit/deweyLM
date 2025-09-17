#!/usr/bin/env python3
"""
High-Performance Embedding Generation and RAG System
Optimized for AMD MI300X GPU with 192GB VRAM

This script creates embeddings using google/embeddinggemma-300m and sets up
a RAG system with google/gemma-3-270m-it for the Dewey Classification text.
"""

import os
import json
import time
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM, 
    pipeline, BitsAndBytesConfig
)
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HighPerformanceRAGSystem:
    def __init__(self, 
                 processed_data_dir: str = "processed_data",
                 models_cache_dir: str = "models_cache",
                 vector_db_dir: str = "vector_db"):
        
        self.processed_data_dir = Path(processed_data_dir)
        self.models_cache_dir = Path(models_cache_dir)
        self.vector_db_dir = Path(vector_db_dir)
        
        # Create directories
        self.models_cache_dir.mkdir(exist_ok=True)
        self.vector_db_dir.mkdir(exist_ok=True)
        
        # Check GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Model configurations
        self.embedding_model_name = "google/embeddinggemma-300m"
        self.llm_model_name = "google/gemma-3-270m-it"
        
        # Initialize models as None (will be loaded when needed)
        self.embedding_model = None
        self.embedding_tokenizer = None
        self.llm_model = None
        self.llm_tokenizer = None
        self.vector_index = None
        
        # Load processed text chunks
        self.chunks_data = self.load_processed_chunks()
        
    def load_processed_chunks(self) -> List[Dict]:
        """Load the processed text chunks."""
        chunks_file = self.processed_data_dir / "text_chunks.json"
        
        if not chunks_file.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        logger.info(f"Loaded {len(chunks)} text chunks")
        return chunks
    
    def setup_embedding_model(self):
        """Load and configure the embedding model."""
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        
        try:
            # Try loading with SentenceTransformers first (more optimized)
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                cache_folder=str(self.models_cache_dir),
                device=self.device
            )
            logger.info("Embedding model loaded with SentenceTransformers")
            
        except Exception as e:
            logger.warning(f"SentenceTransformers loading failed: {e}")
            logger.info("Falling back to transformers library")
            
            # Fallback to transformers
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(
                self.embedding_model_name,
                cache_dir=str(self.models_cache_dir)
            )
            
            self.embedding_model = AutoModel.from_pretrained(
                self.embedding_model_name,
                cache_dir=str(self.models_cache_dir),
                torch_dtype=torch.float16,
                device_map="auto"
            )
    
    def setup_llm_model(self):
        """Load and configure the LLM for generation."""
        logger.info(f"Loading LLM model: {self.llm_model_name}")
        
        # Configure quantization for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            self.llm_model_name,
            cache_dir=str(self.models_cache_dir)
        )
        
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            cache_dir=str(self.models_cache_dir),
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Set padding token
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
    
    def generate_embeddings(self, batch_size: int = 32, max_length: int = 512) -> np.ndarray:
        """Generate embeddings for all text chunks."""
        if self.embedding_model is None:
            self.setup_embedding_model()
        
        logger.info(f"Generating embeddings for {len(self.chunks_data)} chunks")
        
        # Extract texts
        texts = [chunk['text'] for chunk in self.chunks_data]
        
        # Generate embeddings in batches
        all_embeddings = []
        
        if isinstance(self.embedding_model, SentenceTransformer):
            # Use SentenceTransformer's batch processing
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            all_embeddings = embeddings
            
        else:
            # Manual batching for transformers model
            for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize batch
                inputs = self.embedding_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = self.embedding_model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                    embeddings = embeddings.cpu().numpy()
                
                all_embeddings.append(embeddings)
            
            all_embeddings = np.vstack(all_embeddings)
        
        logger.info(f"Generated embeddings shape: {all_embeddings.shape}")
        
        # Save embeddings
        embeddings_file = self.vector_db_dir / "embeddings.npy"
        np.save(embeddings_file, all_embeddings)
        
        return all_embeddings
    
    def create_vector_index(self, embeddings: Optional[np.ndarray] = None):
        """Create FAISS vector index for similarity search."""
        if embeddings is None:
            embeddings_file = self.vector_db_dir / "embeddings.npy"
            if embeddings_file.exists():
                embeddings = np.load(embeddings_file)
            else:
                embeddings = self.generate_embeddings()
        
        logger.info("Creating FAISS vector index")
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        
        # Use IndexFlatIP for cosine similarity (after normalization)
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create index
        self.vector_index = faiss.IndexFlatIP(dimension)
        self.vector_index.add(embeddings.astype(np.float32))
        
        logger.info(f"Created FAISS index with {self.vector_index.ntotal} vectors")
        
        # Save index
        index_file = self.vector_db_dir / "faiss_index.bin"
        faiss.write_index(self.vector_index, str(index_file))
        
        # Save chunk metadata
        metadata_file = self.vector_db_dir / "chunks_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.chunks_data, f, indent=2, ensure_ascii=False)
    
    def load_vector_index(self):
        """Load existing FAISS index."""
        index_file = self.vector_db_dir / "faiss_index.bin"
        metadata_file = self.vector_db_dir / "chunks_metadata.json"
        
        if not index_file.exists() or not metadata_file.exists():
            logger.info("Vector index not found, creating new one...")
            self.create_vector_index()
            return
        
        logger.info("Loading existing FAISS index")
        self.vector_index = faiss.read_index(str(index_file))
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.chunks_data = json.load(f)
        
        logger.info(f"Loaded FAISS index with {self.vector_index.ntotal} vectors")
    
    def search_similar_chunks(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar chunks using vector similarity."""
        if self.vector_index is None:
            self.load_vector_index()
        
        if self.embedding_model is None:
            self.setup_embedding_model()
        
        # Generate query embedding
        if isinstance(self.embedding_model, SentenceTransformer):
            query_embedding = self.embedding_model.encode([query])
        else:
            inputs = self.embedding_tokenizer(
                query,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                query_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        # Normalize query embedding
        faiss.normalize_L2(query_embedding.astype(np.float32))
        
        # Search
        scores, indices = self.vector_index.search(query_embedding.astype(np.float32), k)
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            chunk = self.chunks_data[idx]
            results.append({
                'rank': i + 1,
                'score': float(score),
                'chunk_id': chunk['chunk_id'],
                'text': chunk['text'],
                'source_page': chunk.get('source_page', 'unknown'),
                'word_count': chunk['word_count']
            })
        
        return results
    
    def generate_response(self, query: str, context_chunks: List[Dict], 
                         max_length: int = 512) -> str:
        """Generate response using the LLM with retrieved context."""
        if self.llm_model is None:
            self.setup_llm_model()
        
        # Prepare context
        context = "\n\n".join([
            f"[Source: Page {chunk['source_page']}]\n{chunk['text']}"
            for chunk in context_chunks
        ])
        
        # Create prompt
        prompt = f"""Based on the following information from the Dewey Decimal Classification system, please answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        # Tokenize and generate
        inputs = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.llm_tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.llm_tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def query_rag_system(self, query: str, k: int = 5, max_response_length: int = 512) -> Dict:
        """Complete RAG pipeline: retrieve and generate."""
        logger.info(f"Processing query: {query}")
        
        # Retrieve similar chunks
        similar_chunks = self.search_similar_chunks(query, k=k)
        
        # Generate response
        response = self.generate_response(query, similar_chunks, max_response_length)
        
        return {
            'query': query,
            'response': response,
            'retrieved_chunks': similar_chunks,
            'timestamp': time.time()
        }

def main():
    """Demonstration of the RAG system."""
    
    # Check if processed data exists
    processed_data_dir = Path("processed_data")
    if not processed_data_dir.exists():
        print("Processed data not found. Please run process_pdf_server.py first.")
        return
    
    # Initialize RAG system
    print("Initializing RAG system...")
    rag_system = HighPerformanceRAGSystem()
    
    # Setup models and vector database
    print("Setting up embedding model...")
    rag_system.setup_embedding_model()
    
    print("Generating embeddings...")
    embeddings = rag_system.generate_embeddings(batch_size=64)
    
    print("Creating vector index...")
    rag_system.create_vector_index(embeddings)
    
    print("Setting up LLM...")
    rag_system.setup_llm_model()
    
    # Example queries
    example_queries = [
        "What is the Dewey Decimal Classification for computer science?",
        "How are books about religion classified in the Dewey system?",
        "What classification numbers are used for mathematics?",
        "Explain the structure of the Dewey Decimal Classification system.",
        "What are the main classes in the 000-099 range?"
    ]
    
    print("\n" + "="*60)
    print("RAG SYSTEM READY - RUNNING EXAMPLE QUERIES")
    print("="*60)
    
    for query in example_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        
        result = rag_system.query_rag_system(query, k=3)
        
        print(f"Response: {result['response']}")
        print(f"Sources used: {len(result['retrieved_chunks'])} chunks")
        print()
    
    print("RAG system setup complete!")
    print("You can now use the system for interactive queries.")

if __name__ == "__main__":
    main()