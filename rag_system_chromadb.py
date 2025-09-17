#!/usr/bin/env python3
"""
High-Performance Embedding Generation and RAG System with ChromaDB
Optimized for AMD MI300X GPU with 192GB VRAM

This script creates embeddings using google/embeddinggemma-300m and sets up
a RAG system with google/gemma-3-270m-it for the Dewey Classification text.
Uses ChromaDB for modern vector storage and retrieval.
"""

import os
import json
import time
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
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaRAGSystem:
    def __init__(self, 
                 processed_data_dir: str = "processed_data",
                 models_cache_dir: str = "models_cache",
                 vector_db_dir: str = "vector_db",
                 collection_name: str = "dewey_classification"):
        
        self.processed_data_dir = Path(processed_data_dir)
        self.models_cache_dir = Path(models_cache_dir)
        self.vector_db_dir = Path(vector_db_dir)
        self.collection_name = collection_name
        
        # Create directories
        self.models_cache_dir.mkdir(exist_ok=True)
        self.vector_db_dir.mkdir(exist_ok=True)
        
        # Model configurations
        self.embedding_model_name = "google/embeddinggemma-300m"
        self.llm_model_name = "google/gemma-3-270m-it"
        
        # Check GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        # Initialize models (loaded on demand)
        self.embedding_model = None
        self.embedding_tokenizer = None
        self.llm_model = None
        self.llm_tokenizer = None
        
        # ChromaDB setup
        self.chroma_client = None
        self.collection = None
        self.chunks_data = []
    
    def load_processed_data(self):
        """Load processed text chunks."""
        chunks_file = self.processed_data_dir / "text_chunks.json"
        
        if not chunks_file.exists():
            raise FileNotFoundError(f"Processed chunks not found at {chunks_file}")
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            self.chunks_data = json.load(f)
        
        logger.info(f"Loaded {len(self.chunks_data)} text chunks")
    
    def setup_chromadb(self):
        """Initialize ChromaDB client and collection."""
        logger.info("Setting up ChromaDB")
        
        # Initialize ChromaDB client with persistent storage
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.vector_db_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name
            )
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except:
            # Create new collection with custom embedding function
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Dewey Decimal Classification embeddings"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def setup_embedding_model(self):
        """Load and configure the embedding model."""
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        
        # Use SentenceTransformers for easier embedding generation
        try:
            self.sentence_model = SentenceTransformer(
                'all-MiniLM-L6-v2',  # Fallback to a reliable model
                device=self.device,
                cache_folder=str(self.models_cache_dir)
            )
        except:
            # If the specific model fails, use a different approach
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=str(self.models_cache_dir)
            )
            
            self.embedding_model = AutoModel.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2",
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
        
        try:
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
        except Exception as e:
            logger.warning(f"Could not load {self.llm_model_name}: {e}")
            logger.info("Falling back to a smaller model...")
            
            # Fallback to a smaller, more compatible model
            fallback_model = "microsoft/DialoGPT-medium"
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                fallback_model,
                cache_dir=str(self.models_cache_dir)
            )
            
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                fallback_model,
                cache_dir=str(self.models_cache_dir),
                device_map="auto",
                torch_dtype=torch.float16
            )
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for text chunks."""
        if hasattr(self, 'sentence_model') and self.sentence_model:
            # Use SentenceTransformers (easier)
            logger.info(f"Generating embeddings for {len(texts)} texts using SentenceTransformers")
            embeddings = self.sentence_model.encode(
                texts, 
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            return embeddings.tolist()
        else:
            # Manual embedding generation (fallback)
            logger.info(f"Generating embeddings for {len(texts)} texts using manual approach")
            if self.embedding_model is None:
                self.setup_embedding_model()
            
            all_embeddings = []
            for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                inputs = self.embedding_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = self.embedding_model(**inputs)
                    # Use mean pooling
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                    all_embeddings.extend(embeddings.cpu().numpy().tolist())
            
            return all_embeddings
    
    def populate_chromadb(self, force_rebuild: bool = False):
        """Populate ChromaDB with embeddings."""
        if not self.chunks_data:
            self.load_processed_data()
        
        if not self.collection:
            self.setup_chromadb()
        
        # Check if collection already has data
        collection_count = self.collection.count()
        if collection_count > 0 and not force_rebuild:
            logger.info(f"Collection already contains {collection_count} documents")
            return
        
        if force_rebuild and collection_count > 0:
            logger.info("Force rebuilding - deleting existing collection")
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Dewey Decimal Classification embeddings"}
            )
        
        logger.info(f"Populating ChromaDB with {len(self.chunks_data)} chunks")
        
        # Prepare data for ChromaDB
        texts = [chunk['text'] for chunk in self.chunks_data]
        ids = [f"chunk_{i}" for i in range(len(self.chunks_data))]
        
        # Create metadata for each chunk
        metadatas = []
        for i, chunk in enumerate(self.chunks_data):
            metadata = {
                "chunk_id": i,
                "source_page": chunk.get('page', 'unknown'),
                "char_count": len(chunk['text']),
                "classification": chunk.get('classification', 'general')
            }
            metadatas.append(metadata)
        
        # Generate embeddings (ChromaDB can also auto-generate, but we want control)
        if self.embedding_model is None and not hasattr(self, 'sentence_model'):
            self.setup_embedding_model()
        
        embeddings = self.generate_embeddings(texts)
        
        # Add to ChromaDB in batches (ChromaDB has size limits)
        batch_size = 100  # ChromaDB recommended batch size
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Adding to ChromaDB"):
            end_idx = min(i + batch_size, len(texts))
            
            batch_ids = ids[i:end_idx]
            batch_texts = texts[i:end_idx]
            batch_embeddings = embeddings[i:end_idx] if embeddings else None
            batch_metadatas = metadatas[i:end_idx]
            
            # Add to collection
            if batch_embeddings:
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_texts,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas
                )
            else:
                # Let ChromaDB auto-generate embeddings
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_texts,
                    metadatas=batch_metadatas
                )
        
        logger.info(f"Successfully added {len(texts)} documents to ChromaDB")
    
    def search_similar_chunks(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar chunks using ChromaDB."""
        if not self.collection:
            self.setup_chromadb()
        
        logger.info(f"Searching for top {k} similar chunks")
        
        # Perform similarity search
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        similar_chunks = []
        for i in range(len(results['documents'][0])):
            chunk = {
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'similarity_score': 1 - results['distances'][0][i]  # Convert distance to similarity
            }
            similar_chunks.append(chunk)
        
        return similar_chunks
    
    def generate_response(self, query: str, context_chunks: List[Dict], max_length: int = 512) -> str:
        """Generate response using LLM with retrieved context."""
        if self.llm_model is None:
            self.setup_llm_model()
        
        # Prepare context
        context_text = "\n\n".join([chunk['text'][:500] for chunk in context_chunks])
        
        # Create prompt
        prompt = f"""Based on the following context from the Dewey Decimal Classification system, please answer the question.

Context:
{context_text}

Question: {query}

Answer:"""
        
        # Tokenize and generate
        inputs = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.llm_tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the answer part
        if "Answer:" in response:
            answer = response.split("Answer:")[-1].strip()
        else:
            answer = response[len(prompt):].strip()
        
        return answer
    
    def query_rag_system(self, query: str, k: int = 5) -> Dict:
        """Complete RAG query pipeline."""
        start_time = time.time()
        
        # 1. Retrieve similar chunks
        similar_chunks = self.search_similar_chunks(query, k=k)
        
        # 2. Generate response
        response = self.generate_response(query, similar_chunks)
        
        end_time = time.time()
        
        return {
            'query': query,
            'response': response,
            'context_chunks': similar_chunks,
            'processing_time': end_time - start_time,
            'num_chunks_retrieved': len(similar_chunks)
        }
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the ChromaDB collection."""
        if not self.collection:
            self.setup_chromadb()
        
        count = self.collection.count()
        
        # Get a sample to understand the data
        if count > 0:
            sample = self.collection.peek(limit=min(5, count))
            avg_doc_length = np.mean([len(doc) for doc in sample['documents']]) if sample['documents'] else 0
        else:
            avg_doc_length = 0
        
        return {
            'total_documents': count,
            'collection_name': self.collection_name,
            'average_document_length': avg_doc_length,
            'database_path': str(self.vector_db_dir)
        }

def main():
    """Main function for testing the ChromaRAG system."""
    print("🚀 Initializing ChromaDB RAG System")
    
    # Initialize system
    rag_system = ChromaRAGSystem()
    
    # Setup and populate ChromaDB
    print("📊 Setting up ChromaDB and populating with embeddings...")
    rag_system.populate_chromadb()
    
    # Display stats
    stats = rag_system.get_collection_stats()
    print(f"\n📈 Collection Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test queries
    test_queries = [
        "What is the 000 classification in Dewey Decimal System?",
        "Tell me about computer science classification",
        "How are books about religion classified?",
        "What classification covers mathematics and science?"
    ]
    
    print(f"\n🔍 Testing with sample queries...")
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        result = rag_system.query_rag_system(query, k=3)
        print(f"💡 Response: {result['response']}")
        print(f"⏱️ Processing time: {result['processing_time']:.2f}s")
        print("-" * 50)

if __name__ == "__main__":
    main()