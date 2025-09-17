#!/usr/bin/env python3
"""
Deployment and Demonstration Script for High-Performance Server
Complete pipeline for processing Dewey Classification PDF and running RAG/Fine-tuning
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeweyRAGDeployment:
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.processed_data_dir = self.base_dir / "processed_data"
        self.vector_db_dir = self.base_dir / "vector_db"
        self.fine_tuning_dir = self.base_dir / "fine_tuning_data"
        self.models_cache_dir = self.base_dir / "models_cache"
        
        # PDF file name
        self.pdf_file = "dewey-decimal-classification-ddc23-complete-1-4-23nbsped-1910608815-9781910608814_compress.pdf"
        
    def check_requirements(self) -> bool:
        """Check if all requirements are satisfied."""
        logger.info("Checking requirements...")
        
        # Check if PDF exists
        if not (self.base_dir / self.pdf_file).exists():
            logger.error(f"PDF file not found: {self.pdf_file}")
            return False
        
        # Check Python packages
        required_packages = [
            'torch', 'transformers', 'accelerate', 'datasets',
            'faiss', 'sentence_transformers', 'tqdm', 'numpy',
            'pandas', 'pdfplumber', 'PyPDF2'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing packages: {missing_packages}")
            logger.info("Please install missing packages using:")
            logger.info(f"pip install {' '.join(missing_packages)}")
            return False
        
        logger.info("All requirements satisfied!")
        return True
    
    def run_pdf_processing(self):
        """Run PDF text extraction and processing."""
        logger.info("Starting PDF processing...")
        
        try:
            # Import and run PDF processor
            sys.path.append(str(self.base_dir))
            from process_pdf_server import HighPerformancePDFProcessor
            
            processor = HighPerformancePDFProcessor(
                str(self.base_dir / self.pdf_file),
                str(self.processed_data_dir)
            )
            
            results = processor.process_pdf(chunk_size=512, chunk_overlap=100)
            logger.info("PDF processing completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            return False
    
    def run_embedding_generation(self):
        """Generate embeddings and create vector database."""
        logger.info("Starting embedding generation...")
        
        try:
            sys.path.append(str(self.base_dir))
            from rag_system_server import HighPerformanceRAGSystem
            
            rag_system = HighPerformanceRAGSystem(
                str(self.processed_data_dir),
                str(self.models_cache_dir),
                str(self.vector_db_dir)
            )
            
            # Setup embedding model and generate embeddings
            rag_system.setup_embedding_model()
            embeddings = rag_system.generate_embeddings(batch_size=64)
            rag_system.create_vector_index(embeddings)
            
            logger.info("Embedding generation completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return False
    
    def run_rag_demo(self):
        """Run RAG system demonstration."""
        logger.info("Starting RAG system demonstration...")
        
        try:
            sys.path.append(str(self.base_dir))
            from rag_system_server import HighPerformanceRAGSystem
            
            rag_system = HighPerformanceRAGSystem(
                str(self.processed_data_dir),
                str(self.models_cache_dir),
                str(self.vector_db_dir)
            )
            
            # Load existing vector index
            rag_system.load_vector_index()
            rag_system.setup_llm_model()
            
            # Example queries
            demo_queries = [
                "What is the Dewey Decimal Classification for computer science?",
                "How are books about mathematics classified?",
                "Explain the main structure of the Dewey system.",
                "What classification is used for philosophy books?",
                "How is literature organized in the Dewey system?"
            ]
            
            results = []
            for query in demo_queries:
                logger.info(f"Processing query: {query}")
                result = rag_system.query_rag_system(query, k=3)
                results.append(result)
                
                print(f"\nQuery: {query}")
                print(f"Response: {result['response']}")
                print("-" * 50)
            
            # Save demo results
            demo_file = self.base_dir / "rag_demo_results.json"
            with open(demo_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info("RAG demonstration completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"RAG demonstration failed: {e}")
            return False
    
    def create_fine_tuning_datasets(self):
        """Create datasets for fine-tuning."""
        logger.info("Creating fine-tuning datasets...")
        
        try:
            sys.path.append(str(self.base_dir))
            from create_finetuning_dataset import DeweyFineTuningDatasetCreator
            
            creator = DeweyFineTuningDatasetCreator(
                str(self.processed_data_dir),
                str(self.fine_tuning_dir)
            )
            
            datasets, stats = creator.generate_all_datasets()
            
            logger.info("Fine-tuning datasets created successfully!")
            logger.info(f"Generated {stats['combined_training_examples']} training examples")
            return True
            
        except Exception as e:
            logger.error(f"Fine-tuning dataset creation failed: {e}")
            return False
    
    def run_full_pipeline(self):
        """Run the complete processing pipeline."""
        logger.info("Starting complete Dewey RAG pipeline...")
        
        steps = [
            ("Checking requirements", self.check_requirements),
            ("Processing PDF", self.run_pdf_processing),
            ("Generating embeddings", self.run_embedding_generation),
            ("Creating fine-tuning datasets", self.create_fine_tuning_datasets),
            ("Running RAG demonstration", self.run_rag_demo),
        ]
        
        start_time = time.time()
        
        for step_name, step_func in steps:
            logger.info(f"{'='*20} {step_name} {'='*20}")
            
            step_start = time.time()
            success = step_func()
            step_time = time.time() - step_start
            
            if success:
                logger.info(f"{step_name} completed in {step_time:.2f} seconds")
            else:
                logger.error(f"{step_name} failed!")
                return False
        
        total_time = time.time() - start_time
        
        logger.info(f"\n{'='*60}")
        logger.info("DEWEY RAG PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info(f"{'='*60}")
        
        # Print summary
        self.print_summary()
        
        return True
    
    def print_summary(self):
        """Print pipeline summary."""
        print("\n" + "="*60)
        print("DEWEY DECIMAL CLASSIFICATION RAG SYSTEM")
        print("="*60)
        
        print("\nGenerated Files:")
        print("-" * 30)
        
        # Check processed data
        if self.processed_data_dir.exists():
            print(f"✓ Processed Data: {self.processed_data_dir}")
            for file in self.processed_data_dir.glob("*.json"):
                size = file.stat().st_size / 1024
                print(f"  - {file.name} ({size:.1f} KB)")
        
        # Check vector database
        if self.vector_db_dir.exists():
            print(f"✓ Vector Database: {self.vector_db_dir}")
            for file in self.vector_db_dir.glob("*"):
                size = file.stat().st_size / 1024
                print(f"  - {file.name} ({size:.1f} KB)")
        
        # Check fine-tuning data
        if self.fine_tuning_dir.exists():
            print(f"✓ Fine-tuning Data: {self.fine_tuning_dir}")
            for file in self.fine_tuning_dir.glob("*.json"):
                size = file.stat().st_size / 1024
                print(f"  - {file.name} ({size:.1f} KB)")
        
        print("\nNext Steps:")
        print("-" * 30)
        print("1. Use the RAG system for interactive queries")
        print("2. Fine-tune models using the generated datasets")
        print("3. Deploy as a web service using FastAPI or Streamlit")
        print("4. Scale with multiple GPUs for production")
        
        print("\nUsage Examples:")
        print("-" * 30)
        print("# Interactive RAG queries:")
        print("python rag_system_server.py")
        print("\n# Process new PDFs:")
        print("python process_pdf_server.py")
        print("\n# Create custom datasets:")
        print("python create_finetuning_dataset.py")

def main():
    parser = argparse.ArgumentParser(description="Dewey RAG System Deployment")
    parser.add_argument("--step", choices=[
        "check", "pdf", "embeddings", "finetune", "rag", "full"
    ], default="full", help="Which step to run")
    parser.add_argument("--base-dir", default=".", help="Base directory")
    
    args = parser.parse_args()
    
    deployment = DeweyRAGDeployment(args.base_dir)
    
    if args.step == "check":
        deployment.check_requirements()
    elif args.step == "pdf":
        deployment.run_pdf_processing()
    elif args.step == "embeddings":
        deployment.run_embedding_generation()
    elif args.step == "finetune":
        deployment.create_fine_tuning_datasets()
    elif args.step == "rag":
        deployment.run_rag_demo()
    elif args.step == "full":
        deployment.run_full_pipeline()

if __name__ == "__main__":
    main()