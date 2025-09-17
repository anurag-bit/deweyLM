#!/usr/bin/env python3
"""
High-Performance PDF Processing for Server Environment
Optimized for AMD MI300X GPU with 192GB VRAM and 240GB RAM

This script efficiently processes the large Dewey Decimal Classification PDF
for RAG operations and fine-tuning preparation.
"""

import os
import re
import json
import time
import psutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import multiprocessing as mp

import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
from tqdm import tqdm
import numpy as np
import pandas as pd

class HighPerformancePDFProcessor:
    def __init__(self, pdf_path: str, output_dir: str = "processed_data"):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # System info
        self.cpu_count = mp.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        
        print(f"Initializing PDF Processor")
        print(f"PDF: {self.pdf_path}")
        print(f"CPU Cores: {self.cpu_count}")
        print(f"Available RAM: {self.memory_gb:.1f} GB")
        print(f"Output Directory: {self.output_dir}")
        
    def clean_text(self, text: str) -> str:
        """Advanced text cleaning optimized for classification documents."""
        if not text or not text.strip():
            return ""
            
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'^Page \d+.*$', '', text, flags=re.MULTILINE)
        
        # Clean up classification numbers and preserve structure
        text = re.sub(r'(\d{3}\.\d+)', r'\1', text)  # Preserve Dewey numbers
        
        # Remove excessive punctuation but preserve structure
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\"\'\/\d]', ' ', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s*([\.,:;!?])\s*', r'\1 ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_page_pymupdf(self, page_num: int, pdf_doc) -> Dict:
        """Extract text from a single page using PyMuPDF."""
        try:
            page = pdf_doc[page_num]
            text = page.get_text()
            
            # Get page metadata
            rect = page.rect
            metadata = {
                'page_number': page_num + 1,
                'width': rect.width,
                'height': rect.height,
                'text_length': len(text.strip()) if text else 0
            }
            
            cleaned_text = self.clean_text(text) if text else ""
            
            return {
                'page': page_num + 1,
                'text': cleaned_text,
                'metadata': metadata,
                'method': 'PyMuPDF'
            }
        except Exception as e:
            return {
                'page': page_num + 1,
                'text': '',
                'metadata': {'error': str(e)},
                'method': 'PyMuPDF'
            }
    
    def extract_with_pymupdf(self) -> List[Dict]:
        """Extract text using PyMuPDF (fastest method)."""
        print("Extracting text using PyMuPDF...")
        
        pdf_doc = fitz.open(str(self.pdf_path))
        total_pages = len(pdf_doc)
        
        print(f"Processing {total_pages} pages...")
        
        # Use threading for I/O bound operations
        max_workers = min(self.cpu_count, 8)  # Limit to prevent memory issues
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all pages for processing
            future_to_page = {
                executor.submit(self.extract_page_pymupdf, page_num, pdf_doc): page_num 
                for page_num in range(total_pages)
            }
            
            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_page), 
                             total=total_pages, 
                             desc="Extracting pages"):
                page_num = future_to_page[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing page {page_num + 1}: {e}")
                    results.append({
                        'page': page_num + 1,
                        'text': '',
                        'metadata': {'error': str(e)},
                        'method': 'PyMuPDF'
                    })
        
        pdf_doc.close()
        
        # Sort results by page number
        results.sort(key=lambda x: x['page'])
        
        return results
    
    def create_text_chunks(self, text: str, chunk_size: int = 1000, 
                          overlap: int = 200) -> List[Dict]:
        """Create overlapping text chunks optimized for embeddings."""
        if not text or len(text.strip()) < 50:
            return []
        
        words = text.split()
        if len(words) < chunk_size // 10:  # If text is too short
            return [{
                'chunk_id': 0,
                'text': text,
                'word_count': len(words),
                'char_count': len(text)
            }]
        
        chunks = []
        chunk_id = 0
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if len(chunk_text.strip()) > 50:  # Minimum chunk size
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                    'word_count': len(chunk_words),
                    'char_count': len(chunk_text),
                    'start_word': i,
                    'end_word': min(i + chunk_size, len(words))
                })
                chunk_id += 1
        
        return chunks
    
    def process_pdf(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Main processing pipeline."""
        start_time = time.time()
        
        print(f"Starting PDF processing...")
        print(f"File size: {self.pdf_path.stat().st_size / (1024*1024):.2f} MB")
        
        # Extract text from all pages
        pages_data = self.extract_with_pymupdf()
        
        # Filter out empty pages
        valid_pages = [p for p in pages_data if p['text'] and len(p['text'].strip()) > 50]
        print(f"Successfully processed {len(valid_pages)} pages with content")
        
        # Save page-by-page data
        pages_file = self.output_dir / "extracted_pages.json"
        with open(pages_file, 'w', encoding='utf-8') as f:
            json.dump(pages_data, f, indent=2, ensure_ascii=False)
        
        # Combine all text
        combined_text = "\n\n".join([
            f"[Page {page['page']}]\n{page['text']}" 
            for page in valid_pages
        ])
        
        # Save combined text
        text_file = self.output_dir / "dewey_classification_full_text.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(combined_text)
        
        # Create chunks for embeddings
        print("Creating text chunks for embeddings...")
        all_chunks = []
        
        for page in tqdm(valid_pages, desc="Chunking pages"):
            page_chunks = self.create_text_chunks(
                page['text'], 
                chunk_size=chunk_size, 
                overlap=chunk_overlap
            )
            
            # Add page metadata to each chunk
            for chunk in page_chunks:
                chunk['source_page'] = page['page']
                chunk['global_chunk_id'] = len(all_chunks)
                all_chunks.append(chunk)
        
        # Save chunks
        chunks_file = self.output_dir / "text_chunks.json"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
        # Create chunks DataFrame for easy analysis
        chunks_df = pd.DataFrame(all_chunks)
        chunks_csv = self.output_dir / "text_chunks.csv"
        chunks_df.to_csv(chunks_csv, index=False)
        
        # Generate statistics
        stats = self.generate_statistics(pages_data, all_chunks, combined_text)
        
        # Save statistics
        stats_file = self.output_dir / "processing_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        processing_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"PDF PROCESSING COMPLETED!")
        print(f"{'='*60}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Total pages processed: {len(valid_pages)}")
        print(f"Total characters: {len(combined_text):,}")
        print(f"Total words: {len(combined_text.split()):,}")
        print(f"Total chunks created: {len(all_chunks):,}")
        print(f"Average chunk size: {np.mean([c['word_count'] for c in all_chunks]):.1f} words")
        print(f"\nOutput files:")
        print(f"  - Full text: {text_file}")
        print(f"  - Pages data: {pages_file}")
        print(f"  - Chunks data: {chunks_file}")
        print(f"  - Chunks CSV: {chunks_csv}")
        print(f"  - Statistics: {stats_file}")
        
        return {
            'pages_data': pages_data,
            'chunks': all_chunks,
            'combined_text': combined_text,
            'statistics': stats,
            'processing_time': processing_time
        }
    
    def generate_statistics(self, pages_data, chunks, combined_text):
        """Generate comprehensive processing statistics."""
        valid_pages = [p for p in pages_data if p['text']]
        
        return {
            'processing_info': {
                'total_pages': len(pages_data),
                'valid_pages': len(valid_pages),
                'empty_pages': len(pages_data) - len(valid_pages),
                'total_chunks': len(chunks),
                'total_characters': len(combined_text),
                'total_words': len(combined_text.split()),
            },
            'chunk_statistics': {
                'avg_chunk_words': np.mean([c['word_count'] for c in chunks]),
                'avg_chunk_chars': np.mean([c['char_count'] for c in chunks]),
                'min_chunk_words': min([c['word_count'] for c in chunks]),
                'max_chunk_words': max([c['word_count'] for c in chunks]),
            },
            'page_statistics': {
                'avg_page_words': np.mean([len(p['text'].split()) for p in valid_pages]),
                'avg_page_chars': np.mean([len(p['text']) for p in valid_pages]),
                'min_page_words': min([len(p['text'].split()) for p in valid_pages]),
                'max_page_words': max([len(p['text'].split()) for p in valid_pages]),
            }
        }

def main():
    # Configuration
    pdf_file = "dewey-decimal-classification-ddc23-complete-1-4-23nbsped-1910608815-9781910608814_compress.pdf"
    output_dir = "processed_data"
    
    # Optimized chunk size for embeddings (adjust based on your embedding model)
    chunk_size = 512  # words
    chunk_overlap = 100  # words
    
    if not os.path.exists(pdf_file):
        print(f"Error: PDF file '{pdf_file}' not found!")
        return
    
    # Initialize processor
    processor = HighPerformancePDFProcessor(pdf_file, output_dir)
    
    # Process the PDF
    results = processor.process_pdf(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    print("\nReady for next steps:")
    print("1. Generate embeddings using google/embeddinggemma-300m")
    print("2. Create vector database")
    print("3. Set up RAG system with google/gemma-3-270m-it")
    print("4. Prepare fine-tuning dataset")

if __name__ == "__main__":
    main()