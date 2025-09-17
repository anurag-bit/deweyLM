#!/usr/bin/env python3
"""
PDF Text Extraction Script for Dewey Decimal Classification
Extracts text from the PDF file for tokenization and RAG operations.
"""

import PyPDF2
import pdfplumber
import os
import json
from tqdm import tqdm

def extract_text_pypdf2(pdf_path):
    """Extract text using PyPDF2 (faster but less accurate for complex layouts)"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            print(f"Total pages: {len(pdf_reader.pages)}")
            
            for page_num in tqdm(range(len(pdf_reader.pages)), desc="Extracting with PyPDF2"):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n"
                    text += page_text
                    
    except Exception as e:
        print(f"Error with PyPDF2: {e}")
        return None
    
    return text

def extract_text_pdfplumber(pdf_path):
    """Extract text using pdfplumber (more accurate but slower)"""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"Total pages: {len(pdf.pages)}")
            
            for page_num in tqdm(range(len(pdf.pages)), desc="Extracting with pdfplumber"):
                page = pdf.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n"
                    text += page_text
                    
    except Exception as e:
        print(f"Error with pdfplumber: {e}")
        return None
    
    return text

def clean_text(text):
    """Clean and normalize the extracted text"""
    if not text:
        return ""
    
    # Remove excessive whitespace and normalize
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line:  # Skip empty lines
            cleaned_lines.append(line)
    
    # Join lines back together
    cleaned_text = '\n'.join(cleaned_lines)
    
    # Replace multiple consecutive newlines with double newlines
    import re
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    return cleaned_text

def save_text_to_file(text, output_path):
    """Save extracted text to a file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"Text saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving text: {e}")
        return False

def get_text_stats(text):
    """Get basic statistics about the extracted text"""
    if not text:
        return {}
    
    stats = {
        'total_characters': len(text),
        'total_words': len(text.split()),
        'total_lines': len(text.split('\n')),
        'unique_words': len(set(text.lower().split()))
    }
    
    return stats

def main():
    # PDF file path
    pdf_path = "dewey-decimal-classification-ddc23-complete-1-4-23nbsped-1910608815-9781910608814_compress.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return
    
    print(f"Processing PDF: {pdf_path}")
    print(f"File size: {os.path.getsize(pdf_path) / (1024*1024):.2f} MB")
    
    # Try pdfplumber first (more accurate)
    print("\n=== Attempting extraction with pdfplumber ===")
    text = extract_text_pdfplumber(pdf_path)
    
    if not text or len(text.strip()) < 1000:
        print("\npdfplumber extraction failed or returned minimal text. Trying PyPDF2...")
        text = extract_text_pypdf2(pdf_path)
    
    if not text:
        print("Both extraction methods failed!")
        return
    
    # Clean the extracted text
    print("\nCleaning extracted text...")
    cleaned_text = clean_text(text)
    
    # Get text statistics
    stats = get_text_stats(cleaned_text)
    print(f"\n=== Text Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value:,}")
    
    # Save the extracted text
    output_file = "dewey_decimal_extracted_text.txt"
    if save_text_to_file(cleaned_text, output_file):
        print(f"\nExtraction complete! Text saved to: {output_file}")
        
        # Save statistics as JSON
        stats_file = "extraction_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to: {stats_file}")
        
        # Show a preview of the extracted text
        print(f"\n=== Text Preview (first 500 characters) ===")
        print(cleaned_text[:500] + "...")
    else:
        print("Failed to save extracted text!")

if __name__ == "__main__":
    main()