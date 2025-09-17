#!/usr/bin/env python3
"""
Command Line Interface for Dewey RAG System
Simple interactive CLI for testing queries without GUI
"""

import sys
import os
import time
import json
from pathlib import Path

def setup_system():
    """Initialize the RAG system."""
    print("🚀 Initializing Dewey Classification RAG System...")
    print("   This may take a few minutes on first run...")
    
    try:
        from rag_system_server import HighPerformanceRAGSystem
        
        rag_system = HighPerformanceRAGSystem()
        
        print("📚 Loading vector database...")
        rag_system.load_vector_index()
        
        print("🧠 Loading language model...")
        rag_system.setup_llm_model()
        
        print("✅ System ready!")
        return rag_system
    
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install required packages or run: python deploy_server.py --step full")
        return None
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        print("Please ensure the system has been set up by running: python deploy_server.py --step full")
        return None

def display_help():
    """Display help information."""
    help_text = """
📚 DEWEY RAG SYSTEM - COMMAND LINE INTERFACE

Commands:
  help              - Show this help message
  examples          - Show example queries
  stats             - Display system statistics
  settings          - Show/modify search parameters
  clear             - Clear the screen
  exit, quit, q     - Exit the program

Query Format:
  Just type your question and press Enter!

Examples:
  > What is the classification for computer science?
  > How are philosophy books organized?
  > What does 780.92 represent?
  > Where would I find books about artificial intelligence?

Tips:
  - Be specific in your questions for better results
  - Use library/classification terminology when possible
  - Try different phrasings if you don't get the expected answer
"""
    print(help_text)

def display_examples():
    """Display example queries."""
    examples = [
        "What is the Dewey classification for computer science?",
        "How are philosophy books organized in the Dewey system?",
        "What classification numbers are used for mathematics?",
        "Explain the structure of the 400s section.",
        "What does classification 780.92 represent?",
        "Where would I find books about artificial intelligence?",
        "How is literature classified in the Dewey system?",
        "What are the main divisions of the 000-099 range?",
        "How are religious texts organized?",
        "What classification is used for cookbooks?"
    ]
    
    print("\n💡 EXAMPLE QUERIES:")
    print("=" * 50)
    for i, example in enumerate(examples, 1):
        print(f"{i:2d}. {example}")
    print()

def display_stats():
    """Display system statistics."""
    try:
        stats_file = Path("processed_data/processing_stats.json")
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            print("\n📊 SYSTEM STATISTICS:")
            print("=" * 30)
            print(f"📄 Total Pages: {stats['processing_info']['valid_pages']:,}")
            print(f"📦 Total Chunks: {stats['processing_info']['total_chunks']:,}")
            print(f"📝 Total Words: {stats['processing_info']['total_words']:,}")
            print(f"🔤 Total Characters: {stats['processing_info']['total_characters']:,}")
            print(f"📊 Avg Chunk Size: {stats['chunk_statistics']['avg_chunk_words']:.1f} words")
            print(f"📖 Avg Page Size: {stats['page_statistics']['avg_page_words']:.1f} words")
        else:
            print("❌ Statistics not available. Run setup first.")
    except Exception as e:
        print(f"❌ Error loading statistics: {e}")

class CLISettings:
    """Manage CLI settings."""
    def __init__(self):
        self.k_chunks = 5
        self.max_response_length = 512
        self.show_sources = True
        self.show_timing = True
    
    def display(self):
        print("\n⚙️ CURRENT SETTINGS:")
        print("=" * 25)
        print(f"📦 Chunks to retrieve: {self.k_chunks}")
        print(f"📏 Max response length: {self.max_response_length}")
        print(f"📚 Show sources: {self.show_sources}")
        print(f"⏱️ Show timing: {self.show_timing}")
    
    def modify(self):
        print("\n⚙️ MODIFY SETTINGS:")
        print("Press Enter to keep current value")
        
        # Chunks
        try:
            new_k = input(f"Chunks to retrieve ({self.k_chunks}): ").strip()
            if new_k:
                self.k_chunks = max(1, min(10, int(new_k)))
        except ValueError:
            print("Invalid input, keeping current value")
        
        # Response length
        try:
            new_len = input(f"Max response length ({self.max_response_length}): ").strip()
            if new_len:
                self.max_response_length = max(100, min(1000, int(new_len)))
        except ValueError:
            print("Invalid input, keeping current value")
        
        # Show sources
        show_src = input(f"Show sources ({self.show_sources}) [y/n]: ").strip().lower()
        if show_src in ['y', 'yes', 'n', 'no']:
            self.show_sources = show_src in ['y', 'yes']
        
        # Show timing
        show_time = input(f"Show timing ({self.show_timing}) [y/n]: ").strip().lower()
        if show_time in ['y', 'yes', 'n', 'no']:
            self.show_timing = show_time in ['y', 'yes']
        
        print("✅ Settings updated!")

def format_response(result, settings):
    """Format the response for CLI display."""
    output = []
    
    # Query and response
    output.append(f"\n❓ Question: {result['query']}")
    output.append(f"\n💬 Answer:")
    output.append(f"{result['response']}")
    
    # Sources (if enabled)
    if settings.show_sources and result['retrieved_chunks']:
        output.append(f"\n📚 Sources used:")
        for i, chunk in enumerate(result['retrieved_chunks'], 1):
            score = chunk['score']
            page = chunk['source_page']
            preview = chunk['text'][:100] + "..." if len(chunk['text']) > 100 else chunk['text']
            output.append(f"  {i}. Page {page} (similarity: {score:.3f})")
            output.append(f"     {preview}")
    
    return "\n".join(output)

def main():
    """Main CLI loop."""
    print("🚀 DEWEY DECIMAL CLASSIFICATION RAG SYSTEM")
    print("=" * 50)
    
    # Initialize system
    rag_system = setup_system()
    if rag_system is None:
        return
    
    # Initialize settings
    settings = CLISettings()
    
    print("\nType 'help' for commands or start asking questions!")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            # Get user input
            query = input("🔍 > ").strip()
            
            if not query:
                continue
            
            # Handle commands
            if query.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
            
            elif query.lower() == 'help':
                display_help()
                continue
            
            elif query.lower() == 'examples':
                display_examples()
                continue
            
            elif query.lower() == 'stats':
                display_stats()
                continue
            
            elif query.lower() == 'settings':
                settings.display()
                modify = input("\nModify settings? [y/n]: ").strip().lower()
                if modify in ['y', 'yes']:
                    settings.modify()
                continue
            
            elif query.lower() == 'clear':
                os.system('clear' if os.name == 'posix' else 'cls')
                continue
            
            # Process query
            print("🔍 Searching...")
            start_time = time.time()
            
            try:
                result = rag_system.query_rag_system(
                    query, 
                    k=settings.k_chunks,
                    max_response_length=settings.max_response_length
                )
                
                search_time = time.time() - start_time
                
                # Display result
                formatted_response = format_response(result, settings)
                print(formatted_response)
                
                if settings.show_timing:
                    print(f"\n⏱️ Response time: {search_time:.2f} seconds")
                
                print("\n" + "-" * 50)
                
            except Exception as e:
                print(f"❌ Error processing query: {e}")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()