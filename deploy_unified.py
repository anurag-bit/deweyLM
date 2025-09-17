#!/usr/bin/env python3
"""
Unified RAG System Launcher with Vector Database Choice
Supports both FAISS and ChromaDB backends
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Dewey RAG System with Multiple Vector DB Options")
    parser.add_argument("--vector-db", choices=["faiss", "chromadb"], default="chromadb",
                      help="Choose vector database backend (default: chromadb)")
    parser.add_argument("--step", choices=["process", "embed", "full"], default="full",
                      help="Processing step to run")
    parser.add_argument("--force-rebuild", action="store_true",
                      help="Force rebuild of vector database")
    
    args = parser.parse_args()
    
    print(f"🚀 Dewey RAG System Launcher")
    print(f"📊 Vector Database: {args.vector_db.upper()}")
    print(f"🎯 Processing Step: {args.step}")
    print("=" * 40)
    
    if args.vector_db == "chromadb":
        print("🔄 Loading ChromaDB-based RAG system...")
        try:
            from rag_system_chromadb import ChromaRAGSystem
            
            rag = ChromaRAGSystem()
            
            if args.step in ["process", "full"]:
                print("📚 Loading processed data...")
                rag.load_processed_data()
            
            if args.step in ["embed", "full"]:
                print("🔍 Setting up ChromaDB and embeddings...")
                rag.populate_chromadb(force_rebuild=args.force_rebuild)
                
                # Show stats
                stats = rag.get_collection_stats()
                print(f"\n📈 ChromaDB Stats:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
            
            if args.step == "full":
                print(f"\n✅ ChromaDB RAG system ready!")
                print(f"🔥 Next steps:")
                print(f"   streamlit run streamlit_gui.py")
                print(f"   python cli_interface.py --help")
                
        except ImportError:
            print("❌ ChromaDB not available. Install with:")
            print("   pip install chromadb sentence-transformers")
            sys.exit(1)
        except Exception as e:
            print(f"❌ ChromaDB setup failed: {e}")
            sys.exit(1)
            
    elif args.vector_db == "faiss":
        print("🔄 Loading FAISS-based RAG system...")
        try:
            from rag_system_server import HighPerformanceRAGSystem
            
            rag = HighPerformanceRAGSystem()
            
            if args.step in ["process", "full"]:
                print("📚 Loading processed data...")
                rag.load_processed_data()
            
            if args.step in ["embed", "full"]:
                print("🔍 Creating FAISS vector index...")
                if args.force_rebuild:
                    rag.create_vector_index()
                else:
                    rag.load_vector_index()
            
            if args.step == "full":
                print(f"\n✅ FAISS RAG system ready!")
                print(f"🔥 Next steps:")
                print(f"   streamlit run streamlit_gui.py")
                print(f"   python cli_interface.py --help")
                
        except ImportError:
            print("❌ FAISS not available. Install with:")
            print("   pip install faiss-cpu  # or faiss-gpu")
            sys.exit(1)
        except Exception as e:
            print(f"❌ FAISS setup failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()