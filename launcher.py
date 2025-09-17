#!/usr/bin/env python3
"""
Dewey RAG System Launcher
Helps users choose and launch their preferred interface
"""

import os
import sys
import subprocess
from pathlib import Path

def check_system_ready():
    """Check if the RAG system is properly set up."""
    required_dirs = ["processed_data", "vector_db"]
    required_files = [
        "processed_data/text_chunks.json",
        "vector_db/faiss_index.bin",
        "vector_db/chunks_metadata.json"
    ]
    
    missing = []
    for directory in required_dirs:
        if not Path(directory).exists():
            missing.append(f"Directory: {directory}")
    
    for file in required_files:
        if not Path(file).exists():
            missing.append(f"File: {file}")
    
    return missing

def display_header():
    """Display the application header."""
    header = """
╔══════════════════════════════════════════════════════════════════════╗
║                   📚 DEWEY CLASSIFICATION RAG SYSTEM                 ║
║                                                                      ║
║         Intelligent Question-Answering for Library Classification    ║
║                     Powered by AI and Vector Search                  ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(header)

def display_menu():
    """Display the interface selection menu."""
    menu = """
🚀 Choose Your Interface:

1. 🖥️  Streamlit GUI (Web Interface)
   - User-friendly visual interface
   - Interactive charts and visualizations
   - Perfect for exploration and research
   - Runs in your web browser

2. 💻 Command Line Interface (CLI)
   - Fast, lightweight terminal interface
   - Great for quick queries
   - Scriptable and automatable
   - Minimal resource usage

3. 🐍 Python Interactive Mode
   - Direct API access
   - Full programmatic control
   - Best for developers and advanced users
   - Jupyter-friendly

4. ⚙️  System Setup & Maintenance
   - Run system diagnostics
   - Process new documents
   - Update configurations
   - View system statistics

5. 📖 Documentation & Help
   - View user guide
   - See example queries
   - Troubleshooting tips
   - API documentation

6. ❌ Exit

"""
    print(menu)

def launch_streamlit():
    """Launch the Streamlit GUI."""
    print("🚀 Launching Streamlit Web Interface...")
    print("   Your browser should open automatically to http://localhost:8501")
    print("   If not, manually open that URL in your browser")
    print("   Press Ctrl+C in this terminal to stop the server")
    print()
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_gui.py"])
    except KeyboardInterrupt:
        print("\n👋 Streamlit server stopped.")
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install: pip install streamlit")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")

def launch_cli():
    """Launch the CLI interface."""
    print("🚀 Launching Command Line Interface...")
    print("   Type 'help' for available commands")
    print("   Type 'exit' to return to this menu")
    print()
    
    try:
        subprocess.run([sys.executable, "cli_interface.py"])
    except Exception as e:
        print(f"❌ Error launching CLI: {e}")

def launch_python():
    """Launch Python interactive mode."""
    print("🚀 Launching Python Interactive Mode...")
    print("   The RAG system will be available as 'rag' variable")
    print("   Example: result = rag.query_rag_system('your question')")
    print("   Type 'exit()' to return to this menu")
    print()
    
    startup_script = """
print("Loading Dewey RAG System...")
try:
    from rag_system_server import HighPerformanceRAGSystem
    rag = HighPerformanceRAGSystem()
    rag.load_vector_index()
    print("✅ System loaded! Use: result = rag.query_rag_system('your question')")
    print("Available methods:")
    print("  - rag.query_rag_system(query, k=5)")
    print("  - rag.search_similar_chunks(query, k=5)")
    print("  - rag.setup_llm_model() # if not auto-loaded")
except Exception as e:
    print(f"❌ Error loading system: {e}")
    print("Try running: python deploy_server.py --step full")
"""
    
    try:
        subprocess.run([sys.executable, "-i", "-c", startup_script])
    except Exception as e:
        print(f"❌ Error launching Python: {e}")

def system_setup():
    """Handle system setup and maintenance."""
    setup_menu = """
⚙️ System Setup & Maintenance:

1. 🔍 Check System Status
2. 🚀 Run Full Setup (First Time)
3. 📄 Process PDF Only
4. 🧠 Generate Embeddings Only
5. 📊 View System Statistics
6. 🔧 Test RAG System
7. 🔙 Back to Main Menu

"""
    
    while True:
        print(setup_menu)
        choice = input("Select option (1-7): ").strip()
        
        if choice == '1':
            print("🔍 Checking system status...")
            missing = check_system_ready()
            if not missing:
                print("✅ System is ready!")
            else:
                print("❌ Missing components:")
                for item in missing:
                    print(f"   - {item}")
                print("\nRun option 2 to set up the system.")
        
        elif choice == '2':
            print("🚀 Running full system setup...")
            try:
                subprocess.run([sys.executable, "deploy_server.py", "--step", "full"])
            except Exception as e:
                print(f"❌ Setup failed: {e}")
        
        elif choice == '3':
            print("📄 Processing PDF...")
            try:
                subprocess.run([sys.executable, "deploy_server.py", "--step", "pdf"])
            except Exception as e:
                print(f"❌ PDF processing failed: {e}")
        
        elif choice == '4':
            print("🧠 Generating embeddings...")
            try:
                subprocess.run([sys.executable, "deploy_server.py", "--step", "embeddings"])
            except Exception as e:
                print(f"❌ Embedding generation failed: {e}")
        
        elif choice == '5':
            print("📊 System Statistics:")
            try:
                subprocess.run([sys.executable, "-c", """
import json
from pathlib import Path

stats_file = Path('processed_data/processing_stats.json')
if stats_file.exists():
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    print(f"📄 Pages: {stats['processing_info']['valid_pages']:,}")
    print(f"📦 Chunks: {stats['processing_info']['total_chunks']:,}")
    print(f"📝 Words: {stats['processing_info']['total_words']:,}")
    print(f"💾 Size: {stats['processing_info']['total_characters'] / 1024 / 1024:.1f} MB")
else:
    print("❌ No statistics available. Run setup first.")
"""])
            except Exception as e:
                print(f"❌ Error showing statistics: {e}")
        
        elif choice == '6':
            print("🔧 Testing RAG system...")
            try:
                subprocess.run([sys.executable, "-c", """
from rag_system_server import HighPerformanceRAGSystem

print("Loading system...")
rag = HighPerformanceRAGSystem()
rag.load_vector_index()

print("Testing query...")
result = rag.query_rag_system("What is the classification for computer science?", k=3)
print(f"✅ Test successful!")
print(f"Response: {result['response'][:100]}...")
"""])
            except Exception as e:
                print(f"❌ Test failed: {e}")
        
        elif choice == '7':
            break
        
        else:
            print("❌ Invalid choice. Please select 1-7.")
        
        input("\nPress Enter to continue...")

def show_documentation():
    """Show documentation and help."""
    doc_menu = """
📖 Documentation & Help:

1. 📋 Quick Start Guide
2. 💡 Example Queries
3. 🔧 Troubleshooting Tips
4. 📚 User Guide (Full)
5. 🐍 API Documentation
6. 🔙 Back to Main Menu

"""
    
    while True:
        print(doc_menu)
        choice = input("Select option (1-6): ").strip()
        
        if choice == '1':
            print("""
📋 QUICK START GUIDE:

1. First Time Setup:
   - Choose option 4 → 2 from main menu to run full setup
   - This processes the PDF and creates the knowledge base
   - Takes 5-15 minutes depending on your system

2. Using the System:
   - Streamlit GUI: Best for beginners, visual interface
   - CLI: Fast terminal interface for quick queries  
   - Python API: For developers and advanced users

3. Example Query:
   "What is the Dewey classification for computer science?"
   
4. Tips for Better Results:
   - Be specific in your questions
   - Use proper library terminology
   - Try different phrasings if needed
""")
        
        elif choice == '2':
            examples = [
                "What is the Dewey classification for computer science?",
                "How are philosophy books organized?",
                "What does classification 780.92 represent?",
                "Where would I find books about artificial intelligence?",
                "Explain the structure of the 400s section.",
                "How is literature classified in the Dewey system?",
                "What are the main divisions of science?",
                "How are cookbooks classified?",
                "What classification is used for biographies?",
                "Where are art books located in the library?"
            ]
            
            print("\n💡 EXAMPLE QUERIES:")
            print("=" * 40)
            for i, example in enumerate(examples, 1):
                print(f"{i:2d}. {example}")
        
        elif choice == '3':
            print("""
🔧 TROUBLESHOOTING TIPS:

Common Issues:

1. "System not ready" error:
   → Run full setup: Main Menu → 4 → 2

2. Slow responses:
   → First query is always slower (model loading)
   → Try reducing number of chunks retrieved

3. Poor quality answers:
   → Be more specific in your questions
   → Try different phrasings
   → Check if topic is covered in Dewey system

4. Memory errors:
   → Close other applications
   → Use CLI instead of GUI for lower memory usage

5. Import errors:
   → Install missing packages: pip install -r requirements_server.txt
   → Activate conda environment: conda activate dewey-rag-server

6. Web interface won't start:
   → Install Streamlit: pip install streamlit
   → Check port 8501 isn't in use
""")
        
        elif choice == '4':
            if Path("USER_GUIDE.md").exists():
                print("📚 Opening full user guide...")
                try:
                    if sys.platform.startswith('win'):
                        os.startfile("USER_GUIDE.md")
                    elif sys.platform.startswith('darwin'):
                        subprocess.run(["open", "USER_GUIDE.md"])
                    else:
                        subprocess.run(["xdg-open", "USER_GUIDE.md"])
                except:
                    print("❌ Could not open file automatically.")
                    print("Please manually open USER_GUIDE.md in your text editor.")
            else:
                print("❌ User guide not found.")
        
        elif choice == '5':
            print("""
🐍 API DOCUMENTATION:

Basic Usage:
```python
from rag_system_server import HighPerformanceRAGSystem

# Initialize
rag = HighPerformanceRAGSystem()
rag.load_vector_index()
rag.setup_llm_model()

# Query the system
result = rag.query_rag_system("your question", k=5)
print(result['response'])
```

Key Methods:
- query_rag_system(query, k=5, max_response_length=512)
- search_similar_chunks(query, k=5)
- setup_embedding_model()
- setup_llm_model()

Result Structure:
{
    'query': 'your question',
    'response': 'AI generated answer',
    'retrieved_chunks': [list of source chunks],
    'timestamp': processing_time
}
""")
        
        elif choice == '6':
            break
        
        else:
            print("❌ Invalid choice. Please select 1-6.")
        
        input("\nPress Enter to continue...")

def main():
    """Main application loop."""
    while True:
        display_header()
        
        # Check if system is ready
        missing = check_system_ready()
        if missing:
            print("⚠️  SYSTEM NOT READY")
            print("   Missing components detected. Please run setup first.")
            print("   Choose option 4 to access setup menu.")
            print()
        else:
            print("✅ SYSTEM READY")
            print("   All components loaded. Choose your preferred interface.")
            print()
        
        display_menu()
        
        choice = input("Select option (1-6): ").strip()
        
        if choice == '1':
            if missing:
                print("❌ System not ready. Please run setup first (option 4).")
                input("Press Enter to continue...")
                continue
            launch_streamlit()
        
        elif choice == '2':
            if missing:
                print("❌ System not ready. Please run setup first (option 4).")
                input("Press Enter to continue...")
                continue
            launch_cli()
        
        elif choice == '3':
            if missing:
                print("❌ System not ready. Please run setup first (option 4).")
                input("Press Enter to continue...")
                continue
            launch_python()
        
        elif choice == '4':
            system_setup()
        
        elif choice == '5':
            show_documentation()
        
        elif choice == '6':
            print("👋 Thank you for using the Dewey RAG System!")
            break
        
        else:
            print("❌ Invalid choice. Please select 1-6.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("Please check the system setup and try again.")