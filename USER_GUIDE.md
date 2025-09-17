# 📚 Dewey Classification RAG System - User Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Using the Streamlit GUI](#using-the-streamlit-gui)
3. [Command Line Interface](#command-line-interface)
4. [Query Types and Examples](#query-types-and-examples)
5. [Understanding Results](#understanding-results)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites
Make sure the RAG system has been set up and is running properly:

```bash
# Check if the system is ready
python deploy_server.py --step check

# If not set up, run the full pipeline
python deploy_server.py --step full
```

### Launch Methods

#### Option 1: Streamlit GUI (Recommended for beginners)
```bash
# Activate the environment
conda activate dewey-rag-server

# Launch the web interface
streamlit run streamlit_gui.py

# Your browser will open to http://localhost:8501
```

#### Option 2: Python API (For developers)
```python
from rag_system_server import HighPerformanceRAGSystem

# Initialize system
rag = HighPerformanceRAGSystem()
rag.load_vector_index()
rag.setup_llm_model()

# Query the system
result = rag.query_rag_system("What is the classification for mathematics?")
print(result['response'])
```

#### Option 3: Interactive Python Session
```bash
python -i rag_system_server.py
# System loads automatically, then use interactively
```

---

## Using the Streamlit GUI

### 🖥️ Interface Overview

The Streamlit GUI provides a user-friendly way to interact with the RAG system:

#### Main Components:

1. **Query Input Area** (Left side)
   - Large text area for entering questions
   - "Search" button to submit queries
   - Real-time parameter controls

2. **Control Panel** (Right sidebar)
   - Search parameters (chunks to retrieve, response length)
   - System statistics
   - Example queries for quick testing

3. **Results Display** (Below query area)
   - Main AI response
   - Source information tabs
   - Search analysis and visualizations
   - Export options

### 🎛️ Control Panel Features

#### Search Parameters:
- **Number of chunks to retrieve (1-10)**: 
  - More chunks = more context but potentially more noise
  - Recommended: 3-5 for focused questions, 7-10 for complex topics

- **Maximum response length (100-1000 tokens)**:
  - Shorter responses = more concise answers
  - Longer responses = more detailed explanations
  - Recommended: 512 for balanced responses

#### System Statistics:
- Total pages processed from the PDF
- Number of text chunks created
- Total words and characters in the knowledge base
- Visual overview charts

#### Example Queries:
- Pre-written questions to test the system
- Click any example to auto-populate the query box
- Covers different types of classification questions

### 📊 Results Display

#### Response Tab:
- **Question**: Your original query (echoed back)
- **Answer**: AI-generated response based on retrieved information
- **Processing time**: How long the query took to process

#### Source Information Tab:
- **Source Distribution Chart**: Shows which pages contributed to the answer
- **Detailed Source Table**: 
  - Rank (relevance order)
  - Similarity score (0.0-1.0, higher = more relevant)
  - Source page number
  - Word count of the chunk
  - Preview of the content

#### Search Details Tab:
- **Summary metrics**: Number of chunks, average similarity, total context words
- **Similarity Score Chart**: Visual representation of how relevant each chunk was

#### Export Tab:
- **JSON Export**: Complete result data for programmatic use
- **Text Export**: Human-readable format for sharing or documentation

---

## Query Types and Examples

### 🔍 Basic Classification Lookups

**Format**: "What is the Dewey classification for [subject]?"

```
Examples:
• What is the Dewey classification for computer science?
• What classification number is used for poetry?
• How are philosophy books classified?
```

**Expected Response**: Specific classification numbers with explanations.

### 📖 Reverse Lookups

**Format**: "What does classification [number] represent?"

```
Examples:
• What does the classification number 004 represent?
• What subjects are in the 780s?
• Explain what 150-159 covers.
```

**Expected Response**: Description of topics covered by that range.

### 🏗️ Structural Questions

**Format**: "How is [broad topic] organized?" or "Explain the structure of [section]"

```
Examples:
• How is the 400s section organized?
• Explain the structure of the Dewey Decimal system.
• What are the main divisions of literature classification?
```

**Expected Response**: Hierarchical breakdown and organizational principles.

### 🔄 Comparative Questions

**Format**: "How are [topic A] and [topic B] classified differently?"

```
Examples:
• How are pure sciences and applied sciences classified differently?
• What's the difference between 200s and 900s classifications?
• How does the system distinguish between different types of literature?
```

**Expected Response**: Comparative analysis with specific numbers and rationales.

### 🎯 Specific Use Cases

**Format**: "Where would I find books about [specific topic]?"

```
Examples:
• Where would I find books about artificial intelligence?
• What section contains books on medieval history?
• Where are cookbooks classified?
```

**Expected Response**: Specific location guidance with classification numbers.

---

## Understanding Results

### 📈 Similarity Scores

- **0.9-1.0**: Excellent match - highly relevant to your query
- **0.7-0.9**: Good match - relevant with useful information
- **0.5-0.7**: Fair match - somewhat relevant, may contain useful context
- **Below 0.5**: Poor match - may not be directly relevant

### 📄 Source Pages

The system shows which pages from the original Dewey manual contributed to each answer:

- **Low page numbers** (1-100): Usually introductory material, principles, and overviews
- **Middle pages** (100-500): Main classification tables and detailed breakdowns
- **High page numbers** (500+): Appendices, indexes, and specialized information

### 🎨 Result Quality Indicators

#### High-Quality Results:
- Multiple sources with high similarity scores (>0.7)
- Coherent, specific answers with classification numbers
- Sources from relevant sections of the manual

#### Lower-Quality Results:
- Few sources or low similarity scores (<0.5)
- Vague or generic responses
- Sources from unrelated sections

---

## Advanced Usage

### 🔧 Customizing Search Parameters

#### For Specific Questions:
- Use **fewer chunks (3-5)** for focused, specific queries
- Use **shorter responses (200-300 tokens)** for quick answers

#### For Complex Research:
- Use **more chunks (7-10)** for comprehensive topics
- Use **longer responses (700-1000 tokens)** for detailed explanations

#### For Exploratory Queries:
- Use **moderate settings (5 chunks, 512 tokens)** as a starting point
- Adjust based on result quality

### 🐍 Python API Advanced Usage

```python
from rag_system_server import HighPerformanceRAGSystem
import json

# Initialize system
rag = HighPerformanceRAGSystem()
rag.load_vector_index()
rag.setup_llm_model()

# Batch processing multiple queries
queries = [
    "What is 004.6?",
    "How is art classified?",
    "Where are science fiction books?"
]

results = []
for query in queries:
    result = rag.query_rag_system(query, k=5)
    results.append({
        'query': query,
        'answer': result['response'],
        'confidence': sum(chunk['score'] for chunk in result['retrieved_chunks']) / len(result['retrieved_chunks'])
    })

# Save batch results
with open('batch_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### 📊 Custom Analysis

```python
# Analyze search results
def analyze_result(result):
    chunks = result['retrieved_chunks']
    
    analysis = {
        'confidence': sum(chunk['score'] for chunk in chunks) / len(chunks),
        'source_diversity': len(set(chunk['source_page'] for chunk in chunks)),
        'total_context_words': sum(chunk['word_count'] for chunk in chunks),
        'best_match_score': max(chunk['score'] for chunk in chunks)
    }
    
    return analysis

# Use with any query
result = rag.query_rag_system("Your query here")
analysis = analyze_result(result)
print(f"Confidence: {analysis['confidence']:.3f}")
print(f"Source diversity: {analysis['source_diversity']} different pages")
```

---

## Troubleshooting

### 🚨 Common Issues

#### "RAG System not available" Error
**Solution**: 
```bash
# Check if the system was properly set up
python deploy_server.py --step check

# If not, run the full setup
python deploy_server.py --step full
```

#### Slow Response Times
**Possible causes**:
- First query after startup (models loading)
- Too many chunks requested
- Complex query requiring extensive search

**Solutions**:
- Wait for initial model loading to complete
- Reduce number of chunks in search parameters
- Try simpler, more specific queries

#### Poor Quality Answers
**Possible causes**:
- Query too vague or ambiguous
- Topic not well-covered in the source material
- Search parameters not optimal

**Solutions**:
- Be more specific in your questions
- Try related or broader queries
- Increase number of retrieved chunks
- Check similarity scores in results

#### Memory Issues
**Solutions**:
```bash
# Check available memory
free -h

# If low memory, reduce batch sizes in code:
# Edit rag_system_server.py, reduce BATCH_SIZE values
```

### 🔍 Debugging Tips

#### Enable Detailed Logging:
```python
import logging
logging.basicConfig(level=logging.INFO)

# Now run your queries - you'll see detailed processing information
```

#### Check System Status:
```python
# Verify all components are loaded
print(f"Vector index loaded: {rag.vector_index is not None}")
print(f"Embedding model loaded: {rag.embedding_model is not None}")
print(f"LLM loaded: {rag.llm_model is not None}")
```

#### Test Individual Components:
```python
# Test embedding generation
test_embedding = rag.embedding_model.encode(["test query"])
print(f"Embedding shape: {test_embedding.shape}")

# Test vector search
similar_chunks = rag.search_similar_chunks("test query", k=3)
print(f"Found {len(similar_chunks)} similar chunks")

# Test LLM generation (without full RAG pipeline)
response = rag.generate_response("test query", similar_chunks[:1])
print(f"Generated response: {response}")
```

---

### 📞 Getting Help

1. **Check the logs**: Look for error messages in the terminal/console
2. **Verify setup**: Run `python deploy_server.py --step check`
3. **Try simpler queries**: Test with basic classification questions first
4. **Check resources**: Ensure adequate RAM and disk space
5. **Restart the system**: Sometimes a fresh start resolves issues

### 💡 Tips for Best Results

1. **Be specific**: "What is the classification for Java programming?" is better than "Tell me about computers"
2. **Use proper terminology**: The system understands library science terms
3. **Try variations**: If one phrasing doesn't work, try rephrasing your question
4. **Check sources**: Look at the retrieved chunks to understand why you got specific answers
5. **Experiment with parameters**: Adjust chunk count and response length based on your needs

---

## Conclusion

The Dewey Classification RAG system provides powerful search and question-answering capabilities for library classification. Whether you're a librarian, student, researcher, or just curious about how knowledge is organized, this system can help you navigate and understand the Dewey Decimal Classification system effectively.

For advanced usage, fine-tuning, or integration into other systems, refer to the main README.md and the Python API documentation.