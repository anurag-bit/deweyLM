#!/usr/bin/env python3
"""
Streamlit GUI for Dewey Decimal Classification RAG System
Interactive web interface for querying the RAG system
"""

import streamlit as st
import json
import time
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
import logging

# Configure Streamlit page
st.set_page_config(
    page_title="Dewey Classification RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging to suppress warnings
logging.getLogger().setLevel(logging.ERROR)

@st.cache_resource
def load_rag_system():
    """Load the RAG system (cached for performance)."""
    try:
        import sys
        sys.path.append(".")
        from rag_system_server import HighPerformanceRAGSystem
        
        rag_system = HighPerformanceRAGSystem()
        
        # Load vector index
        with st.spinner("Loading vector database..."):
            rag_system.load_vector_index()
        
        # Setup embedding model
        with st.spinner("Loading embedding model..."):
            rag_system.setup_embedding_model()
        
        # Setup LLM model
        with st.spinner("Loading language model (this may take a while)..."):
            rag_system.setup_llm_model()
        
        return rag_system
    except Exception as e:
        st.error(f"Failed to load RAG system: {e}")
        st.info("Please ensure the system has been properly set up by running: python deploy_server.py --step full")
        return None

@st.cache_data
def load_statistics():
    """Load processing statistics."""
    stats_file = Path("processed_data/processing_stats.json")
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            return json.load(f)
    return None

def display_statistics():
    """Display system statistics in sidebar."""
    stats = load_statistics()
    if stats:
        st.sidebar.subheader("📊 System Statistics")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Total Pages", f"{stats['processing_info']['valid_pages']:,}")
            st.metric("Total Chunks", f"{stats['processing_info']['total_chunks']:,}")
        
        with col2:
            st.metric("Total Words", f"{stats['processing_info']['total_words']:,}")
            st.metric("Total Characters", f"{stats['processing_info']['total_characters']:,}")
        
        # Create a simple chart
        chart_data = pd.DataFrame({
            'Metric': ['Pages', 'Chunks', 'Words (K)', 'Characters (K)'],
            'Value': [
                stats['processing_info']['valid_pages'],
                stats['processing_info']['total_chunks'],
                stats['processing_info']['total_words'] // 1000,
                stats['processing_info']['total_characters'] // 1000
            ]
        })
        
        fig = px.bar(chart_data, x='Metric', y='Value', 
                     title="System Overview",
                     color='Value',
                     color_continuous_scale='viridis')
        fig.update_layout(height=300, showlegend=False)
        st.sidebar.plotly_chart(fig, use_container_width=True)

def display_example_queries():
    """Display example queries for users."""
    st.sidebar.subheader("💡 Example Queries")
    
    examples = [
        "What is the Dewey classification for computer science?",
        "How are philosophy books organized in the Dewey system?",
        "What classification numbers are used for mathematics?",
        "Explain the structure of the 400s section (Language).",
        "How is literature classified in the Dewey system?",
        "What are the main divisions of the 000-099 range?",
        "How are religious texts organized?",
        "What classification is used for art and music?",
        "Explain the 900s section of the Dewey system.",
        "How are science books categorized?"
    ]
    
    for i, example in enumerate(examples):
        if st.sidebar.button(f"📝 {example[:50]}...", key=f"example_{i}"):
            st.session_state.query_input = example
            st.rerun()

def format_search_results(results: List[Dict]) -> pd.DataFrame:
    """Format search results for display."""
    df_data = []
    for result in results:
        df_data.append({
            'Rank': result['rank'],
            'Similarity Score': f"{result['score']:.3f}",
            'Source Page': result['source_page'],
            'Word Count': result['word_count'],
            'Preview': result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
        })
    
    return pd.DataFrame(df_data)

def create_response_visualization(retrieved_chunks: List[Dict]):
    """Create visualization of retrieved chunks."""
    if not retrieved_chunks:
        return None
    
    # Create source distribution chart
    source_pages = [chunk['source_page'] for chunk in retrieved_chunks]
    source_counts = pd.Series(source_pages).value_counts().sort_index()
    
    fig = px.bar(
        x=source_counts.index,
        y=source_counts.values,
        title="Sources Used in Response",
        labels={'x': 'Source Page', 'y': 'Number of Chunks'},
        color=source_counts.values,
        color_continuous_scale='blues'
    )
    fig.update_layout(height=300, showlegend=False)
    
    return fig

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("📚 Dewey Decimal Classification RAG System")
    st.markdown("""
    Welcome to the interactive Dewey Decimal Classification RAG (Retrieval-Augmented Generation) system. 
    Ask questions about library classification, and get accurate answers backed by the official Dewey classification manual.
    """)
    
    # Load RAG system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = load_rag_system()
    
    rag_system = st.session_state.rag_system
    
    if rag_system is None:
        st.error("⚠️ RAG System not available. Please check the setup instructions.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Search parameters
        st.subheader("Search Parameters")
        k_chunks = st.slider("Number of chunks to retrieve", 1, 10, 5, 
                            help="More chunks provide more context but may include irrelevant information")
        
        max_response_length = st.slider("Maximum response length", 100, 1000, 512, 
                                      help="Longer responses provide more detail")
        
        # Display statistics
        display_statistics()
        
        # Example queries
        display_example_queries()
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Ask Your Question")
        
        # Query input
        query = st.text_area(
            "Enter your question about the Dewey Decimal Classification:",
            value=st.session_state.get('query_input', ''),
            height=100,
            placeholder="e.g., What is the classification for computer programming books?"
        )
        
        # Search button
        if st.button("🚀 Search", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner("🔍 Searching knowledge base and generating response..."):
                    start_time = time.time()
                    
                    try:
                        # Perform RAG query
                        result = rag_system.query_rag_system(
                            query, 
                            k=k_chunks, 
                            max_response_length=max_response_length
                        )
                        
                        search_time = time.time() - start_time
                        
                        # Store result in session state
                        st.session_state.last_result = result
                        st.session_state.search_time = search_time
                        
                        st.success(f"✅ Response generated in {search_time:.2f} seconds")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating response: {e}")
            else:
                st.warning("⚠️ Please enter a question!")
    
    with col2:
        st.subheader("ℹ️ System Status")
        
        # System status indicators
        st.success("🟢 Vector Database: Ready")
        st.success("🟢 Embedding Model: Ready")
        st.success("🟢 Language Model: Ready")
        
        if 'search_time' in st.session_state:
            st.info(f"⏱️ Last query took {st.session_state.search_time:.2f}s")
    
    # Display results
    if 'last_result' in st.session_state:
        result = st.session_state.last_result
        
        st.markdown("---")
        st.subheader("💬 Response")
        
        # Main response
        st.markdown(f"**Question:** {result['query']}")
        
        with st.container():
            st.markdown("### 🎯 Answer")
            st.write(result['response'])
        
        # Tabs for additional information
        tab1, tab2, tab3 = st.tabs(["📄 Source Information", "📊 Search Details", "💾 Export"])
        
        with tab1:
            st.subheader("Sources Used")
            
            # Display retrieved chunks
            retrieved_df = format_search_results(result['retrieved_chunks'])
            
            # Show source visualization
            fig = create_response_visualization(result['retrieved_chunks'])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed source table
            st.dataframe(
                retrieved_df,
                use_container_width=True,
                column_config={
                    "Preview": st.column_config.TextColumn(
                        "Content Preview",
                        width="large"
                    )
                }
            )
        
        with tab2:
            st.subheader("Search Analysis")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Chunks Retrieved", len(result['retrieved_chunks']))
            with col2:
                avg_score = sum(chunk['score'] for chunk in result['retrieved_chunks']) / len(result['retrieved_chunks'])
                st.metric("Avg. Similarity", f"{avg_score:.3f}")
            with col3:
                total_words = sum(chunk['word_count'] for chunk in result['retrieved_chunks'])
                st.metric("Context Words", total_words)
            
            # Score distribution
            scores = [chunk['score'] for chunk in result['retrieved_chunks']]
            fig = px.bar(
                x=[f"Chunk {i+1}" for i in range(len(scores))],
                y=scores,
                title="Similarity Scores by Chunk",
                labels={'x': 'Retrieved Chunks', 'y': 'Similarity Score'},
                color=scores,
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Export Results")
            
            # JSON export
            result_json = json.dumps(result, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Download as JSON",
                data=result_json,
                file_name=f"dewey_rag_result_{int(time.time())}.json",
                mime="application/json"
            )
            
            # Text export
            text_export = f"""
Query: {result['query']}

Response: {result['response']}

Sources Used:
{'-' * 50}
"""
            for i, chunk in enumerate(result['retrieved_chunks'], 1):
                text_export += f"\n{i}. Page {chunk['source_page']} (Score: {chunk['score']:.3f})\n"
                text_export += f"   {chunk['text'][:300]}...\n"
            
            st.download_button(
                label="📄 Download as Text",
                data=text_export,
                file_name=f"dewey_rag_result_{int(time.time())}.txt",
                mime="text/plain"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        Powered by Google EmbeddingGemma-300M and Gemma-3-270M-IT • 
        Built with Streamlit and FAISS • 
        Source: Dewey Decimal Classification 23rd Edition
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Initialize session state
    if 'query_input' not in st.session_state:
        st.session_state.query_input = ''
    
    main()