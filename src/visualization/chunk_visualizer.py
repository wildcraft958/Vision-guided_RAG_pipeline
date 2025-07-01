# Streamlit web app
# src/visualization/chunk_visualizer.py

"""
Streamlit-based visualization for exploring PDF chunks and chunking quality.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import our pipeline components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.pipeline.main_pipeline import PDFChunkingPipeline, create_pipeline
from src.chunkers.chunk_processor import ProcessedChunk
from config.settings import settings

def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'chunks' not in st.session_state:
        st.session_state.chunks = []
    if 'current_pdf' not in st.session_state:
        st.session_state.current_pdf = None
    if 'processing_stats' not in st.session_state:
        st.session_state.processing_stats = {}

def setup_sidebar():
    """Setup the sidebar with configuration options."""
    st.sidebar.title("⚙️ Configuration")
    
    # API Key inputs
    st.sidebar.subheader("API Keys")
    
    llama_key = st.sidebar.text_input(
        "LlamaIndex API Key",
        value=st.session_state.get('llama_key', ''),
        type="password",
        help="Get your free API key from https://cloud.llamaindex.ai"
    )
    
    openrouter_key = st.sidebar.text_input(
        "OpenRouter API Key", 
        value=st.session_state.get('openrouter_key', ''),
        type="password",
        help="Get your free API key from https://openrouter.ai"
    )
    
    # Model selection
    st.sidebar.subheader("Model Settings")
    model_options = [
        "google/gemma-3-12b-it:free",
        "google/gemma-3-4b-it:free",
        "google/gemma-2-27b-it:free"
    ]
    
    selected_model = st.sidebar.selectbox(
        "Model",
        options=model_options,
        index=0,
        help="Choose the OpenRouter model to use"
    )
    
    # Processing settings
    st.sidebar.subheader("Processing Settings")
    batch_size = st.sidebar.slider(
        "Batch Size (pages)",
        min_value=2,
        max_value=8,
        value=4,
        help="Number of pages to process in each batch"
    )
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Sampling temperature for the LLM"
    )
    
    return {
        'llama_key': llama_key,
        'openrouter_key': openrouter_key,
        'model': selected_model,
        'batch_size': batch_size,
        'temperature': temperature
    }

def create_pipeline_from_config(config):
    """Create pipeline from configuration."""
    try:
        # Update session state
        st.session_state.llama_key = config['llama_key']
        st.session_state.openrouter_key = config['openrouter_key']
        
        # Validate API keys
        if not config['llama_key']:
            st.error("LlamaParse API key is required")
            return None
            
        if not config['openrouter_key']:
            st.error("OpenRouter API key is required")
            return None
        
        # Create pipeline
        pipeline = create_pipeline(
            llama_api_key=config['llama_key'],
            openrouter_api_key=config['openrouter_key'],
            batch_size=config['batch_size'],
            model_name=config['model']
        )
        
        # Test API access
        with st.spinner("Validating API access..."):
            validation_results = pipeline.validate_api_access()
            
            if not all(validation_results.values()):
                failed_apis = [api for api, status in validation_results.items() if not status]
                st.error(f"API validation failed for: {', '.join(failed_apis)}")
                return None
        
        st.success("✅ Pipeline configured successfully!")
        return pipeline
        
    except Exception as e:
        st.error(f"Failed to create pipeline: {str(e)}")
        return None

def display_chunk_statistics(chunks: List[ProcessedChunk]):
    """Display statistics about the processed chunks."""
    if not chunks:
        st.warning("No chunks to analyze")
        return
    
    # Calculate statistics using pipeline method
    if st.session_state.pipeline:
        stats = st.session_state.pipeline.get_chunk_statistics(chunks)
    else:
        # Fallback calculation
        stats = {
            "total_chunks": len(chunks),
            "total_words": sum(chunk.metadata.get("word_count", 0) for chunk in chunks),
            "avg_words_per_chunk": sum(chunk.metadata.get("word_count", 0) for chunk in chunks) / len(chunks)
        }
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Chunks", stats["total_chunks"])
    
    with col2:
        st.metric("Total Words", f"{stats['total_words']:,}")
    
    with col3:
        st.metric("Avg Words/Chunk", f"{stats['avg_words_per_chunk']:.1f}")
    
    with col4:
        st.metric("Total Characters", f"{stats.get('total_characters', 0):,}")
    
    # Charts
    st.subheader("📊 Chunk Analysis")
    
    # Create dataframe for visualization
    chunk_data = []
    for i, chunk in enumerate(chunks):
        chunk_data.append({
            "Chunk ID": chunk.id,
            "Word Count": chunk.metadata.get("word_count", 0),
            "Character Count": chunk.metadata.get("char_count", 0),
            "Continues": chunk.continues,
            "Level 1": chunk.level_1_heading,
            "Level 2": chunk.level_2_heading,
            "Level 3": chunk.level_3_heading,
            "Content Types": ", ".join(chunk.metadata.get("content_types", []))
        })
    
    df = pd.DataFrame(chunk_data)
    
    # Word count distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = px.histogram(
            df, 
            x="Word Count",
            title="Word Count Distribution",
            nbins=20
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        fig_cont = px.pie(
            df,
            names="Continues",
            title="Continuation Distribution",
            hole=0.3
        )
        st.plotly_chart(fig_cont, use_container_width=True)
    
    # Content types analysis
    if "content_type_distribution" in stats:
        content_types_df = pd.DataFrame(
            list(stats["content_type_distribution"].items()),
            columns=["Content Type", "Count"]
        )
        
        if not content_types_df.empty:
            fig_content = px.bar(
                content_types_df,
                x="Content Type",
                y="Count",
                title="Content Type Distribution"
            )
            st.plotly_chart(fig_content, use_container_width=True)

def display_chunk_explorer(chunks: List[ProcessedChunk]):
    """Interactive chunk explorer."""
    if not chunks:
        st.warning("No chunks to explore")
        return
    
    st.subheader("🔍 Chunk Explorer")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Filter by level 1 heading
        level_1_options = ["All"] + list(set(chunk.level_1_heading for chunk in chunks if chunk.level_1_heading))
        selected_level_1 = st.selectbox("Filter by Document Section", level_1_options)
    
    with col2:
        # Filter by content type
        all_content_types = set()
        for chunk in chunks:
            all_content_types.update(chunk.metadata.get("content_types", []))
        
        content_type_options = ["All"] + list(all_content_types)
        selected_content_type = st.selectbox("Filter by Content Type", content_type_options)
    
    with col3:
        # Filter by continuation
        continuation_options = ["All", "True", "False", "Partial"]
        selected_continuation = st.selectbox("Filter by Continuation", continuation_options)
    
    # Apply filters
    filtered_chunks = chunks
    
    if selected_level_1 != "All":
        filtered_chunks = [c for c in filtered_chunks if c.level_1_heading == selected_level_1]
    
    if selected_content_type != "All":
        filtered_chunks = [c for c in filtered_chunks if selected_content_type in c.metadata.get("content_types", [])]
    
    if selected_continuation != "All":
        filtered_chunks = [c for c in filtered_chunks if c.continues == selected_continuation]
    
    st.write(f"Showing {len(filtered_chunks)} of {len(chunks)} chunks")
    
    # Chunk selector
    if filtered_chunks:
        chunk_options = [f"{chunk.id}: {chunk.heading[:60]}..." for chunk in filtered_chunks]
        selected_chunk_idx = st.selectbox("Select Chunk", range(len(chunk_options)), format_func=lambda x: chunk_options[x])
        
        # Display selected chunk
        selected_chunk = filtered_chunks[selected_chunk_idx]
        
        st.subheader(f"📄 {selected_chunk.heading}")
        
        # Metadata
        with st.expander("Chunk Metadata"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**ID:** {selected_chunk.id}")
                st.write(f"**Continues:** {selected_chunk.continues}")
                st.write(f"**Word Count:** {selected_chunk.metadata.get('word_count', 0)}")
                st.write(f"**Character Count:** {selected_chunk.metadata.get('char_count', 0)}")
            
            with col2:
                st.write(f"**Level 1:** {selected_chunk.level_1_heading}")
                st.write(f"**Level 2:** {selected_chunk.level_2_heading}")
                st.write(f"**Level 3:** {selected_chunk.level_3_heading}")
                st.write(f"**Content Types:** {', '.join(selected_chunk.metadata.get('content_types', []))}")
        
        # Content
        st.subheader("Content")
        st.text_area("Chunk Content", selected_chunk.content, height=400, disabled=True)
        
        # Raw text
        with st.expander("Raw LLM Output"):
            st.text_area("Raw Text", selected_chunk.raw_text, height=200, disabled=True)

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="PDF Chunker - Vision-Guided Document Processing",
        page_icon="📄",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.title("📄 PDF Chunker - Vision-Guided Document Processing")
    st.markdown("""
    Transform your PDFs into meaningful chunks using **LlamaParse**, **OpenRouter Gemma**, and **LangChain** 
    based on the research paper *"Vision-Guided Chunking Is All You Need"*.
    """)
    
    # Sidebar configuration
    config = setup_sidebar()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Process PDF", "📊 Analysis", "ℹ️ About"])
    
    with tab1:
        st.header("Process PDF Document")
        
        # Create or update pipeline
        if st.button("🔧 Configure Pipeline"):
            pipeline = create_pipeline_from_config(config)
            if pipeline:
                st.session_state.pipeline = pipeline
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF document to process into meaningful chunks"
        )
        
        if uploaded_file is not None and st.session_state.pipeline is not None:
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.button("🚀 Process PDF"):
                try:
                    with st.spinner("Processing PDF... This may take a few minutes."):
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("Parsing PDF with LlamaParse...")
                        progress_bar.progress(25)
                        
                        # Process the PDF
                        chunks = st.session_state.pipeline.process_pdf(temp_path)
                        
                        progress_bar.progress(100)
                        status_text.text("Processing complete!")
                        
                        # Store results
                        st.session_state.chunks = chunks
                        st.session_state.current_pdf = uploaded_file.name
                        
                        # Display summary
                        st.success(f"✅ Successfully processed {uploaded_file.name}")
                        st.info(f"Generated {len(chunks)} chunks")
                        
                        # Show preview
                        if chunks:
                            st.subheader("📋 Preview")
                            preview_chunk = chunks[0]
                            st.write(f"**First Chunk:** {preview_chunk.heading}")
                            st.text_area("Content Preview", preview_chunk.content[:500] + "...", height=150, disabled=True)
                
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        
        elif uploaded_file is not None:
            st.warning("⚠️ Please configure the pipeline first by clicking 'Configure Pipeline'")
        
        elif st.session_state.pipeline is None:
            st.info("👈 Configure your API keys in the sidebar to get started")
    
    with tab2:
        st.header("Chunk Analysis")
        
        if st.session_state.chunks:
            st.success(f"Analyzing {len(st.session_state.chunks)} chunks from {st.session_state.current_pdf}")
            
            # Statistics
            display_chunk_statistics(st.session_state.chunks)
            
            st.markdown("---")
            
            # Explorer
            display_chunk_explorer(st.session_state.chunks)
            
        else:
            st.info("No chunks available. Process a PDF in the 'Process PDF' tab first.")
    
    with tab3:
        st.header("About This Application")
        
        st.markdown("""
        ### 🎯 Vision-Guided PDF Chunking
        
        This application implements the methodology from the research paper 
        **"Vision-Guided Chunking Is All You Need: Enhancing RAG with Multimodal Document Understanding"** 
        by Tripathi et al.
        
        ### 🔧 Key Features
        
        - **LlamaParse Integration**: High-quality PDF parsing with table and figure extraction
        - **OpenRouter LLM**: Free access to Gemma 3 models for intelligent chunking
        - **Context Preservation**: Maintains semantic continuity across document sections
        - **Multimodal Understanding**: Processes both text and visual document elements
        - **Interactive Analysis**: Explore and analyze generated chunks
        
        ### 🏗️ Architecture
        
        1. **PDF Parsing**: LlamaParse extracts structured content from PDFs
        2. **Batch Processing**: Documents are processed in 4-page batches
        3. **Context-Aware Chunking**: OpenRouter Gemma generates semantically coherent chunks
        4. **Post-Processing**: Chunks are validated, merged, and structured
        5. **Visualization**: Interactive exploration of results
        
        ### 🔑 API Keys Required
        
        - **LlamaParse**: Free tier available at [cloud.llamaindex.ai](https://cloud.llamaindex.ai)
        - **OpenRouter**: Free tier available at [openrouter.ai](https://openrouter.ai)
        
        ### 📊 Research Benefits
        
        - **11% improvement** in RAG accuracy over traditional chunking
        - **5x more granular** chunks with better semantic boundaries
        - **Preserved document structure** including tables, lists, and procedures
        - **Cross-page continuity** for complex documents
        """)