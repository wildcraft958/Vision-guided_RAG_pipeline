# Command-line demo
# examples/run_example.py

"""
Example script demonstrating the PDF chunking pipeline.
"""

import os
import sys
import logging
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline.main_pipeline import create_pipeline
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main example function."""
    print("🚀 PDF Chunking Pipeline Example")
    print("=" * 50)
    
    # Check if API keys are configured
    try:
        settings.validate()
        print("✅ Configuration validation passed")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("\nPlease set the following environment variables:")
        print("- LLAMAINDEX_API_KEY: Get from https://cloud.llamaindex.ai")
        print("- OPENROUTER_API_KEY: Get from https://openrouter.ai")
        print("\nOr create a .env file with these keys.")
        return
    
    # Create pipeline
    print("\n🔧 Initializing pipeline...")
    try:
        pipeline = create_pipeline()
        print("✅ Pipeline initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return
    
    # Test API access
    print("\n🔍 Testing API access...")
    try:
        validation_results = pipeline.validate_api_access()
        
        for api, status in validation_results.items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {api.upper()}: {'Connected' if status else 'Failed'}")
        
        if not all(validation_results.values()):
            print("⚠️ Some APIs failed validation. Check your API keys.")
            return
            
    except Exception as e:
        print(f"❌ API validation error: {e}")
        return
    
    # Look for example PDF files
    sample_dir = Path(__file__).parent / "sample_pdfs"
    pdf_files = list(sample_dir.glob("*.pdf")) if sample_dir.exists() else []
    
    if not pdf_files:
        print(f"\n📁 No sample PDFs found in {sample_dir}")
        print("Please add a PDF file to the sample_pdfs directory or provide a file path.")
        
        # Ask user for PDF path
        pdf_path = input("\nEnter path to PDF file (or press Enter to skip): ").strip()
        if not pdf_path:
            print("No PDF provided. Creating a sample demonstration...")
            demonstrate_pipeline_features(pipeline)
            return
        
        if not os.path.exists(pdf_path):
            print(f"❌ File not found: {pdf_path}")
            return
            
        pdf_files = [Path(pdf_path)]
    
    # Process the first PDF
    pdf_file = pdf_files[0]
    print(f"\n📄 Processing PDF: {pdf_file.name}")
    
    try:
        # Process the PDF
        chunks = pipeline.process_pdf(str(pdf_file))
        
        print(f"\n✅ Processing completed!")
        print(f"📊 Generated {len(chunks)} chunks")
        
        # Display statistics
        display_results(pipeline, chunks)
        
        # Save results
        output_file = f"results_{pdf_file.stem}.json"
        pipeline._save_results(chunks, output_file, {"source_file": str(pdf_file)})
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        logging.exception("Processing error")

def display_results(pipeline, chunks):
    """Display processing results and statistics."""
    if not chunks:
        print("No chunks generated")
        return
    
    print("\n📈 Processing Statistics")
    print("-" * 30)
    
    # Get statistics
    stats = pipeline.get_chunk_statistics(chunks)
    
    print(f"Total Chunks: {stats['total_chunks']}")
    print(f"Total Words: {stats['total_words']:,}")
    print(f"Average Words per Chunk: {stats['avg_words_per_chunk']:.1f}")
    print(f"Total Characters: {stats.get('total_characters', 0):,}")
    
    # Content type distribution
    if 'content_type_distribution' in stats:
        print(f"\nContent Types:")
        for content_type, count in stats['content_type_distribution'].items():
            print(f"  {content_type}: {count}")
    
    # Continuation distribution
    if 'continuation_distribution' in stats:
        print(f"\nContinuation Flags:")
        for flag, count in stats['continuation_distribution'].items():
            print(f"  {flag}: {count}")
    
    # Sample chunks
    print(f"\n📋 Sample Chunks")
    print("-" * 30)
    
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"\nChunk {i+1}: {chunk.id}")
        print(f"Heading: {chunk.heading}")
        print(f"Continues: {chunk.continues}")
        print(f"Content Types: {chunk.metadata.get('content_types', [])}")
        print(f"Word Count: {chunk.metadata.get('word_count', 0)}")
        
        # Show first 200 characters of content
        content_preview = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
        print(f"Content Preview:\n{content_preview}")
        print("-" * 50)

def demonstrate_pipeline_features(pipeline):
    """Demonstrate pipeline features without processing a real PDF."""
    print("\n🎯 Pipeline Feature Demonstration")
    print("-" * 40)
    
    print("✅ LlamaParse integration configured")
    print("✅ OpenRouter LLM wrapper ready")
    print("✅ Context manager initialized")
    print("✅ Chunk processor configured")
    print("✅ Vision-guided prompts loaded")
    
    print(f"\nPipeline Configuration:")
    print(f"- Batch Size: {pipeline.batch_size} pages")
    print(f"- Model: {pipeline.llm.model_name}")
    print(f"- Temperature: {pipeline.llm.temperature}")
    
    print(f"\nReady to process PDFs with:")
    print("- Multimodal document understanding")
    print("- Context preservation across pages")
    print("- Intelligent chunk boundaries")
    print("- Table and figure extraction")
    print("- Hierarchical heading structure")

def create_sample_pdf_directory():
    """Create sample PDF directory with instructions."""
    sample_dir = Path(__file__).parent / "sample_pdfs"
    sample_dir.mkdir(exist_ok=True)
    
    readme_content = """# Sample PDFs

Place your PDF files in this directory to test the chunking pipeline.

## Recommended Document Types

The vision-guided chunking approach works best with:

- **Technical manuals** with procedures and tables
- **Research papers** with figures and references  
- **Business reports** with mixed content types
- **Regulatory documents** with structured sections
- **Educational materials** with step-by-step content

## Example Documents

You can test with various document types:

1. **Simple text documents** - to see basic chunking
2. **Documents with tables** - to test table preservation
3. **Multi-page procedures** - to test step continuity
4. **Mixed content documents** - to test content type detection

## Results

After processing, you'll get:
- Structured chunks with hierarchical headings
- Preserved table and list structures
- Context-aware chunk boundaries
- Metadata about content types
- Processing statistics and analysis
"""
    
    readme_path = sample_dir / "README.md"
    if not readme_path.exists():
        with open(readme_path, 'w') as f:
            f.write(readme_content)

if __name__ == "__main__":
    # Create sample directory
    create_sample_pdf_directory()
    
    # Run main example
    main()
    
    print("\n🎉 Example completed!")
    print("\nTo run the Streamlit visualization:")
    print("streamlit run src/visualization/chunk_visualizer.py")