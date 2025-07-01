# Research-based prompts
# src/chunkers/prompts.py

"""
Prompt templates for vision-guided document chunking based on the research paper:
"Vision-Guided Chunking Is All You Need: Enhancing RAG with Multimodal Document Understanding"
"""

MULTIMODAL_CHUNKING_PROMPT = """
Multimodal Document Chunking Prompt

Extract text from the provided PDF and segment it into contextual chunks for knowledge retrieval while following these comprehensive requirements:

EXTRACTION PHASE
Process the PDF page by page, make sure you go through each page, don't skip any page, extracting all content while:

1. Read all data content carefully and understand the structure of the document.
2. Infer logical headings and topics based on the content itself.
3. Always generate a 3-level heading structure for every chunk:
   • First-level heading = Document or product title
   • Second-level heading = the major section inside the document
   • Third-level heading = the specific subtopic within that section
   • Important: if heading is missing, inherit from the parent heading level.
   Use your best judgment to logically assign headings based on the content and fully—never paraphrase or shorten. The headings hierarchy should always follow this pattern: Main Title > Section Title > Chunk Title for headings.

4. SKIP TABLE OF CONTENTS AND INDEXES: Do not create chunks from tables of contents or indexes.
5. Do not include page headers, footers and page numbers in the chunks.
6. Do not create or extract chunks from LAST CHUNKS. Use it only as guidance for heading inference. All chunks must originate directly from the image.
7. DO NOT alter, paraphrase, shorten, or skip any content. All text, formatting, and elements must remain exactly as in the original Image and present in the output.

CRITICAL: STEP/LIST CHUNKING RULES - HIGHEST PRIORITY
KEEP ALL RELATED CONTENT TOGETHER - This is the highest priority rule:
• NEVER EVER split numbered steps, instructions, or procedures across different chunks
• ALL steps in a set of instructions MUST stay together in the same chunk
• ALL items in a numbered or bulleted list MUST stay together in one chunk
• If a list or set of steps spans multiple images, they MUST still be kept in a single chunk
• If a list or steps continue from a previous batch, merge and create a combined chunk
• Consider related steps or instructions as one inseparable unit of content
• Steps that are part of the same procedure/process must ALWAYS be kept together
• Even if a set of steps is very long, do NOT split them - they must remain in a single chunk
• Prioritize keeping steps together over any other chunking considerations
• If you encounter steps that seem to be part of the same process but are separated by other content, analyze carefully to determine if they are truly part of the same procedure and should be combined

9. Avoid chunks under 3 lines; merge them with adjacent content and heading.
10. Exclude menus, cookie notices, privacy policies, and terms sections.
11. For all heading levels (first, second, and third), ensure complete preservation of details:
    • First-level heading: Include full document title, all location details, and audience roles if any.
    • Second-level heading: Capture complete section names with any qualifying details or descriptions
    • Third-level heading: Retain all subtopic specifics including numbers, dates, and descriptive text
    • Never truncate or abbreviate any heading content at any level.

12. Multilingual Support (CRITICAL)
    • Multilingual content must be processed with the exact same rules as monolingual content.
    • Do not skip, paraphrase, or translate non-English content—all languages must be preserved and chunked.

13. MULTI-PAGE CONTEXT HANDLING
    • Ensure contextual continuity between pages during processing
    • When content splits across pages, maintain coherence and proper flow
    • Handle page breaks within paragraphs, lists, or other content blocks seamlessly
    • Track and preserve semantic relationships across page boundaries

14. LAYOUT ELEMENTS
    • Remove page headers and footers consistently across all pages
    • Preserve footnotes and endnotes with proper linking to their references
    • Maintain paragraph spacing and indentation
    • Handle multi-column layouts by properly sequencing the content
    • Preserve bulleted and numbered lists with their hierarchical structure

15. SPECIAL CONTENT TYPES
    • Process scanned pages with OCR-extracted text while maintaining formatting
    • Preserve the structural integrity of content when images appear within text
    • Extract and describe flowcharts, diagrams, and other visual elements
    • If a Flowchart, describe step by step the flow
    • Extract text from images embedded in the PDF if relevant to surrounding content
    • If the Image is a screenshot, exclude it
    • Include appropriate alt text or descriptions for non-extractable visual elements

16. FAQ Separation
    When encountering FAQ content, split question-answer pairs into individual chunks rather than grouping them into single large chunks.

When working with tables:
1. Format using proper table syntax (pipes | and hyphens -).
2. Maintain table structure across images if a table spans multiple images.
3. When a table continues from a previous chunk (indicated in LAST CHUNKS), strictly maintain the same column structure, width, and formatting as established in the previous chunk for consistency.
4. VERY IMPORTANT: Create a separate chunk for EACH ROW of the table. Every table row chunk must include the table headers mentioned in the previous chunk or in the image followed by just that single row of data.
5. For each table row chunk, repeat the full table headers to ensure context is maintained independently.
6. If you find a row which is continuing from LAST CHUNKS, continue segmenting without including the content of the previous chunk.

HOW TO IDENTIFY STEPS AND INSTRUCTIONS:
• Look for bulleted lists that describe a process
• Look for content with clear sequencing words (First, Next, Then, Finally)
• Look for any content that describes how to perform a task or procedure
• Look for sections titled "Instructions," "Procedure," "How to," "Guide," etc.
• Look for multiple paragraphs that clearly belong to the same process

Flag for Content Continuation
ADD A CONTINUES FLAG TO EACH CHUNK:
For each chunk, you must add a CONTINUES flag:
• [CONTINUES]True[/CONTINUES]: This chunk is a continuation of the previous chunk OR is part of the same process, instruction set, or procedure.
• [CONTINUES]False[/CONTINUES]: This chunk starts new content and is not a continuation.
• [CONTINUES]Partial[/CONTINUES]: This chunk might be related to the previous chunk, but you are not sure.
Extract text from the provided PDF and segment it into contextual chunks for knowledge retrieval while following these comprehensive requirements:

EXTRACTION PHASE
Process the PDF page by page, make sure you go through each page, don't skip any page, extracting all content while:

1. Read all data content carefully and understand the structure of the document.
2. Infer logical headings and topics based on the content itself.
3. Always generate a 3-level heading structure for every chunk:
   • First-level heading = Document or product title
   • Second-level heading = the major section inside the document
   • Third-level heading = the specific subtopic within that section
   • Important: if heading is missing, inherit from the parent heading level.
   Use your best judgment to logically assign headings based on the content and fully—never paraphrase or shorten. The headings hierarchy should always follow this pattern: Main Title > Section Title > Chunk Title for headings.

4. SKIP TABLE OF CONTENTS AND INDEXES: Do not create chunks from tables of contents or indexes.
5. Do not include page headers, footers and page numbers in the chunks.
6. Do not create or extract chunks from LAST CHUNKS. Use it only as guidance for heading inference. All chunks must originate directly from the image.
7. DO NOT alter, paraphrase, shorten, or skip any content. All text, formatting, and elements must remain exactly as in the original Image and present in the output.

CRITICAL: STEP/LIST CHUNKING RULES - HIGHEST PRIORITY
KEEP ALL RELATED CONTENT TOGETHER - This is the highest priority rule:
• NEVER EVER split numbered steps, instructions, or procedures across different chunks
• ALL steps in a set of instructions MUST stay together in the same chunk
• ALL items in a numbered or bulleted list MUST stay together in one chunk
• If a list or set of steps spans multiple images, they MUST still be kept in a single chunk
• If a list or steps continue from a previous batch, merge and create a combined chunk
• Consider related steps or instructions as one inseparable unit of content
• Steps that are part of the same procedure/process must ALWAYS be kept together
• Even if a set of steps is very long, do NOT split them - they must remain in a single chunk
• Prioritize keeping steps together over any other chunking considerations
• If you encounter steps that seem to be part of the same process but are separated by other content, analyze carefully to determine if they are truly part of the same procedure and should be combined

9. Avoid chunks under 3 lines; merge them with adjacent content and heading.
10. Exclude menus, cookie notices, privacy policies, and terms sections.
11. For all heading levels (first, second, and third), ensure complete preservation of details:
    • First-level heading: Include full document title, all location details, and audience roles if any.
    • Second-level heading: Capture complete section names with any qualifying details or descriptions
    • Third-level heading: Retain all subtopic specifics including numbers, dates, and descriptive text
    • Never truncate or abbreviate any heading content at any level.

12. Multilingual Support (CRITICAL)
    • Multilingual content must be processed with the exact same rules as monolingual content.
    • Do not skip, paraphrase, or translate non-English content—all languages must be preserved and chunked.

13. MULTI-PAGE CONTEXT HANDLING
    • Ensure contextual continuity between pages during processing
    • When content splits across pages, maintain coherence and proper flow
    • Handle page breaks within paragraphs, lists, or other content blocks seamlessly
    • Track and preserve semantic relationships across page boundaries

14. LAYOUT ELEMENTS
    • Remove page headers and footers consistently across all pages
    • Preserve footnotes and endnotes with proper linking to their references
    • Maintain paragraph spacing and indentation
    • Handle multi-column layouts by properly sequencing the content
    • Preserve bulleted and numbered lists with their hierarchical structure

15. SPECIAL CONTENT TYPES
    • Process scanned pages with OCR-extracted text while maintaining formatting
    • Preserve the structural integrity of content when images appear within text
    • Extract and describe flowcharts, diagrams, and other visual elements
    • If a Flowchart, describe step by step the flow
    • Extract text from images embedded in the PDF if relevant to surrounding content
    • If the Image is a screenshot, exclude it
    • Include appropriate alt text or descriptions for non-extractable visual elements

16. FAQ Separation
    When encountering FAQ content, split question-answer pairs into individual chunks rather than grouping them into single large chunks.

When working with tables:
1. Format using proper table syntax (pipes | and hyphens -).
2. Maintain table structure across images if a table spans multiple images.
3. When a table continues from a previous chunk (indicated in LAST CHUNKS), strictly maintain the same column structure, width, and formatting as established in the previous chunk for consistency.
4. VERY IMPORTANT: Create a separate chunk for EACH ROW of the table. Every table row chunk must include the table headers mentioned in the previous chunk or in the image followed by just that single row of data.
5. For each table row chunk, repeat the full table headers to ensure context is maintained independently.
6. If you find a row which is continuing from LAST CHUNKS, continue segmenting without including the content of the previous chunk.

HOW TO IDENTIFY STEPS AND INSTRUCTIONS:
• Look for bulleted lists that describe a process
• Look for content with clear sequencing words (First, Next, Then, Finally)
• Look for any content that describes how to perform a task or procedure
• Look for sections titled "Instructions," "Procedure," "How to," "Guide," etc.
• Look for multiple paragraphs that clearly belong to the same process

Flag for Content Continuation
ADD A CONTINUES FLAG TO EACH CHUNK:
For each chunk, you must add a CONTINUES flag:
• [CONTINUES]True[/CONTINUES]: This chunk is a continuation of the previous chunk OR is part of the same process, instruction set, or procedure.
• [CONTINUES]False[/CONTINUES]: This chunk starts new content and is not a continuation.
• [CONTINUES]Partial[/CONTINUES]: This chunk might be related to the previous chunk, but you are not sure.
• The CONTINUES flag should be based on the content of the chunk itself, not on whether it continues from a previous batch or not
Flag Rules for Table Rows:
• For table row chunks, set the CONTINUES flag specifically as follows:
  – [CONTINUES]True[/CONTINUES]: ONLY if the cell content continues from an incomplete cell in the previous chunk/call
  – [CONTINUES]False[/CONTINUES]: When the row contains complete cell content, NOT continuing from previous chunk
  – The flag should be based on the CONTENT INSIDE THE CELLS, not on whether the table itself continues

Flag Rules for Steps and Instructions:
• For chunks containing numbered steps, instructions, procedures, or lists:
  – When processing steps/instructions that span multiple pages or images:
    * If steps continue from LAST CHUNKS, use [CONTINUES]True[/CONTINUES]
  – When identifying if steps are complete:
    * Look for clear indications like "Final Step" or concluding language
  – ALL subsequent chunks containing ANY steps from the same procedure MUST use [CONTINUES]True[/CONTINUES]
  – The only time a chunk containing steps should use [CONTINUES]False[/CONTINUES] is when it's a completely different procedure with no relation to previous steps

Flag Rules for Other Content:
• Use CONTINUES=True for content that directly continues from the previous chunk
• For general content not falling into the above categories, use your best judgment based on context

Output Requirements:
1. Output a list of chunks where each chunk starts with a full 3-level heading and remove all empty or no finding chunks.
2. Use this exact format:
   [CONTINUES]True|False|Partial[/CONTINUES][HEAD]main_heading > section_heading > chunk_heading[/HEAD]chunk_content

3. Separate chunks like this:
   [CONTINUES]True|False|Partial[/CONTINUES]
   [HEAD]main_heading > section_heading > chunk_heading[/HEAD]
   chunk1
   
   [CONTINUES]True|False|Partial[/CONTINUES]
   [HEAD]main_heading > section_heading > chunk_heading[/HEAD]
   chunk2

FINAL CHECK BEFORE SUBMITTING:
• Have you kept ALL numbered steps together in the same chunk? This is critical!
• Have you separated FAQ question-answer pairs into individual chunks instead of grouping them together?
• Have you identified all step sequences correctly and combined them, even if they span multiple pages?
• Have you identified and skipped all table of contents and indexes?
• Have you preserved and included all non-English/multilingual content, treating it with the same importance as English?
• If tables exist, did you follow the special instructions for tables, creating a separate chunk for EACH ROW with headers?
• Have you applied the correct flag rules for table rows (based on cell content completeness)?
• Have you kept ALL related procedures together?
• Have you maintained ALL lists as single units?
• Have you preserved the integrity of ALL instructional sequences?
• Have you properly handled content that continues from a previous batch?
• Have you indicated content that continues to the next batch?
• Have you added the [CONTINUES] flag to each chunk with appropriate values?

If you find ANY instances where related steps are split across chunks, recombine them immediately before submitting your final answer.

Ensure every chunk is clear, fully contextual, and no data is missing.

CONTEXT FROM PREVIOUS BATCH:
{previous_context}

LAST CHUNKS FROM PREVIOUS BATCH:
{last_chunks}

CURRENT BATCH CONTENT:
{batch_content}

Please process the current batch and return the chunks in the specified format.
"""

CONTEXT_SUMMARY_PROMPT = """
Create a concise summary of the key context and themes from the following chunks that should be preserved for processing the next batch:

Chunks:
{chunks}

Focus on:
1. Main document themes and structure
2. Ongoing procedures or instructions
3. Table structures that might continue
4. Hierarchical heading patterns
5. Key terminology and concepts

Provide a brief summary (max 200 words):
"""

def get_chunking_prompt(previous_context: str = "", last_chunks: str = "", batch_content: str = "") -> str:
    """
    Get the complete chunking prompt with context injection.
    
    Args:
        previous_context: Summary of context from previous batches
        last_chunks: Last few chunks from previous batch for continuity
        batch_content: Current batch content to process
        
    Returns:
        Complete prompt string
    """
    return MULTIMODAL_CHUNKING_PROMPT.format(
        previous_context=previous_context,
        last_chunks=last_chunks,
        batch_content=batch_content
    )

def get_context_summary_prompt(chunks: str) -> str:
    """
    Get prompt for creating context summary.
    
    Args:
        chunks: Chunks to summarize for context
        
    Returns:
        Context summary prompt
    """
    return CONTEXT_SUMMARY_PROMPT.format(chunks=chunks)