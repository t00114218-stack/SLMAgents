import sys
import os

# Auto-inject virtual environment's site-packages to sys.path if running under system python
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "slm_orchestrator"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "slm_summarizer"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "slm_rag"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "slm_text_to_sql"))

from slm_summarizer.summarizer import SLMSummarizer
from slm_rag.rag import SLMRag
from slm_text_to_sql.text_to_sql import SLMTextToSQL

def stream_with_thought_parsing(stream):
    """
    Parses <thought>...</thought> tags in a token stream in real-time,
    displaying the thought process and the final answer separately (Claude-style).
    """
    in_thought = False
    thought_header_printed = False
    answer_header_printed = False
    buffer = ""
    
    for chunk in stream:
        buffer += chunk
        
        # Check for start tag
        if "<thought>" in buffer:
            if not thought_header_printed:
                print("\n[Thinking Process]")
                thought_header_printed = True
            in_thought = True
            parts = buffer.split("<thought>", 1)
            buffer = parts[1]
            
        # Check for end tag
        if "</thought>" in buffer:
            parts = buffer.split("</thought>", 1)
            if in_thought:
                print(parts[0], end="", flush=True)
            in_thought = False
            if not answer_header_printed:
                print("\n\n[Final Answer]")
                answer_header_printed = True
            buffer = parts[1]
            
        # Print content depending on the state
        if in_thought:
            # Print only what is safe (keeping a buffer margin to avoid leaking part of </thought> tag)
            if len(buffer) > 10:
                print(buffer[:-10], end="", flush=True)
                buffer = buffer[-10:]
        else:
            if not answer_header_printed and thought_header_printed:
                print("\n\n[Final Answer]")
                answer_header_printed = True
            if len(buffer) > 10:
                print(buffer[:-10], end="", flush=True)
                buffer = buffer[-10:]
                
    # Flush remaining buffer
    if buffer:
        cleaned_buffer = buffer.replace("<thought>", "").replace("</thought>", "")
        # If we didn't print any header at all (model didn't generate tags), default to Answer
        if not thought_header_printed and not answer_header_printed:
            print("\n[Final Answer]")
        print(cleaned_buffer, end="", flush=True)
    print()

def main():
    print("=" * 60)
    print("Claude-Style Streaming & Thought Process Verification Suite")
    print("=" * 60)
    
    # 1. Text Summarizer Streaming
    print("\n--- Testing SLMSummarizer Streaming ---")
    summarizer = SLMSummarizer(n_ctx=2048)
    text = (
        "SpaceX successfully launched its Falcon 9 rocket on Friday, sending 22 Starlink satellites "
        "into low Earth orbit. The mission lifted off from Cape Canaveral Space Force Station in Florida. "
        "About eight minutes after launch, the rocket's first stage returned to Earth, landing safely on the "
        "droneship 'A Shortfall of Gravitas' stationed in the Atlantic Ocean. This marked the 15th successful "
        "flight and landing for this particular booster, representing another milestone in SpaceX's reuse technology."
    )
    print("Requesting summary with real-time stream parsing...")
    summary_stream = summarizer.summarize(text, format="paragraph", stream=True)
    stream_with_thought_parsing(summary_stream)
    
    # 2. SLM RAG Streaming
    print("\n--- Testing SLMRag Streaming ---")
    rag = SLMRag(n_ctx=2048)
    chunks = [
        "AegisShield is NebulaCorp's flagship encryption system, released in 2025."
    ]
    print("Requesting RAG answer with real-time stream parsing...")
    rag_stream = rag.answer(
        chunks=chunks,
        question="What is the flagship product of NebulaCorp?",
        instruction="State the product and its release year.",
        stream=True
    )
    stream_with_thought_parsing(rag_stream)
    
    # 3. SLM Text-to-SQL Streaming
    print("\n--- Testing SLMTextToSQL Streaming ---")
    text_to_sql = SLMTextToSQL(n_ctx=2048)
    schema = "CREATE TABLE employees (id INT, name VARCHAR(50), salary INT);"
    print("Requesting SQL generation with real-time stream parsing...")
    sql_stream = text_to_sql.generate_sql(
        schema=schema,
        question="Get names of employees with salary > 80000",
        stream=True
    )
    stream_with_thought_parsing(sql_stream)

if __name__ == "__main__":
    main()
