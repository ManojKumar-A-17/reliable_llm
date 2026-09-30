"""
Demo of Interactive Reliable LLM System
Tests the interactive mode with sample questions.
"""

from src.pipelines.reliable_pipeline import ReliableLLMPipeline
import time


def demo_interactive_mode():
    """Demo interactive mode with sample questions."""
    
    print("="*80)
    print("RELIABLE LLM INTERACTIVE MODE - DEMO".center(80))
    print("="*80)
    
    # Initialize pipeline
    print("\nInitializing pipeline...\n")
    pipeline = ReliableLLMPipeline(
        confidence_threshold=0.70,
        num_samples=3
    )
    
    # Sample questions to test
    sample_questions = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the chemical formula for water?"
    ]
    
    print("\n" + "="*80)
    print("DEMO: Processing 3 Sample Questions")
    print("="*80)
    
    for i, question in enumerate(sample_questions, 1):
        print(f"\n{'─'*80}")
        print(f"Question {i}/3")
        print(f"{'─'*80}\n")
        
        start = time.time()
        result = pipeline.process_question(question)
        elapsed = time.time() - start
        
        print(f"\n📝 Question: {question}")
        print(f"\n💭 Answer: {result['final_answer'][:150]}...")
        print(f"\n📊 Confidence: {result['confidence_level']} ({result['confidence_score']:.2%})")
        print(f"⏱️  Time: {elapsed:.2f}s")
        
        if result['improved']:
            print(f"✨ Status: IMPROVED (was {result['confidence_score']:.2%})")
        else:
            print(f"✅ Status: {result['confidence_level'].upper()} CONFIDENCE")
    
    print("\n" + "="*80)
    print("✓ DEMO COMPLETE - Interactive mode is working!".center(80))
    print("="*80)
    print("\nTo run interactive mode:")
    print("  python main.py  (select option 1)")
    print("  or")
    print("  python interactive_qa.py\n")


if __name__ == "__main__":
    demo_interactive_mode()
