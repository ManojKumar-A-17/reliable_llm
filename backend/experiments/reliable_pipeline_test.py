"""
Test the complete Reliable LLM Pipeline.
"""

from src.pipelines.reliable_pipeline import ReliableLLMPipeline

if __name__ == "__main__":
    
    # Initialize pipeline
    pipeline = ReliableLLMPipeline(
        confidence_threshold=0.70,
        num_samples=3  # Use fewer samples for testing
    )
    
    # Run on a few examples
    results = pipeline.run(num_examples=2)
    
    print("\n" + "="*80)
    print("PIPELINE TEST COMPLETE")
    print("="*80)
    
    for result in results:
        print(f"\nQuestion: {result['question']}")
        print(f"Confidence: {result['confidence_level']} ({result['confidence_score']:.2%})")
        print(f"Time: {result['total_time']:.2f}s")
        print(f"Improved: {result['improved']}")
