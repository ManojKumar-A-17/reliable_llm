"""
Reliable LLM System - Main Entry Point
Choose between interactive mode or dataset-based evaluation.
"""

import sys


def print_menu():
    """Display main menu."""
    print("\n" + "="*80)
    print("RELIABLE LLM SYSTEM".center(80))
    print("="*80)
    print("\nChoose an option:\n")
    print("  1. Interactive Q&A Mode")
    print("     Ask questions and get reliable answers with confidence metrics")
    print("     (Recommended for testing and exploration)\n")
    print("  2. Batch Evaluation Pipeline")
    print("     Process questions from dataset and save results")
    print("     (Recommended for research and evaluation)\n")
    print("  3. Exit\n")


def main():
    """Main entry point."""
    
    while True:
        print_menu()
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            print("\n" + "="*80)
            print("Starting Interactive Q&A Mode...".center(80))
            print("="*80)
            from interactive_qa import main as interactive_main
            interactive_main()
            
        elif choice == "2":
            print("\n" + "="*80)
            print("Starting Batch Evaluation Pipeline...".center(80))
            print("="*80)
            from src.pipelines.reliable_pipeline import ReliableLLMPipeline
            pipeline = ReliableLLMPipeline(confidence_threshold=0.70)
            pipeline.run()
            
        elif choice == "3":
            print("\nGoodbye! 👋\n")
            sys.exit(0)
            
        else:
            print("\n❌ Invalid choice. Please enter 1, 2, or 3.\n")


if __name__ == "__main__":
    main()
