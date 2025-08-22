#!/usr/bin/env python3
"""
Example script showing how to use the GRU chatbot programmatically
Run: python run_examples.py
"""

import os
import sys
from inference_utils import load_artifacts, greedy_decode, beam_search_decode

def test_chatbot():
    """Test the chatbot with various inputs"""
    
    print("🤖 GRU Chatbot Examples")
    print("="*50)
    
    # Load model
    try:
        model, tokenizer, meta = load_artifacts('artifacts')
        print("✅ Model loaded successfully!")
        print(f"Vocabulary size: {meta['vocab_size']}")
        print(f"Max sequence length: {meta['max_len']}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure you've trained the model first:")
        print("  python train_gru_chatbot.py")
        return
    
    # Test prompts
    test_prompts = [
        "hi",
        "hello there",
        "how are you doing?",
        "what is artificial intelligence?",
        "tell me a joke",
        "thanks for your help",
        "goodbye"
    ]
    
    print("\n" + "="*50)
    print("GREEDY DECODING EXAMPLES")
    print("="*50)
    
    for prompt in test_prompts:
        try:
            response = greedy_decode(model, tokenizer, meta, prompt, max_new_tokens=30)
            print(f"👤 User: {prompt}")
            print(f"🤖 Bot:  {response}")
            print("-" * 40)
        except Exception as e:
            print(f"❌ Error with prompt '{prompt}': {e}")
    
    print("\n" + "="*50)
    print("BEAM SEARCH EXAMPLES")
    print("="*50)
    
    # Test beam search with a few examples
    beam_prompts = ["hello", "what is ai", "tell me a joke"]
    
    for prompt in beam_prompts:
        try:
            response = beam_search_decode(model, tokenizer, meta, prompt, 
                                        beam_width=3, max_new_tokens=30)
            print(f"👤 User: {prompt}")
            print(f"🤖 Bot:  {response}")
            print("-" * 40)
        except Exception as e:
            print(f"❌ Error with prompt '{prompt}': {e}")

def interactive_chat():
    """Interactive chat session"""
    print("\n" + "="*50)
    print("INTERACTIVE CHAT")
    print("="*50)
    print("Type 'quit' to exit")
    
    try:
        model, tokenizer, meta = load_artifacts('artifacts')
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            response = greedy_decode(model, tokenizer, meta, user_input)
            print(f"🤖 Bot: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def benchmark_performance():
    """Benchmark the model performance"""
    print("\n" + "="*50)
    print("PERFORMANCE BENCHMARK")
    print("="*50)
    
    try:
        import time
        model, tokenizer, meta = load_artifacts('artifacts')
        
        test_prompt = "hello how are you"
        num_tests = 10
        
        print(f"Running {num_tests} inference tests...")
        
        # Greedy decoding benchmark
        start_time = time.time()
        for _ in range(num_tests):
            _ = greedy_decode(model, tokenizer, meta, test_prompt)
        greedy_time = time.time() - start_time
        
        # Beam search benchmark
        start_time = time.time()
        for _ in range(num_tests):
            _ = beam_search_decode(model, tokenizer, meta, test_prompt, beam_width=3)
        beam_time = time.time() - start_time
        
        print(f"⚡ Greedy Decoding: {greedy_time/num_tests:.3f}s per response")
        print(f"⚡ Beam Search:     {beam_time/num_tests:.3f}s per response")
        print(f"📊 Beam search is {beam_time/greedy_time:.1f}x slower than greedy")
        
    except Exception as e:
        print(f"❌ Benchmark error: {e}")

def validate_model():
    """Validate model outputs"""
    print("\n" + "="*50)
    print("MODEL VALIDATION")
    print("="*50)
    
    try:
        model, tokenizer, meta = load_artifacts('artifacts')
        
        # Test edge cases
        edge_cases = [
            "",  # Empty string
            "   ",  # Whitespace only
            "a",  # Single character
            "!@#$%",  # Special characters only
            "how are you " * 10,  # Very long input
            "xyz abc def qwe",  # Nonsense words
        ]
        
        print("Testing edge cases:")
        for i, test_case in enumerate(edge_cases):
            try:
                response = greedy_decode(model, tokenizer, meta, test_case)
                status = "✅" if response and len(response) > 0 else "⚠️"
                print(f"{status} Test {i+1}: '{test_case[:20]}...' -> '{response[:30]}...'")
            except Exception as e:
                print(f"❌ Test {i+1} failed: {e}")
        
        # Test consistency
        print("\nTesting response consistency:")
        test_prompt = "hello"
        responses = []
        for i in range(5):
            response = greedy_decode(model, tokenizer, meta, test_prompt)
            responses.append(response)
        
        all_same = all(r == responses[0] for r in responses)
        print(f"{'✅' if all_same else '⚠️'} Consistency: {'All responses identical' if all_same else 'Responses vary'}")
        
        if not all_same:
            for i, resp in enumerate(responses):
                print(f"  Response {i+1}: {resp}")
    
    except Exception as e:
        print(f"❌ Validation error: {e}")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        print("Select mode:")
        print("1. Examples (e)")
        print("2. Interactive chat (i)")
        print("3. Benchmark (b)")
        print("4. Validation (v)")
        print("5. All tests (a)")
        
        choice = input("\nEnter choice (1-5 or e/i/b/v/a): ").strip().lower()
        mode_map = {
            '1': 'e', '2': 'i', '3': 'b', '4': 'v', '5': 'a',
            'e': 'e', 'i': 'i', 'b': 'b', 'v': 'v', 'a': 'a'
        }
        mode = mode_map.get(choice, 'e')
    
    if mode == 'e' or mode == 'a':
        test_chatbot()
    
    if mode == 'i':
        interactive_chat()
    elif mode == 'b' or mode == 'a':
        benchmark_performance()
    
    if mode == 'v' or mode == 'a':
        validate_model()

if __name__ == "__main__":
    main()