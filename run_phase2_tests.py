"""
Phase 2 Test Runner - Interactive Menu
Helps you run Vision and Audio tests easily
"""

import sys
import os

def print_header():
    print("=" * 60)
    print("PHASE 2: CHECK-AND-UNDERSTAND FLOW")
    print("=" * 60)
    print()
    print("Test your Vision and Audio modules individually")
    print("to understand how they perform on your RTX 3050 GPU")
    print()
    print("=" * 60)
    print()

def run_vision_test():
    print("\n" + "=" * 60)
    print("Starting Vision Test...")
    print("=" * 60 + "\n")
    os.system("python test_vision.py")

def run_audio_test():
    print("\n" + "=" * 60)
    print("Starting Audio Test...")
    print("=" * 60 + "\n")
    os.system("python test_audio.py")

def run_audio_test_record():
    print("\n" + "=" * 60)
    print("Starting Audio Test with Recording...")
    print("=" * 60 + "\n")
    os.system("python test_audio.py --record")

def main():
    print_header()
    
    print("Available Tests:")
    print("  1. Vision Test (Face Mesh Detection)")
    print("  2. Audio Test (Whisper Transcription)")
    print("  3. Audio Test with Recording (Record + Transcribe)")
    print("  4. Run All Tests")
    print("  5. Exit")
    print()
    
    while True:
        try:
            choice = input("Select an option (1-5): ").strip()
            
            if choice == "1":
                run_vision_test()
                break
            elif choice == "2":
                run_audio_test()
                break
            elif choice == "3":
                run_audio_test_record()
                break
            elif choice == "4":
                print("\nRunning all tests...")
                print("\n--- Test 1: Vision ---")
                run_vision_test()
                print("\n--- Test 2: Audio ---")
                run_audio_test()
                break
            elif choice == "5":
                print("\nExiting...")
                break
            else:
                print("Invalid choice. Please enter 1-5.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            break

if __name__ == "__main__":
    main()
