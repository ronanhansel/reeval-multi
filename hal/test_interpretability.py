# Quick test of the interpretability setup
import os

# Check if OpenAI is installed
try:
    from openai import OpenAI
    print("✓ OpenAI library is installed")
    HAS_OPENAI = True
except ImportError:
    print("✗ OpenAI library not found. Install with: pip install openai")
    HAS_OPENAI = False

# Check for API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
if OPENAI_API_KEY:
    print(f"✓ OPENAI_API_KEY is set (length: {len(OPENAI_API_KEY)} chars)")
    print(f"  Starts with: {OPENAI_API_KEY[:7]}...")
else:
    print("✗ OPENAI_API_KEY not set")
    print("\nTo set your API key, use one of these methods:")
    print("  1. Environment variable: export OPENAI_API_KEY='sk-...'")
    print("  2. In notebook cell: OPENAI_API_KEY = 'sk-...'")

if HAS_OPENAI and OPENAI_API_KEY:
    print("\n✓ Ready for interpretability analysis!")
    print("  The notebook will use OpenAI API to interpret SAE features")
else:
    print("\n⚠️  Setup incomplete. Follow the instructions above.")
