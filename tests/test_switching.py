import sys
import os

sys.path.insert(0, os.path.abspath("src"))
from gpal.server import consult_gemini

print("Testing Model Switching...")

try:
    sid = "switch-test-1"
    
    # Step 1: Flash
    print("\n[Step 1] Asking Flash (2+2)...")
    r1 = consult_gemini.fn("What is 2+2? Only answer with the number.", session_id=sid, model="flash")
    print(f"Flash Answer: {r1}")
    
    # Step 2: Pro (Recall)
    print("\n[Step 2] Asking Pro (Recall)...")
    # asking it to recall ensures history was migrated
    r2 = consult_gemini.fn("Multiply that number by 10. Answer with the number only.", session_id=sid, model="pro")
    print(f"Pro Answer: {r2}")
    
    if "40" in r2:
        print("\nSUCCESS: Context preserved across model switch!")
    else:
        print("\nFAILURE: Context lost.")

except Exception as e:
    print(f"Error: {e}")
