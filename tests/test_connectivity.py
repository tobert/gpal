import sys
import os

sys.path.insert(0, os.path.abspath("src"))

from gpal.server import consult_gemini_flash, consult_gemini_pro

print("Testing Connectivity...")

try:
    print("Ping Flash...")
    r1 = consult_gemini_flash.fn("Ping", session_id="conn-test")
    print(f"Flash: {r1}")

    print("Ping Pro...")
    r2 = consult_gemini_pro.fn("Ping", session_id="conn-test")
    print(f"Pro: {r2}")
    
except Exception as e:
    print(f"Error: {e}")
