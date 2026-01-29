import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.abspath("src"))

from gpal.server import consult_gemini_flash

print("Testing Agentic Capabilities (Flash).")
print("Query: 'What license does this project use? Please verify by reading the file.'")

try:
    # Note: We do NOT pass file_paths. We expect Gemini to find it.
    response = consult_gemini_flash.fn(
        "What license does this project use? You MUST list the directory to find the license file, READ the file content, and ONLY THEN answer. Do not guess. Verify by reading the file.", 
        session_id="agentic-test-v2"
    )
    print("\n--- Response from Gemini ---\n")
    print(response)
    print("\n----------------------------\n")
except Exception as e:
    print(f"Error: {e}")