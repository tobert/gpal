import os
import sys
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gpal.server import (
    _gemini_search,
    _gemini_code_exec,
    _consult,
)

def test_gemini_search():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set")
        
    result = _gemini_search("What is the capital of France?")
    print(f"\nSearch Result:\n{result}")
    assert "Paris" in result
    assert "Error" not in result

def test_gemini_code_exec():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set")
        
    code = "print(123 + 456)"
    result = _gemini_code_exec(code)
    print(f"\nCode Execution Result:\n{result}")
    assert "579" in result
    assert "Error" not in result

def test_consult_gemini_still_works():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set")
        
    # This should use the codebase tools default
    result = _consult(
        query="List the files in the current directory.",
        session_id="test-session",
        model_alias="flash"
    )
    print(f"\nConsult Result:\n{result}")
    # It might mention the files it found
    assert len(result) > 0

if __name__ == "__main__":
    test_gemini_search()
    test_gemini_code_exec()
    test_consult_gemini_still_works()
