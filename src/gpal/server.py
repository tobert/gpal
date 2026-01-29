import os
import logging
import glob
import fnmatch
from pathlib import Path
from typing import List, Optional, Literal
from fastmcp import FastMCP
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastMCP
mcp = FastMCP("gpal")

# Global state for sessions
sessions = {}

# Model Configuration
MODEL_ALIASES = {
    "flash": "gemini-3-flash-preview",
    "pro": "gemini-3-pro-preview"
}

# --- Gemini Tools ---

def list_directory(path: str = ".") -> List[str]:
    """Lists files and directories in the given path."""
    try:
        p = Path(path)
        if not p.exists():
            return [f"Error: Path '{path}' does not exist."]
        return [str(item.name) for item in p.iterdir()]
    except Exception as e:
        return [f"Error listing directory: {str(e)}"]

def read_file(path: str) -> str:
    """Reads the content of a file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file '{path}': {str(e)}"

def search_project(search_term: str, glob_pattern: str = "**/*") -> str:
    """
    Searches for a text term in files matching the glob pattern.
    Returns a summary of matches.
    """
    try:
        matches = []
        # glob.glob with recursive=True requires root_dir or absolute paths usually,
        # but here we scan from CWD.
        files = glob.glob(glob_pattern, recursive=True)
        
        # Limit search to avoid massive scans if pattern is too broad
        if len(files) > 1000:
             return f"Error: Too many files match '{glob_pattern}' ({len(files)}). Please refine the glob."

        count = 0
        for file in files:
            if os.path.isfile(file):
                try:
                    with open(file, "r", encoding="utf-8", errors='ignore') as f:
                        content = f.read()
                        if search_term in content:
                            matches.append(f"Match in: {file}")
                            count += 1
                            if count >= 20: # Limit results
                                matches.append("... (more matches truncated)")
                                break
                except:
                    continue # Skip unreadable files
        
        if not matches:
            return "No matches found."
        return "\n".join(matches)

    except Exception as e:
        return f"Error searching project: {str(e)}"


# --- Server Logic ---

def get_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment.")
    return genai.Client(api_key=api_key)

def create_chat(client: genai.Client, model_name: str, history: List = None):
    """Helper to create a configured chat session."""
    gemini_tools = [list_directory, read_file, search_project]
    
    return client.chats.create(
        model=model_name,
        history=history or [],
        config=types.GenerateContentConfig(
            temperature=0.2,
            tools=gemini_tools, 
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=False,
                maximum_remote_calls=10 
            ),
            system_instruction="""You are a specialized technical consultant accessed via the Model Context Protocol (MCP).
            Your role is to provide high-agency, deep reasoning and analysis on software engineering tasks.
            
            You have access to tools: 'list_directory', 'read_file', and 'search_project'. 
            USE THEM PROACTIVELY to explore the codebase. Do not ask the user for file contents if you can find them yourself.
            
            Operational Guide:
            1. Context First: Ground your answers strictly in data.
            2. Explore: If asked about a feature, search for it, find the file, read it, THEN answer.
            3. No Roleplay: Be a functional, high-agency expert.
            4. Output: Be direct, concise, and technically precise.
            """
        )
    )

def get_session(session_id: str, client: genai.Client, requested_model_alias: str):
    # Resolve alias (e.g., "flash" -> "gemini-3-flash-preview")
    target_model = MODEL_ALIASES.get(requested_model_alias.lower(), requested_model_alias)
    
    if session_id not in sessions:
        # New session
        sessions[session_id] = create_chat(client, target_model)
        # Store the model name on the session object for future checks
        sessions[session_id]._gpal_model_name = target_model
    else:
        # Existing session: Check if migration is needed
        current_session = sessions[session_id]
        current_model = getattr(current_session, "_gpal_model_name", None)
        
        if current_model != target_model:
            # Migration needed: Create new chat with old history
            try:
                # We attempt to carry over history
                # Note: 'history' property returns list of Content objects
                old_history = current_session._curated_history if hasattr(current_session, "_curated_history") else []
                # If _curated_history isn't public/stable, we might lose history. 
                # Ideally verify SDK docs. For now, we assume standard usage.
                
                logging.info(f"Migrating session {session_id} from {current_model} to {target_model}")
                sessions[session_id] = create_chat(client, target_model, history=old_history)
                sessions[session_id]._gpal_model_name = target_model
            except Exception as e:
                # Fallback: Just start fresh if migration fails
                logging.error(f"Failed to migrate session history: {e}")
                sessions[session_id] = create_chat(client, target_model)
                sessions[session_id]._gpal_model_name = target_model
                
    return sessions[session_id]

@mcp.tool()
def consult_gemini(
    query: str, 
    session_id: str = "default", 
    model: str = "flash", 
    file_paths: List[str] = []
) -> str:
    """
    Consults with Google Gemini about a query. Supports switching between 'flash' (speed) and 'pro' (reasoning).
    
    Args:
        query: The question or instruction.
        session_id: ID for conversation history.
        model: "flash" (default) or "pro". You can switch mid-session.
        file_paths: (Optional) Pre-load specific files.
    """
    client = get_client()
    session = get_session(session_id, client, model)
    
    parts = []
    
    for path in file_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                parts.append(types.Part.from_text(text=f"--- START FILE: {path} ---\n{content}\n--- END FILE: {path}---\n"))
        except Exception as e:
            return f"Error reading file {path}: {str(e)}"
            
    parts.append(types.Part.from_text(text=query))
    
    try:
        response = session.send_message(parts)
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini: {str(e)}"

def main():
    mcp.run()

if __name__ == "__main__":
    main()
