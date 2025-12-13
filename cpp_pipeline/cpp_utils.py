"""
Core utilities for C++ code compilation, execution, and reward computation.
Shared across pipeline stages.
"""

import subprocess
import tempfile
import os
import re
from typing import List, Dict, Tuple, Optional
from datetime import datetime


def compile_cpp_code(code: str, compiler: str = "g++") -> Tuple[bool, str, List[Dict]]:
    """
    Compile C++ code and extract error information.
    
    Args:
        code: C++ source code as string
        compiler: Compiler to use (default: g++)
    
    Returns:
        Tuple of:
        - success: bool - whether compilation succeeded
        - stderr: str - compiler error output
        - errors: List[Dict] - parsed error information with line numbers
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
        f.write(code)
        temp_file = f.name
    
    try:
        # Compile with maximum error reporting (C++ mode) and Sanitizers for runtime checks
        # -fsanitize=address: catch memory errors (segfaults, buffer overflows)
        # -fsanitize=undefined: catch undefined behavior (null deref, div by zero)
        # -g: include debug symbols for line numbers in stack traces
        result = subprocess.run(
            [
                compiler, 
                "-std=c++17", 
                "-c", 
                "-g",
                "-fsanitize=address",
                "-fsanitize=undefined",
                "-Wall", "-Wextra", "-Werror", 
                temp_file, 
                "-o", temp_file + ".o"
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        stderr = result.stderr
        success = result.returncode == 0
        
        # Parse errors
        errors = parse_compiler_errors(stderr, temp_file)
        
        return success, stderr, errors
    
    except subprocess.TimeoutExpired:
        return False, "Compilation timeout", []
    except Exception as e:
        return False, f"Compilation error: {str(e)}", []
    finally:
        # Cleanup
        try:
            os.unlink(temp_file)
            if os.path.exists(temp_file + ".o"):
                os.unlink(temp_file + ".o")
        except:
            pass


def link_executable(obj_file_path: str, output_path: str, compiler: str = "g++") -> bool:
    """
    Link object file into an executable with sanitizer libraries.
    """
    try:
        result = subprocess.run(
            [
                compiler,
                "-g",
                "-fsanitize=address",
                "-fsanitize=undefined",
                obj_file_path,
                "-o", output_path
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def run_cpp_executable(executable_path: str, timeout: float = 5.0) -> Tuple[bool, str, str, List[Dict]]:
    """
    Run the compiled executable and capture runtime errors.
    
    Returns:
        Tuple of:
        - success: bool - True if exit code 0, False otherwise
        - stdout: str
        - stderr: str (contains sanitizer output)
        - runtime_errors: List[Dict] with parsed error locations
    """
    try:
        # Set ASAN_OPTIONS to ensure we get symbolic stack traces if possible
        # (Though -g usually handles it)
        env = os.environ.copy()
        env["ASAN_OPTIONS"] = "symbolize=1:halt_on_error=1"
        env["UBSAN_OPTIONS"] = "print_stacktrace=1:halt_on_error=1"

        result = subprocess.run(
            [executable_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )
        
        success = result.returncode == 0
        stderr = result.stderr
        
        runtime_errors = []
        if not success:
            runtime_errors = parse_runtime_errors(stderr)
            
        return success, result.stdout, stderr, runtime_errors

    except subprocess.TimeoutExpired:
        return False, "", "Runtime timeout", [{"line": None, "message": "Runtime timeout"}]
    except Exception as e:
        return False, "", f"Execution error: {str(e)}", [{"line": None, "message": str(e)}]


def parse_compiler_errors(stderr: str, filename: str) -> List[Dict]:
    """
    Parse compiler error output to extract line numbers and error messages.
    """
    errors = []
    
    # Pattern for GCC/Clang errors: filename:line:column: type: message
    pattern = r'(?:^|\n)(?:.*/)?([^:]+):(\d+)(?::(\d+))?:\s*(error|warning|note):\s*(.+?)(?=\n|$)'
    
    for match in re.finditer(pattern, stderr, re.MULTILINE):
        file_match = match.group(1)
        line = int(match.group(2))
        column = int(match.group(3)) if match.group(3) else None
        error_type = match.group(4)
        message = match.group(5).strip()
        
        # Only count errors (not warnings) for now, or if it's our file
        # Note: temp files might have random names, so we trust the parser found a file
        errors.append({
            "line": line,
            "column": column,
            "type": error_type,
            "message": message,
        })
    
    return errors


def parse_runtime_errors(stderr: str) -> List[Dict]:
    """
    Parse ASan/UBSan output to find the crashing line in our code.
    
    Looking for patterns like:
    - "#0 0x... in main /path/to/file.cpp:10"
    - "runtime error: ... at /path/to/file.cpp:10:5"
    """
    errors = []
    
    # Pattern for ASan stack traces (looking for our source file)
    # We look for lines ending with .cpp:LINE or .cpp:LINE:COL
    # We want the FIRST occurrence usually, as that's the deepest call in our code
    
    # UBSan pattern: file.cpp:10:5: runtime error: ...
    ubsan_pattern = r'([^:\n]+\.cpp):(\d+)(?::(\d+))?: runtime error: (.+)'
    
    # ASan/Stack trace pattern: #N 0x... in func file.cpp:LINE
    asan_pattern = r'#\d+\s+0x[0-9a-f]+\s+in\s+.*\s+([^:\s]+\.cpp):(\d+)'

    # Check UBSan first
    for match in re.finditer(ubsan_pattern, stderr):
        errors.append({
            "line": int(match.group(2)),
            "column": int(match.group(3)) if match.group(3) else None,
            "type": "runtime_error",
            "message": match.group(4).strip()
        })
        # Usually just one runtime error triggers termination
        break
        
    if not errors:
        # Check ASan stack trace
        # We want to find the top-most frame that corresponds to our source code
        for match in re.finditer(asan_pattern, stderr):
            errors.append({
                "line": int(match.group(2)),
                "type": "runtime_error",
                "message": "Crash/Sanitizer error (see stack trace)"
            })
            break # Take top-most frame in our code

    return errors


def create_token_rewards_from_compiler_errors(
    code: str,
    errors: List[Dict],
    tokenizer,
    error_reward: float = 0.0,
    runtime_error_reward: float = 0.2,
    warning_reward: float = 0.5,
    clean_reward: float = 1.0,
) -> List[float]:
    """
    Create token-level reward vector based on compiler/runtime errors.
    
    Args:
        code: Source code
        errors: List of error dicts
        tokenizer: Tokenizer
        error_reward: Reward for compile errors (default 0.0)
        runtime_error_reward: Reward for runtime errors (default 0.2)
        warning_reward: Reward for warnings (default 0.5)
        clean_reward: Reward for clean lines (default 1.0)
    """
    # Try to use offset mapping if available (more accurate)
    try:
        encoding = tokenizer(code, add_special_tokens=False, return_offsets_mapping=True)
        token_offsets = encoding['offset_mapping']
        tokens = encoding['input_ids']
    except:
        # Fallback: tokenize and decode
        tokens = tokenizer.encode(code, add_special_tokens=False)
        decoded_tokens = [tokenizer.decode([t]) for t in tokens]
        token_offsets = []
        char_pos = 0
        for token_str in decoded_tokens:
            start = char_pos
            char_pos += len(token_str)
            token_offsets.append((start, char_pos))
    
    # Create line -> reward mapping
    lines = code.split('\n')
    line_rewards = {}
    for i in range(len(lines)):
        line_rewards[i + 1] = clean_reward  # 1-indexed line numbers
    
    # Apply error penalties
    for error in errors:
        line_num = error["line"]
        if line_num is None: 
            continue # Skip unknown lines
            
        if error["type"] == "error":
            line_rewards[line_num] = error_reward
        elif error["type"] == "runtime_error":
            line_rewards[line_num] = runtime_error_reward
        elif error["type"] == "warning":
            line_rewards[line_num] = min(line_rewards.get(line_num, clean_reward), warning_reward)
    
    # Map character positions to line numbers
    line_start_positions = [0]
    cumulative_pos = 0
    for line in lines:
        line_start_positions.append(cumulative_pos)
        cumulative_pos += len(line) + 1 
    
    def char_pos_to_line(char_pos: int) -> int:
        if char_pos >= len(code):
            return len(lines)
        for i in range(len(line_start_positions) - 1):
            if line_start_positions[i] <= char_pos < line_start_positions[i + 1]:
                return i + 1
        return len(lines)
    
    # Map tokens to lines using offsets
    token_rewards = []
    for idx, (start_char, end_char) in enumerate(token_offsets):
        line_num = char_pos_to_line(start_char)
        reward = line_rewards.get(line_num, clean_reward)
        token_rewards.append(reward)
    
    return token_rewards
