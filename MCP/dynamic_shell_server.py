from mcp.server.fastmcp import FastMCP, Context
import asyncio
import os
import time
import uuid
from typing import List, Optional, Dict, Union, Any
from datetime import datetime
import threading
import json
from pathlib import Path

# Initialize the MCP server
mcp = FastMCP("Shell Commander")

# Process tracking
PROCESSES = {}
LOGS_DIR = Path.home() / ".config" / "mcp-shell-server" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

class ProcessInfo:
    def __init__(self, command: str, process_id: str):
        self.command = command
        self.process_id = process_id
        self.start_time = datetime.now()
        self.status = "running"
        self.result = None
        self.error = None
        self.log_file = LOGS_DIR / f"{process_id}.log"
        
    def to_dict(self):
        return {
            "process_id": self.process_id,
            "command": self.command,
            "start_time": self.start_time.isoformat(),
            "status": self.status,
            "runtime": str(datetime.now() - self.start_time),
            "log_file": str(self.log_file)
        }

async def run_process_in_background(cmd: str, is_shell: bool, process_info: ProcessInfo):
    """Run a process in the background and update its status"""
    try:
        with open(process_info.log_file, "w") as log_file:
            log_file.write(f"Command: {cmd}\n")
            log_file.write(f"Started: {process_info.start_time.isoformat()}\n")
            log_file.write(f"Process ID: {process_info.process_id}\n\n")
            log_file.write("=== OUTPUT ===\n")
            
            if is_shell:
                process = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
            else:
                cmd_parts = cmd.split()
                process = await asyncio.create_subprocess_exec(
                    cmd_parts[0],
                    *cmd_parts[1:],
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
            
            # Read output
            stdout, stderr = await process.communicate()
            
            # Get result
            stdout_str = stdout.decode('utf-8') if stdout else ""
            stderr_str = stderr.decode('utf-8') if stderr else ""
            
            # Write to log
            log_file.write(f"\n=== STDOUT ===\n{stdout_str}\n")
            log_file.write(f"\n=== STDERR ===\n{stderr_str}\n")
            
            # Update process info
            process_info.status = "completed" if process.returncode == 0 else "failed"
            process_info.result = stdout_str
            process_info.error = stderr_str if process.returncode != 0 else None
            
            log_file.write(f"\n=== RESULT ===\nStatus: {process_info.status}\n")
            log_file.write(f"Return code: {process.returncode}\n")
            log_file.write(f"End time: {datetime.now().isoformat()}\n")
    
    except Exception as e:
        # Update process info on error
        process_info.status = "error"
        process_info.error = str(e)
        
        # Write to log
        with open(process_info.log_file, "a") as log_file:
            log_file.write(f"\n=== ERROR ===\n{str(e)}\n")

@mcp.tool()
async def execute_command(command: str, args: Optional[List[str]] = None, shell: bool = True) -> str:
    """
    Execute a shell command asynchronously and return a process ID.
    
    Args:
        command: The command to execute
        args: Optional list of command arguments
        shell: Whether to use shell execution (default: True)
    
    Returns:
        Process ID for tracking the command. IMPORTANT: The process will continue running in the background.
        Use get_process_status(process_id) to check if it has completed, and get_process_output(process_id)
        to view the results when finished.
    """
    try:
        # Handle both string and list commands
        if args:
            if shell:
                cmd = f"{command} {' '.join(args)}"
            else:
                cmd = [command] + args if shell else command
        else:
            cmd = command

        # Generate a unique process ID
        process_id = str(uuid.uuid4())
        
        # Create process info
        command_str = cmd if isinstance(cmd, str) else " ".join(cmd)
        process_info = ProcessInfo(command_str, process_id)
        PROCESSES[process_id] = process_info
        
        # Start process in background
        if isinstance(cmd, list):
            cmd_str = " ".join(cmd)
        else:
            cmd_str = cmd
            
        asyncio.create_task(run_process_in_background(cmd_str, shell, process_info))
        
        return f"Process started with ID: {process_id}\n\nIMPORTANT: This is a background process that will continue running after this response.\nNo further updates will be provided automatically.\n\nWhen you want to check if it's complete, ask to run:\nget_process_status({process_id})\n\nTo see output when finished, ask to run:\nget_process_output({process_id})"
                
    except Exception as e:
        return f"Unexpected Error: {str(e)}"

@mcp.tool()
async def run_in_venv(venv_path: str, command: str) -> str:
    """
    Run a command in a specific virtual environment.
    
    Args:
        venv_path: Path to the virtual environment
        command: Command to execute in the venv
    
    Returns:
        Process ID for tracking the command. IMPORTANT: The process will continue running in the background.
        Use get_process_status(process_id) to check if it has completed, and get_process_output(process_id)
        to view the results when finished.
    """
    # Construct the command to activate venv and run command
    activation_command = f"source {os.path.join(venv_path, 'bin', 'activate')} && {command}"
    
    return await execute_command("/bin/bash", ["-c", activation_command])

@mcp.tool()
async def get_process_status(process_id: str) -> str:
    """
    Get the status of a running process.
    
    Args:
        process_id: The ID of the process to check
    """
    if process_id not in PROCESSES:
        return f"Process with ID {process_id} not found"
    
    process_info = PROCESSES[process_id]
    status_dict = process_info.to_dict()
    
    # Format the response
    response_text = f"Process ID: {status_dict['process_id']}\n"
    response_text += f"Command: {status_dict['command']}\n"
    response_text += f"Status: {status_dict['status']}\n"
    response_text += f"Started: {status_dict['start_time']}\n"
    response_text += f"Runtime: {status_dict['runtime']}\n"
    
    # Add result or error if available
    if process_info.status == "completed":
        # For completed process, display first 500 chars of result
        preview = (process_info.result[:500] + "...") if len(process_info.result) > 500 else process_info.result
        response_text += f"\nOutput preview:\n{preview}\n"
        response_text += f"\nFull output is available in: {process_info.log_file}"
    elif process_info.status == "failed" or process_info.status == "error":
        error_text = process_info.error if process_info.error else "Unknown error"
        response_text += f"\nError: {error_text}\n"
        response_text += f"\nFull log is available in: {process_info.log_file}"
    
    return response_text

@mcp.tool()
async def get_process_output(process_id: str, max_lines: int = 100) -> str:
    """
    Get the output of a process.
    
    Args:
        process_id: The ID of the process
        max_lines: Maximum number of lines to return
    """
    if process_id not in PROCESSES:
        return f"Process with ID {process_id} not found"
    
    process_info = PROCESSES[process_id]
    
    try:
        with open(process_info.log_file, "r") as log_file:
            lines = log_file.readlines()
            
        # Limit the number of lines
        if len(lines) > max_lines:
            output = f"Showing last {max_lines} lines of output...\n"
            output += "".join(lines[-max_lines:])
        else:
            output = "".join(lines)
            
        return output
    except Exception as e:
        return f"Error reading log file: {str(e)}"

@mcp.resource("processes://list")
def list_processes() -> str:
    """Resource that lists all processes."""
    if not PROCESSES:
        return "No processes have been started."
    
    process_list = []
    for pid, info in PROCESSES.items():
        status_dict = info.to_dict()
        process_list.append(
            f"ID: {pid} | Command: {status_dict['command'][:50]}... | "
            f"Status: {status_dict['status']} | Runtime: {status_dict['runtime']}"
        )
    
    return "\n".join(process_list)

@mcp.tool()
async def list_all_processes() -> str:
    """List all processes with their status."""
    if not PROCESSES:
        return "No processes have been started."
    
    process_list = []
    for pid, info in PROCESSES.items():
        status_dict = info.to_dict()
        process_list.append(
            f"ID: {pid}\n"
            f"Command: {status_dict['command'][:50]}...\n"
            f"Status: {status_dict['status']}\n"
            f"Started: {status_dict['start_time']}\n"
            f"Runtime: {status_dict['runtime']}\n"
        )
    
    return "\n".join(process_list)

if __name__ == "__main__":
    mcp.run()