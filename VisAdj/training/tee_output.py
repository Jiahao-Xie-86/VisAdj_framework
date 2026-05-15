"""
Tee output to both console and file - captures tqdm progress bars
"""

import sys
import re
from pathlib import Path


class TeeOutput:
    """Redirect stdout/stderr to both console and file."""
    
    def __init__(self, log_file: str, mode: str = 'w'):
        """
        Args:
            log_file: Path to log file
            mode: File open mode ('w' to overwrite, 'a' to append)
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.terminal = sys.stdout
        self.log = open(self.log_file, mode, buffering=1)  # Line buffered
        # Compile ANSI escape code regex once
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        
    def write(self, message):
        """Write to both terminal and file."""
        try:
            if message:  # Only write non-empty messages
                self.terminal.write(message)
                self.terminal.flush()
                # Filter out ANSI escape codes for file (tqdm uses these for progress bars)
                clean_message = self.ansi_escape.sub('', message)
                if clean_message.strip():  # Only write if there's content after cleaning
                    self.log.write(clean_message)
                    self.log.flush()
        except (AttributeError, ValueError):
            # Handle cases where sys.meta_path is None during shutdown
            pass
    
    def flush(self):
        """Flush both streams."""
        self.terminal.flush()
        self.log.flush()
    
    def __enter__(self):
        """Context manager entry."""
        sys.stdout = self
        sys.stderr = self
        return self
    
    def __exit__(self, *args):
        """Context manager exit."""
        sys.stdout = self.terminal
        sys.stderr = self.terminal
        self.log.close()

