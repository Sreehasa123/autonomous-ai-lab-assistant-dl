"""Safe python execution sandbox for running agent-generated scripts."""

import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional


class ExecutionResult:
    def __init__(
        self,
        exit_code: int,
        stdout: str,
        stderr: str,
        duration_sec: float,
        timed_out: bool = False,
    ):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.duration_sec = duration_sec
        self.timed_out = timed_out

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class SandboxedRunner:
    """Executes generated training/preprocessing scripts with strict timeout and pipe management."""

    def __init__(self, timeout_sec: int = 600):
        self.timeout_sec = timeout_sec

    def run_script(
        self, script_path: Path, args: Optional[list] = None
    ) -> ExecutionResult:
        start_time = time.time()
        cmd = [sys.executable, str(script_path)] + (args or [])

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = process.communicate(timeout=self.timeout_sec)
            duration = time.time() - start_time
            return ExecutionResult(
                exit_code=process.returncode,
                stdout=stdout,
                stderr=stderr,
                duration_sec=duration,
            )
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            duration = time.time() - start_time
            return ExecutionResult(
                exit_code=-1,
                stdout=stdout,
                stderr=stderr,
                duration_sec=duration,
                timed_out=True,
            )
        except Exception as e:
            duration = time.time() - start_time
            return ExecutionResult(
                exit_code=1,
                stdout="",
                stderr=str(e),
                duration_sec=duration,
            )
