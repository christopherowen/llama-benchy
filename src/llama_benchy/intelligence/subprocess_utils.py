import subprocess
import sys
from typing import Callable, Dict, List, Optional


def run_command_capture_stream(
    cmd: List[str],
    env: Optional[Dict[str, str]] = None,
    prefix: str = "",
    cwd: Optional[str] = None,
    log_file: Optional[str] = None,
    console_filter: Optional[Callable[[str], bool]] = None,
    progress_line: Optional[Callable[[str], Optional[str]]] = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command while streaming output to console and capturing it."""
    process = subprocess.Popen(
        cmd,
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines: List[str] = []
    log_handle = open(log_file, "a", encoding="utf-8") if log_file else None
    active_progress_len = 0
    try:
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line)
            text = line.rstrip()
            if log_handle:
                log_handle.write(line)
                log_handle.flush()
            if progress_line is not None:
                status = progress_line(text)
                if status:
                    if sys.stdout.isatty():
                        rendered = f"{prefix} {status}" if prefix else status
                        # Clear prior longer status and repaint in-place.
                        padded = rendered.ljust(active_progress_len)
                        sys.stdout.write("\r" + padded)
                        sys.stdout.flush()
                        active_progress_len = max(active_progress_len, len(rendered))
                    continue

            if active_progress_len > 0 and sys.stdout.isatty():
                # Clear any in-place progress line before printing regular output.
                sys.stdout.write("\r" + (" " * active_progress_len) + "\r")
                sys.stdout.flush()
                active_progress_len = 0
            if console_filter is None or console_filter(text):
                if prefix:
                    print(f"{prefix} {text}")
                else:
                    print(text)
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
        raise
    finally:
        if log_handle:
            log_handle.close()
        if active_progress_len > 0 and sys.stdout.isatty():
            sys.stdout.write("\n")
            sys.stdout.flush()

    return_code = process.wait()
    output = "".join(lines)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd, output=output, stderr=output)
    return subprocess.CompletedProcess(cmd, return_code, stdout=output, stderr=output)

