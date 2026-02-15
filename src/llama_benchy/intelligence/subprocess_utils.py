import subprocess
from typing import Dict, List, Optional


def run_command_capture_stream(
    cmd: List[str],
    env: Optional[Dict[str, str]] = None,
    prefix: str = "",
    cwd: Optional[str] = None,
    log_file: Optional[str] = None,
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
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line)
        text = line.rstrip()
        if log_handle:
            log_handle.write(line)
            log_handle.flush()
        if prefix:
            print(f"{prefix} {text}")
        else:
            print(text)

    return_code = process.wait()
    output = "".join(lines)
    if log_handle:
        log_handle.close()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd, output=output, stderr=output)
    return subprocess.CompletedProcess(cmd, return_code, stdout=output, stderr=output)

