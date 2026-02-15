import subprocess
from typing import Dict, List, Optional


def run_command_capture_stream(
    cmd: List[str],
    env: Optional[Dict[str, str]] = None,
    prefix: str = "",
) -> subprocess.CompletedProcess[str]:
    """Run a command while streaming output to console and capturing it."""
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines: List[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line)
        text = line.rstrip()
        if prefix:
            print(f"{prefix} {text}")
        else:
            print(text)

    return_code = process.wait()
    output = "".join(lines)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd, output=output, stderr=output)
    return subprocess.CompletedProcess(cmd, return_code, stdout=output, stderr=output)

