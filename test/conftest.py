import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_agent(*flags: str, port: int = 0) -> tuple[subprocess.Popen, str]:
    """Start test/a2a_agent.py on ``port`` (a free one by default) and wait for its card."""
    port = port or free_port()
    proc = subprocess.Popen([sys.executable, str(Path(__file__).with_name("a2a_agent.py")),
                             str(port), *flags])
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            if httpx.get(url + "/.well-known/agent-card.json").status_code == 200:
                return proc, url
        except httpx.TransportError:
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError("the a2a-sdk test agent did not start")


@pytest.fixture(scope="session")
def a2a_agent_url():
    """Base URL of a real a2a-sdk agent shared by the whole test session."""
    proc, url = start_agent()
    yield url
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture
def a2a_agent_process():
    """A private real a2a-sdk agent the test may kill: ``(process, base_url)``."""
    proc, url = start_agent()
    yield proc, url
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=10)


@pytest.fixture
def start_a2a_agent():
    """``start(*flags, port=0)`` starts a private agent; the fixture kills every one it started."""
    started = []

    def start(*flags: str, port: int = 0) -> tuple[subprocess.Popen, str]:
        proc, url = start_agent(*flags, port=port)
        started.append(proc)
        return proc, url

    yield start
    for proc in started:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
