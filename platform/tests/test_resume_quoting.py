"""A crafted folder name cannot inject commands into a session resume."""

import os
import subprocess

import pytest

from session_core.resume import ps_encoded, ps_single_quote

RIGHT_QUOTE = "’"


@pytest.mark.parametrize("quote", ["'", "‘", RIGHT_QUOTE, "‚", "‛"])
def test_every_powershell_single_quote_is_doubled(quote):
    assert ps_single_quote(f"a{quote}b") == f"'a{quote}{quote}b'"


@pytest.mark.skipif(os.name != "nt", reason="needs Windows PowerShell")
def test_a_malicious_folder_name_stays_one_string_in_real_powershell():
    name = f"x{RIGHT_QUOTE}; Write-Output INJECTED; {RIGHT_QUOTE}"
    command = f"Write-Output {ps_single_quote(name)}"
    out = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", ps_encoded(command)],
        capture_output=True, text=True, encoding="utf-8", timeout=60, check=False,
    ).stdout
    lines = [line for line in out.splitlines() if line.strip()]
    assert lines == [name] or (len(lines) == 1 and "INJECTED" in lines[0])
    assert "INJECTED" not in [line.strip() for line in lines]
