from __future__ import annotations

from lingua_eval.code_auditor import PythonCodeVerifier


def test_valid_code_execution() -> None:
    verifier = PythonCodeVerifier(timeout_seconds=1.0)
    result = verifier.verify(
        "```python\ndef add(a, b):\n    return a + b\n```",
        ["assert add(2, 3) == 5"],
    )

    assert result.syntax_ok
    assert result.safe_to_execute
    assert result.executed
    assert result.passed
    assert result.execution_time_ms >= 0


def test_syntax_error_is_caught() -> None:
    result = PythonCodeVerifier().verify("```python\ndef broken(:\n    pass\n```")

    assert not result.syntax_ok
    assert not result.executed
    assert any(issue.category == "syntax" for issue in result.issues)


def test_infinite_loop_times_out() -> None:
    result = PythonCodeVerifier(timeout_seconds=0.2).verify(
        "```python\nwhile True:\n    pass\n```"
    )

    assert result.executed
    assert not result.passed
    assert any(issue.category == "timeout" for issue in result.issues)


def test_dangerous_ast_nodes_are_blocked() -> None:
    result = PythonCodeVerifier().verify(
        "```python\nimport subprocess\nsubprocess.run(['whoami'])\n```"
    )

    assert not result.safe_to_execute
    assert not result.executed
    assert any(issue.category == "dangerous_import" for issue in result.issues)
    assert any(issue.category == "dangerous_call" for issue in result.issues)
