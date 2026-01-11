"""Tests for the agent module."""


from tinker_agent.agent import (
    _is_write_tool,
    _is_write_command,
    _extract_paths_from_input,
    _is_path_within_root,
    _find_latest_results_dir,
)


class TestPathSandboxing:
    """Tests for path sandboxing helper functions."""

    def test_is_path_within_root_direct_child(self, tmp_path):
        """Path directly inside root should be allowed."""
        child = tmp_path / "file.txt"
        assert _is_path_within_root(str(child), tmp_path) is True

    def test_is_path_within_root_nested(self, tmp_path):
        """Deeply nested path should be allowed."""
        nested = tmp_path / "a" / "b" / "c" / "file.txt"
        assert _is_path_within_root(str(nested), tmp_path) is True

    def test_is_path_within_root_outside(self, tmp_path):
        """Path outside root should be denied."""
        outside = tmp_path.parent / "outside.txt"
        assert _is_path_within_root(str(outside), tmp_path) is False

    def test_is_path_within_root_escape_attempt(self, tmp_path):
        """Path with .. trying to escape should be denied."""
        escape = tmp_path / "subdir" / ".." / ".." / "outside.txt"
        assert _is_path_within_root(str(escape), tmp_path) is False

    def test_is_write_tool_recognizes_write_tools(self):
        """Should recognize common write tools."""
        write_tools = [
            "Write",
            "write",
            "Edit",
            "edit",
            "Delete",
            "delete",
            "StrReplace",
        ]
        for tool in write_tools:
            assert _is_write_tool(tool) is True, f"{tool} should be a write tool"

    def test_is_write_tool_recognizes_read_tools(self):
        """Should not flag read-only tools."""
        read_tools = ["Read", "read", "Glob", "glob", "Grep", "grep", "LS", "ls"]
        for tool in read_tools:
            assert _is_write_tool(tool) is False, f"{tool} should not be a write tool"

    def test_is_write_command_recognizes_rm(self):
        """Should recognize rm command."""
        assert _is_write_command("rm file.txt") is True
        assert _is_write_command("rm -rf dir/") is True

    def test_is_write_command_recognizes_echo_redirect(self):
        """Should recognize echo with redirect."""
        assert _is_write_command("echo 'hello' > file.txt") is True

    def test_is_write_command_allows_read_commands(self):
        """Should allow read-only commands."""
        assert _is_write_command("cat file.txt") is False
        assert _is_write_command("ls -la") is False
        assert _is_write_command("grep pattern file.txt") is False


class TestPathExtraction:
    """Tests for _extract_paths_from_input."""

    def test_extract_path_parameter(self):
        """Should extract path parameter."""
        paths = _extract_paths_from_input("Read", {"path": "/home/user/file.txt"})
        assert "/home/user/file.txt" in paths

    def test_extract_file_path_parameter(self):
        """Should extract file_path parameter."""
        paths = _extract_paths_from_input("Write", {"file_path": "/tmp/out.txt"})
        assert "/tmp/out.txt" in paths

    def test_extract_from_bash_command(self):
        """Should extract paths from bash commands."""
        paths = _extract_paths_from_input("Bash", {"command": "cat /etc/passwd"})
        assert "/etc/passwd" in paths

    def test_extract_home_dir_reference(self):
        """Should detect home directory references."""
        paths = _extract_paths_from_input("Bash", {"command": "cat ~/secrets.txt"})
        # Should return a path outside typical sandbox
        assert any("/home" in p for p in paths)


class TestFindResultsDir:
    """Tests for _find_latest_results_dir."""

    def test_finds_direct_results_dir(self, tmp_path):
        """Should find results/ directly in cwd."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        found = _find_latest_results_dir(tmp_path)
        assert found == results_dir

    def test_finds_results_in_runs(self, tmp_path):
        """Should find results/ inside runs/*/."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        run1 = runs_dir / "20250101_120000"
        run1.mkdir()
        (run1 / "results").mkdir()

        found = _find_latest_results_dir(tmp_path)
        assert found == run1 / "results"

    def test_finds_latest_run(self, tmp_path):
        """Should find the most recent run by name."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        # Create multiple runs
        run_old = runs_dir / "20250101_120000"
        run_old.mkdir()
        (run_old / "results").mkdir()

        run_new = runs_dir / "20250115_180000"
        run_new.mkdir()
        (run_new / "results").mkdir()

        found = _find_latest_results_dir(tmp_path)
        assert found == run_new / "results"

    def test_returns_none_when_not_found(self, tmp_path):
        """Should return None when no results directory exists."""
        found = _find_latest_results_dir(tmp_path)
        assert found is None
