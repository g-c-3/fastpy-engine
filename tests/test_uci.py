"""
FastPy-Engine — UCI Protocol Integration Tests
===============================================
Tests the engine's UCI I/O by spawning it as a subprocess and exchanging
commands exactly the way Arena, Cutechess, and python-chess do.

Run:
    cd fastpy-engine/
    pytest tests/test_uci.py -v

The engine must be in the repo root as engine.py.
"""

import pytest
import queue
import subprocess
import threading
import time
from pathlib import Path


# =============================================================================
# FIXTURE — spawns one engine process per test function
# =============================================================================

ENGINE_CMD = ["python3", str(Path(__file__).parent.parent / "run.py")]


class UCISession:
    """
    Thin wrapper around a running engine process.
    Provides send() and collect_until() for test readability.
    """

    def __init__(self):
        self._proc = subprocess.Popen(
            ENGINE_CMD,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._q: queue.Queue = queue.Queue()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self):
        """Background thread: pulls lines from engine stdout into a queue."""
        for line in self._proc.stdout:
            self._q.put(line.rstrip())

    def send(self, cmd: str) -> None:
        """Send a UCI command to the engine."""
        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.flush()

    def collect_until(self, keyword: str, timeout: float = 5.0) -> list[str]:
        """
        Collect output lines until a line containing `keyword` is received,
        or `timeout` seconds elapse.  Returns all lines collected (inclusive).
        """
        lines: list[str] = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                line = self._q.get(timeout=0.05)
                lines.append(line)
                if keyword in line:
                    return lines
            except queue.Empty:
                continue
        return lines  # timed out — caller must assert what it needs

    def quit(self) -> None:
        """Send quit and wait for the process to exit."""
        try:
            self.send("quit")
            self._proc.wait(timeout=3.0)
        except Exception:
            self._proc.kill()


@pytest.fixture
def engine():
    """Provide a fresh UCISession for each test; quit cleanly on teardown."""
    session = UCISession()
    yield session
    session.quit()


# =============================================================================
# HELPER
# =============================================================================

def bestmove_from(lines: list[str]) -> str | None:
    """Extract the bestmove token from a list of engine output lines."""
    for line in lines:
        if line.startswith("bestmove"):
            return line
    return None


# =============================================================================
# UCI HANDSHAKE
# =============================================================================

class TestUCIHandshake:
    def test_uci_sends_id_name(self, engine):
        engine.send("uci")
        lines = engine.collect_until("uciok")
        assert any("id name FastPy-Engine" in l for l in lines), \
            f"Expected 'id name FastPy-Engine' in: {lines}"

    def test_uci_sends_id_author(self, engine):
        engine.send("uci")
        lines = engine.collect_until("uciok")
        assert any("id author" in l for l in lines), \
            f"Expected 'id author' in: {lines}"

    def test_uci_ends_with_uciok(self, engine):
        engine.send("uci")
        lines = engine.collect_until("uciok")
        assert "uciok" in lines, f"Expected 'uciok' in: {lines}"

    def test_isready_returns_readyok(self, engine):
        engine.send("uci")
        engine.collect_until("uciok")
        engine.send("isready")
        lines = engine.collect_until("readyok")
        assert "readyok" in lines, f"Expected 'readyok' in: {lines}"

    def test_isready_without_uci_returns_readyok(self, engine):
        """isready must respond even if uci was not sent first."""
        engine.send("isready")
        lines = engine.collect_until("readyok")
        assert "readyok" in lines

    def test_unknown_commands_ignored(self, engine):
        """Unknown commands must not crash the engine."""
        engine.send("nonexistent_command")
        engine.send("isready")
        lines = engine.collect_until("readyok")
        assert "readyok" in lines


# =============================================================================
# POSITION PARSING
# =============================================================================

class TestPosition:
    def test_position_startpos_no_moves(self, engine):
        """position startpos sets up the initial position — engine must respond to go."""
        engine.send("position startpos")
        engine.send("go depth 1")
        lines = engine.collect_until("bestmove", timeout=10.0)
        bm = bestmove_from(lines)
        assert bm is not None, f"No bestmove in: {lines}"
        assert bm != "bestmove 0000", "Got null move — no legal moves found"

    def test_position_startpos_single_move(self, engine):
        """position startpos moves e2e4 applies one move."""
        engine.send("position startpos moves e2e4")
        engine.send("go depth 1")
        lines = engine.collect_until("bestmove", timeout=10.0)
        bm = bestmove_from(lines)
        assert bm is not None, f"No bestmove after e2e4: {lines}"

    def test_position_startpos_two_moves(self, engine):
        """position startpos moves e2e4 e7e5 applies two moves."""
        engine.send("position startpos moves e2e4 e7e5")
        engine.send("go depth 1")
        lines = engine.collect_until("bestmove", timeout=10.0)
        bm = bestmove_from(lines)
        assert bm is not None, f"No bestmove after e2e4 e7e5: {lines}"

    def test_position_startpos_four_moves(self, engine):
        """Apply 4 moves — verify engine still responds."""
        engine.send("position startpos moves e2e4 e7e5 g1f3 b8c6")
        engine.send("go depth 1")
        lines = engine.collect_until("bestmove", timeout=10.0)
        bm = bestmove_from(lines)
        assert bm is not None, f"No bestmove after 4 moves: {lines}"

    def test_ucinewgame_resets_board(self, engine):
        """ucinewgame resets to the starting position."""
        engine.send("position startpos moves e2e4 e7e5")
        engine.send("ucinewgame")
        engine.send("position startpos")
        engine.send("go depth 1")
        lines = engine.collect_until("bestmove", timeout=10.0)
        bm = bestmove_from(lines)
        assert bm is not None, f"No bestmove after ucinewgame: {lines}"


# =============================================================================
# SEARCH OUTPUT
# =============================================================================

class TestSearch:
    def test_go_depth_1_returns_bestmove(self, engine):
        engine.send("position startpos")
        engine.send("go depth 1")
        lines = engine.collect_until("bestmove", timeout=10.0)
        bm = bestmove_from(lines)
        assert bm is not None and bm != "bestmove 0000"

    def test_go_depth_2_returns_bestmove(self, engine):
        engine.send("position startpos")
        engine.send("go depth 2")
        lines = engine.collect_until("bestmove", timeout=30.0)
        bm = bestmove_from(lines)
        assert bm is not None and bm != "bestmove 0000"

    def test_bestmove_format_is_4_chars(self, engine):
        """bestmove token must be 4 characters: file+rank+file+rank (e.g. e2e4)."""
        engine.send("position startpos")
        engine.send("go depth 1")
        lines = engine.collect_until("bestmove", timeout=10.0)
        bm = bestmove_from(lines)
        assert bm is not None
        # e.g. "bestmove e2e4" → token "e2e4" is 4 chars
        token = bm.split()[1]
        assert len(token) == 4, f"Expected 4-char move token, got: '{token}'"

    def test_bestmove_token_is_valid_algebraic(self, engine):
        """bestmove token must match algebraic square notation."""
        engine.send("position startpos")
        engine.send("go depth 1")
        lines = engine.collect_until("bestmove", timeout=10.0)
        bm = bestmove_from(lines)
        assert bm is not None
        token = bm.split()[1]
        # File chars must be a-h, rank chars must be 1-8
        assert token[0] in 'abcdefgh', f"Bad from-file: {token}"
        assert token[1] in '12345678', f"Bad from-rank: {token}"
        assert token[2] in 'abcdefgh', f"Bad to-file: {token}"
        assert token[3] in '12345678', f"Bad to-rank: {token}"

    def test_multiple_searches_in_sequence(self, engine):
        """Engine must handle multiple go commands without state corruption."""
        for move_seq in ["", "e2e4", "e2e4 e7e5", "e2e4 e7e5 g1f3"]:
            if move_seq:
                engine.send(f"position startpos moves {move_seq}")
            else:
                engine.send("position startpos")
            engine.send("go depth 1")
            lines = engine.collect_until("bestmove", timeout=10.0)
            bm = bestmove_from(lines)
            assert bm is not None, f"No bestmove after '{move_seq}': {lines}"

    def test_go_with_no_depth_uses_default(self, engine):
        """'go' without 'depth' should still return a bestmove."""
        engine.send("position startpos")
        engine.send("go")
        lines = engine.collect_until("bestmove", timeout=30.0)
        bm = bestmove_from(lines)
        assert bm is not None, f"No bestmove for bare 'go': {lines}"


# =============================================================================
# ROBUSTNESS
# =============================================================================

class TestRobustness:
    def test_stop_does_not_crash(self, engine):
        engine.send("stop")
        engine.send("isready")
        lines = engine.collect_until("readyok")
        assert "readyok" in lines

    def test_setoption_silently_ignored(self, engine):
        engine.send("setoption name Hash value 128")
        engine.send("isready")
        lines = engine.collect_until("readyok")
        assert "readyok" in lines

    def test_debug_silently_ignored(self, engine):
        engine.send("debug on")
        engine.send("isready")
        lines = engine.collect_until("readyok")
        assert "readyok" in lines

    def test_position_fen_does_not_crash(self, engine):
        """position fen (not yet supported) must not crash the engine."""
        engine.send("position fen rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
        engine.send("isready")
        lines = engine.collect_until("readyok")
        assert "readyok" in lines, f"Engine crashed on 'position fen': {lines}"
