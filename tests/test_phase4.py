"""
FastPy-Engine — Phase 4 Search Tests
=====================================
Tests for:
  - piece_at_square: piece value lookup
  - mvv_lva: move scoring
  - sort_moves: move ordering
  - generate_captures (via Python wrapper): captures-only move gen
  - quiescence (via Python wrapper): stand-pat + capture search
  - Iterative deepening via run.py
  - Time-management UCI commands (movetime, wtime/btime)

Note on Python-mode limitations:
  generate_captures() and quiescence() have internal uint64[218] local arrays
  (FastPy stack arrays). These are unbound in Python just like
  generate_legal_moves() and alpha_beta(). Tests use r._generate_captures_py
  and r._quiescence_py (Python wrappers in run.py). The compiled C++ functions
  are verified by the fastpy emit + g++ compile step.

Run: pytest tests/test_phase4.py -v
"""

import copy
import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from engine import (
    BoardState, make_move, evaluate,
    encode_move, encode_move_promo, encode_move_flag,
    move_from, move_to, move_promo, move_flag,
    piece_at_square, mvv_lva, sort_moves,
    generate_all_moves, is_in_check,
    BIT_ONE,
    VAL_PAWN, VAL_KNIGHT, VAL_BISHOP, VAL_ROOK, VAL_QUEEN,
    FLAG_EN_PASSANT, FLAG_CASTLING,
    NEG_INF, INF,
    PROMO_QUEEN,
)
import run as r


# =============================================================================
# HELPERS
# =============================================================================

def starting_board():
    return BoardState()


def _legal_moves_py(board):
    pseudo = [0] * 218
    pcount = generate_all_moves(board, pseudo, 0)
    out = []
    for i in range(pcount):
        nb = make_move(copy.copy(board), pseudo[i])
        if not is_in_check(nb):
            out.append(pseudo[i])
    return out


# =============================================================================
# piece_at_square
# =============================================================================

class TestPieceAtSquare:
    def test_white_pawn_on_e2(self):
        board = starting_board()
        assert piece_at_square(12, board) == VAL_PAWN   # e2

    def test_white_knight_on_g1(self):
        board = starting_board()
        assert piece_at_square(6, board) == VAL_KNIGHT  # g1

    def test_white_bishop_on_c1(self):
        board = starting_board()
        assert piece_at_square(2, board) == VAL_BISHOP  # c1

    def test_white_rook_on_a1(self):
        board = starting_board()
        assert piece_at_square(0, board) == VAL_ROOK    # a1

    def test_white_queen_on_d1(self):
        board = starting_board()
        assert piece_at_square(3, board) == VAL_QUEEN   # d1

    def test_black_pawn_on_e7(self):
        board = starting_board()
        assert piece_at_square(52, board) == VAL_PAWN   # e7

    def test_black_rook_on_h8(self):
        board = starting_board()
        assert piece_at_square(63, board) == VAL_ROOK   # h8

    def test_empty_e4(self):
        assert piece_at_square(28, starting_board()) == 0

    def test_empty_e5(self):
        assert piece_at_square(36, starting_board()) == 0

    def test_king_returns_zero(self):
        # King not in MVV-LVA table — returns 0
        assert piece_at_square(4, starting_board()) == 0   # e1


# =============================================================================
# mvv_lva
# =============================================================================

class TestMvvLva:
    def test_quiet_move_scores_zero(self):
        board = starting_board()
        move = encode_move(12, 28)   # e2e4 (quiet push)
        assert mvv_lva(move, board) == 0

    def test_knight_takes_pawn(self):
        board = starting_board()
        board.white_pawns = 0
        board.white_knights = BIT_ONE << 21   # f3
        board.black_pawns = BIT_ONE << 27     # d4
        move = encode_move(21, 27)
        score = mvv_lva(move, board)
        assert score == VAL_PAWN * 10 - VAL_KNIGHT

    def test_pawn_captures_queen_highest(self):
        board = starting_board()
        board.white_pawns = BIT_ONE << 27     # d4
        board.black_queens = BIT_ONE << 36    # e5
        move = encode_move(27, 36)
        assert mvv_lva(move, board) == VAL_QUEEN * 10 - VAL_PAWN

    def test_queen_captures_pawn_lower(self):
        board = starting_board()
        board.white_queens = BIT_ONE << 27
        board.black_pawns = BIT_ONE << 36
        move = encode_move(27, 36)
        assert mvv_lva(move, board) == VAL_PAWN * 10 - VAL_QUEEN

    def test_pxq_higher_priority_than_qxp(self):
        board_pxq = starting_board()
        board_pxq.white_pawns = BIT_ONE << 27
        board_pxq.black_queens = BIT_ONE << 36
        pxq = mvv_lva(encode_move(27, 36), board_pxq)

        board_qxp = starting_board()
        board_qxp.white_queens = BIT_ONE << 27
        board_qxp.black_pawns = BIT_ONE << 36
        qxp = mvv_lva(encode_move(27, 36), board_qxp)

        assert pxq > qxp


# =============================================================================
# sort_moves
# =============================================================================

class TestSortMoves:
    def test_sort_preserves_move_set(self):
        board = starting_board()
        moves = [0] * 218
        count = generate_all_moves(board, moves, 0)
        before = set(moves[i] for i in range(count))
        sort_moves(moves, count, board)
        after = set(moves[i] for i in range(count))
        assert before == after

    def test_capture_sorted_first(self):
        board = starting_board()
        board = make_move(board, encode_move(12, 28))  # e2e4
        board = make_move(board, encode_move(52, 36))  # e7e5
        board = make_move(board, encode_move(11, 27))  # d2d4

        moves = [0] * 218
        count = generate_all_moves(board, moves, 0)
        sort_moves(moves, count, board)

        # First move after sorting must be a capture (score > 0)
        assert mvv_lva(moves[0], board) > 0

    def test_sort_handles_empty_list(self):
        board = starting_board()
        moves = [0] * 218
        sort_moves(moves, 0, board)   # must not crash


# =============================================================================
# generate_captures (via Python wrapper r._generate_captures_py)
# =============================================================================

class TestGenerateCaptures:

    def test_no_captures_from_starting_position(self):
        board = starting_board()
        caps, count = r._generate_captures_py(board)
        assert count == 0

    def test_captures_available_after_open(self):
        board = starting_board()
        board = make_move(board, encode_move(12, 28))  # e2e4
        board = make_move(board, encode_move(52, 36))  # e7e5
        board = make_move(board, encode_move(11, 27))  # d2d4

        _, all_count = r._generate_legal_moves_py(board)
        _, cap_count = r._generate_captures_py(board)

        assert cap_count >= 1           # d4xe5
        assert cap_count < all_count    # Fewer than all moves

    def test_all_captures_land_on_enemy(self):
        # 4 half-moves → white to move; white can capture black's e5 pawn
        board = starting_board()
        board = make_move(board, encode_move(12, 28))  # e2e4
        board = make_move(board, encode_move(52, 36))  # e7e5
        board = make_move(board, encode_move(11, 27))  # d2d4
        board = make_move(board, encode_move(48, 40))  # a7a6 → white to move

        assert board.white_to_move
        enemy_bb = (board.black_pawns | board.black_knights |
                    board.black_bishops | board.black_rooks |
                    board.black_queens | board.black_king)

        caps, cap_count = r._generate_captures_py(board)
        assert cap_count >= 1   # d4xe5 available

        for i in range(cap_count):
            to_sq = move_to(caps[i])
            to_bb = 1 << to_sq
            is_ep = move_flag(caps[i]) == FLAG_EN_PASSANT
            assert (to_bb & enemy_bb) or is_ep, \
                f'Capture {i} to sq {to_sq} not on enemy'

    def test_captures_are_legal(self):
        board = starting_board()
        board = make_move(board, encode_move(12, 28))
        board = make_move(board, encode_move(52, 36))

        caps, cap_count = r._generate_captures_py(board)

        for i in range(cap_count):
            nb = make_move(copy.copy(board), caps[i])
            assert not is_in_check(nb), \
                f'Capture {i} leaves king in check'


# =============================================================================
# quiescence (via Python wrapper r._quiescence_py)
# =============================================================================

class TestQuiescence:

    def test_quiet_position_equals_static_eval(self):
        board = starting_board()
        static = evaluate(board)
        q = r._quiescence_py(board, -INF, INF)
        assert q == static

    def test_returns_int(self):
        board = starting_board()
        assert isinstance(r._quiescence_py(board, NEG_INF, INF), int)

    def test_stand_pat_prunes_at_beta(self):
        board = starting_board()
        # static eval (0) >= beta (-10000): return beta
        result = r._quiescence_py(board, -INF, -10000)
        assert result == -10000

    def test_after_open_position_reasonable(self):
        board = starting_board()
        board = make_move(board, encode_move(12, 28))  # e2e4
        board = make_move(board, encode_move(52, 36))  # e7e5
        q = r._quiescence_py(board, NEG_INF, INF)
        assert isinstance(q, int)
        assert -5000 < q < 5000

    def test_respects_alpha_beta_window(self):
        board = starting_board()
        q = r._quiescence_py(board, -100, 100)
        assert -100 <= q <= 100


# =============================================================================
# Alpha-Beta with quiescence (via Python wrapper)
# =============================================================================

class TestAlphaBetaPy:
    def test_depth0_returns_qsearch(self):
        board = starting_board()
        # depth 0 calls quiescence which returns static eval for quiet positions
        result = r._alpha_beta_py(board, 0, NEG_INF, INF)
        assert result == 0

    def test_depth1_returns_int(self):
        result = r._alpha_beta_py(starting_board(), 1, NEG_INF, INF)
        assert isinstance(result, int)

    def test_depth2_no_crash(self):
        result = r._alpha_beta_py(starting_board(), 2, NEG_INF, INF)
        assert isinstance(result, int)

    def test_respects_window(self):
        result = r._alpha_beta_py(starting_board(), 2, -50, 50)
        assert -50 <= result <= 50

    def test_capture_first_after_open(self):
        board = starting_board()
        board = make_move(board, encode_move(12, 28))
        board = make_move(board, encode_move(52, 36))
        board = make_move(board, encode_move(11, 27))

        moves, count = r._generate_legal_moves_py(board)
        move_list = [(moves[i], r._mvv_lva_py(moves[i], board))
                     for i in range(count)]
        move_list.sort(key=lambda x: -x[1])
        # First sorted move must be a capture
        assert move_list[0][1] > 0


# =============================================================================
# Iterative Deepening
# =============================================================================

class TestIterativeDeepening:
    def test_returns_legal_move(self):
        board = starting_board()
        move, score, depth = r._iterative_deepening_py(
            board, max_time_ms=500, max_depth=3
        )
        assert move != 0
        legal = _legal_moves_py(board)
        assert move in legal

    def test_returns_completed_depth(self):
        _, _, depth = r._iterative_deepening_py(
            starting_board(), max_time_ms=500, max_depth=3
        )
        assert depth >= 1

    def test_outputs_info_lines(self):
        import io
        board = starting_board()
        buf = io.StringIO()
        orig = sys.stdout
        sys.stdout = buf
        try:
            r._iterative_deepening_py(board, max_time_ms=200, max_depth=2)
        finally:
            sys.stdout = orig
        output = buf.getvalue()
        assert 'info depth 1' in output
        assert 'score cp' in output

    def test_find_best_move_returns_tuple(self):
        m, s = r._find_best_move_py(starting_board(), 1)
        assert isinstance(m, int) and isinstance(s, int)
        assert m != 0


# =============================================================================
# UCI time-control integration (subprocess)
# =============================================================================

class TestUCITimeControl:

    def _run_uci(self, commands, timeout=12):
        import subprocess
        from pathlib import Path
        run_path = str(Path(REPO_ROOT) / 'run.py')
        proc = subprocess.Popen(
            ['python3', run_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )
        stdin_data = '\n'.join(commands) + '\nquit\n'
        try:
            stdout, _ = proc.communicate(stdin_data, timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, _ = proc.communicate()
        return stdout

    def test_go_movetime_returns_bestmove(self):
        out = self._run_uci([
            'uci', 'isready', 'position startpos', 'go movetime 400',
        ])
        assert 'bestmove' in out

    def test_go_movetime_outputs_info(self):
        out = self._run_uci([
            'uci', 'isready', 'position startpos', 'go movetime 500',
        ])
        assert 'info depth' in out
        assert 'score cp' in out

    def test_go_wtime_btime_returns_bestmove(self):
        out = self._run_uci([
            'uci', 'isready', 'position startpos',
            'go wtime 10000 btime 10000',
        ])
        assert 'bestmove' in out

    def test_go_depth_outputs_all_info_lines(self):
        out = self._run_uci([
            'uci', 'isready', 'position startpos', 'go depth 3',
        ])
        assert 'info depth 1' in out
        assert 'info depth 2' in out
        assert 'info depth 3' in out
        assert 'bestmove' in out
