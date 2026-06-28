"""
FastPy-Engine — Python UCI Runner
==================================
Standalone Python entry point. Imports all compiled functions from engine.py
and provides:

  - Python-mode wrappers for compiled functions that use uint64[218] stack
    arrays (generate_legal_moves, alpha_beta, perft — unrunnable directly
    in Python due to bare uint64[218] local declarations)
  - UCI protocol loop (uci, isready, position, go, quit)
  - Square/move string helpers

Usage:
    python run.py          ← connect any UCI-compatible GUI

FastPy compiled binary:
    fastpy build engine.py --optimize=O3
    ./engine               ← UCI loop not yet in compiled binary (Phase 4)

Author: Gokul Chandar
Project: FastPy-Engine (github.com/g-c-3/fastpy-engine)
License: GPL v3
"""

import sys
import copy

from engine import (
    BoardState, make_move, evaluate,
    generate_all_moves, is_in_check,
    encode_move, encode_move_promo, encode_move_flag,
    move_from, move_to, move_promo, move_flag,
    FLAG_EN_PASSANT, FLAG_CASTLING,
    PROMO_QUEEN, PROMO_KNIGHT, PROMO_BISHOP, PROMO_NONE,
    NEG_INF, INF,
)


# =============================================================================
# PYTHON-MODE WRAPPERS
#
# The compiled functions alpha_beta(), generate_legal_moves(), and perft()
# declare local `uint64[218]` arrays. FastPy emits these as C-style stack
# arrays (uint64_t moves[218] = {}).
#
# In Python, a bare `x: uint64[218]` annotation with no value leaves x
# unbound — the annotation is stored as a string and never evaluated
# (due to `from __future__ import annotations` in engine.py).
#
# These wrappers replicate the same logic using Python lists.
#
# CRITICAL — Python reference semantics vs C++ value semantics:
# make_move(board, move) modifies `board` in place in Python
# (objects are passed by reference). In C++, the struct is copied
# automatically on function entry, so the caller's board is untouched.
#
# Every call to make_move() here uses copy.copy(board) to get a fresh
# copy first. The compiled functions are correct without copies —
# C++ value semantics handle it automatically.
# =============================================================================

def _generate_legal_moves_py(board):
    """
    Python-mode legal move generator.
    Generates pseudo-legal moves, filters those that leave the king in check.
    Returns (moves_list, count).
    """
    pseudo = [0] * 218
    pcount = generate_all_moves(board, pseudo, 0)
    out = [0] * 218
    count = 0
    for i in range(pcount):
        nb = make_move(copy.copy(board), pseudo[i])
        if not is_in_check(nb):
            out[count] = pseudo[i]
            count += 1
    return out, count


def _alpha_beta_py(board, depth, alpha, beta):
    """Python-mode negamax alpha-beta search."""
    if depth == 0:
        return evaluate(board)
    moves, count = _generate_legal_moves_py(board)
    if count == 0:
        return 0
    best = NEG_INF
    for i in range(count):
        nb = make_move(copy.copy(board), moves[i])
        score = -_alpha_beta_py(nb, depth - 1, -beta, -alpha)
        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    return best


def _find_best_move_py(board, depth):
    """Python-mode root search. Returns best move encoded as int."""
    moves, count = _generate_legal_moves_py(board)
    best_move  = 0
    best_score = NEG_INF
    alpha      = NEG_INF
    beta       = INF
    for i in range(count):
        m  = moves[i]
        nb = make_move(copy.copy(board), m)
        score = -_alpha_beta_py(nb, depth - 1, -beta, -alpha)
        if score > best_score:
            best_score = score
            best_move  = m
        if score > alpha:
            alpha = score
    return best_move


def _perft_py(board, depth):
    """
    Python-mode perft — counts leaf nodes at the given depth.
    Correctness reference from starting position:
        perft(1)=20  perft(2)=400  perft(3)=8902
        perft(4)=197281  perft(5)=4865609
    """
    moves, count = _generate_legal_moves_py(board)
    if depth == 1:
        return count
    nodes = 0
    for i in range(count):
        nodes += _perft_py(make_move(copy.copy(board), moves[i]), depth - 1)
    return nodes


# =============================================================================
# SQUARE / MOVE STRING HELPERS
# =============================================================================

def _sq_to_str(sq):
    """Convert square index to algebraic notation: 28 → 'e4'."""
    return chr(ord('a') + (sq & 7)) + chr(ord('1') + (sq >> 3))


def _move_to_uci(move):
    """Convert packed move word to UCI string: 28 → 'e2e4', with promo suffix."""
    s = _sq_to_str(move_from(move)) + _sq_to_str(move_to(move))
    promo = move_promo(move)
    if promo == PROMO_QUEEN:  return s + 'q'
    if promo == PROMO_KNIGHT: return s + 'n'
    if promo == PROMO_BISHOP: return s + 'b'
    return s


def _parse_sq(token, offset):
    """Parse two characters of a UCI move string into a square index."""
    return (ord(token[offset + 1]) - ord('1')) * 8 + (ord(token[offset]) - ord('a'))


def _parse_uci_move(token, board):
    """
    Match a UCI move string like 'e2e4' or 'e7e8q' against the current
    pseudo-legal move list and return the matching packed move word.
    Falls back to a bare encode if no match found (shouldn't happen in
    well-formed UCI input).
    """
    from_sq    = _parse_sq(token, 0)
    to_sq      = _parse_sq(token, 2)
    want_promo = PROMO_NONE
    if len(token) >= 5:
        pc = token[4]
        if   pc == 'q': want_promo = PROMO_QUEEN
        elif pc == 'n': want_promo = PROMO_KNIGHT
        elif pc == 'b': want_promo = PROMO_BISHOP
    buf = [0] * 218
    count = generate_all_moves(board, buf, 0)
    for i in range(count):
        m = buf[i]
        if move_from(m) == from_sq and move_to(m) == to_sq:
            if want_promo == PROMO_NONE or move_promo(m) == want_promo:
                return m
    return encode_move(from_sq, to_sq) | (want_promo << 12)


def _apply_position(line):
    """
    Parse a UCI 'position' command and return the resulting BoardState.
    Handles: 'position startpos', 'position startpos moves e2e4 d7d5 ...'
    """
    board = BoardState()
    idx = line.find(' moves ')
    if idx == -1:
        return board
    for token in line[idx + 7:].split():
        if token:
            board = make_move(board, _parse_uci_move(token, board))
    return board


# =============================================================================
# UCI PROTOCOL LOOP
# =============================================================================

def uci_loop():
    """
    Main UCI loop. Reads commands from stdin, writes responses to stdout.

    Supported commands:
        uci            → id + uciok
        isready        → readyok
        ucinewgame     → reset board
        position       → set up board from startpos + moves
        go depth N     → search and output bestmove
        quit           → exit
    """
    board         = BoardState()
    default_depth = 4

    while True:
        try:
            line = sys.stdin.readline()
        except (KeyboardInterrupt, EOFError):
            break
        if not line:
            break
        cmd = line.strip()
        if not cmd:
            continue

        if cmd == 'uci':
            sys.stdout.write('id name FastPy-Engine\n')
            sys.stdout.write('id author Gokul Chandar\n')
            sys.stdout.write('uciok\n')
            sys.stdout.flush()

        elif cmd == 'isready':
            sys.stdout.write('readyok\n')
            sys.stdout.flush()

        elif cmd == 'ucinewgame':
            board = BoardState()

        elif cmd.startswith('position'):
            if 'startpos' in cmd:
                board = _apply_position(cmd)

        elif cmd.startswith('go'):
            search_depth = default_depth
            parts = cmd.split()
            if 'depth' in parts:
                di = parts.index('depth')
                if di + 1 < len(parts):
                    try:
                        search_depth = int(parts[di + 1])
                    except ValueError:
                        pass
            best = _find_best_move_py(board, search_depth)
            if best == 0:
                sys.stdout.write('bestmove 0000\n')
            else:
                sys.stdout.write('bestmove ' + _move_to_uci(best) + '\n')
            sys.stdout.flush()

        elif cmd in ('stop', 'setoption', 'register', 'debug', 'ponderhit'):
            pass

        elif cmd == 'quit':
            break


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    uci_loop()
