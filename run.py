"""
FastPy-Engine — Python UCI Runner
==================================
Standalone Python entry point. Imports all compiled functions from engine.py
and provides:

  - Python-mode wrappers for compiled functions that use uint64[218] stack
    arrays (generate_legal_moves, alpha_beta, quiescence, perft — unrunnable
    directly in Python due to bare uint64[218] local declarations)
  - UCI protocol loop (uci, isready, position, go, quit)
  - Iterative deepening with time management
  - Square/move string helpers

Usage:
    python run.py          ← connect any UCI-compatible GUI

FastPy compiled binary:
    fastpy build engine.py --optimize=O3
    ./engine               ← UCI loop not yet in compiled binary (Phase 4)

Phase 4 additions:
  - _quiescence_py: stand-pat + capture search to avoid horizon effect
  - _generate_captures_py: legal captures only (for quiescence)
  - _alpha_beta_py: calls _quiescence_py at depth 0, MVV-LVA ordering
  - _iterative_deepening_py: IDS with time limit + info line output
  - uci_loop: handles wtime/btime/movetime, outputs info depth lines

Author: Gokul Chandar
Project: FastPy-Engine (github.com/g-c-3/fastpy-engine)
License: GPL v3
"""

import sys
import copy
import time

from engine import (
    BoardState, make_move, evaluate,
    generate_all_moves, is_in_check, is_side_to_move_in_check,
    piece_at_square,
    encode_move, encode_move_promo, encode_move_flag,
    move_from, move_to, move_promo, move_flag,
    FLAG_EN_PASSANT, FLAG_CASTLING,
    PROMO_QUEEN, PROMO_KNIGHT, PROMO_BISHOP, PROMO_NONE,
    NEG_INF, INF,
    VAL_PAWN, VAL_KNIGHT, VAL_BISHOP, VAL_ROOK, VAL_QUEEN,
)


# =============================================================================
# PYTHON-MODE WRAPPERS
#
# The compiled functions alpha_beta(), quiescence(), generate_legal_moves(),
# and perft() declare local `uint64[218]` arrays. FastPy emits these as
# C-style stack arrays (uint64_t moves[218] = {}).
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


def _generate_captures_py(board):
    """
    Python-mode legal capture generator.
    Returns only moves that land on an enemy piece or are en passant.
    Used by _quiescence_py to search only capture moves at leaf nodes.
    """
    pseudo = [0] * 218
    pcount = generate_all_moves(board, pseudo, 0)

    if board.white_to_move:
        # enemy squares occupied by black pieces
        enemy_bb = (board.black_pawns | board.black_knights |
                    board.black_bishops | board.black_rooks |
                    board.black_queens | board.black_king)
    else:
        enemy_bb = (board.white_pawns | board.white_knights |
                    board.white_bishops | board.white_rooks |
                    board.white_queens | board.white_king)

    out = [0] * 218
    count = 0
    for i in range(pcount):
        m = pseudo[i]
        to_sq = move_to(m)
        to_bb = 1 << to_sq
        is_ep = move_flag(m) == FLAG_EN_PASSANT
        if (to_bb & enemy_bb) or is_ep:
            nb = make_move(copy.copy(board), m)
            if not is_in_check(nb):
                out[count] = m
                count += 1
    return out, count


def _mvv_lva_py(move, board):
    """
    Python-mode MVV-LVA score for move ordering.
    Higher = try first. Quiet moves score 0.
    """
    to_sq = move_to(move)
    from_sq = move_from(move)
    victim = piece_at_square(to_sq, board)
    if victim == 0:
        return 0
    attacker = piece_at_square(from_sq, board)
    return victim * 10 - attacker


def _quiescence_py(board, alpha, beta):
    """
    Python-mode quiescence search.
    Stand-pat: if static eval already beats beta, return immediately.
    Then search all captures to avoid the horizon effect.
    """
    stand_pat = evaluate(board)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    captures, cap_count = _generate_captures_py(board)
    for i in range(cap_count):
        nb = make_move(copy.copy(board), captures[i])
        score = -_quiescence_py(nb, -beta, -alpha)
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


def _alpha_beta_py(board, depth, alpha, beta):
    """
    Python-mode negamax alpha-beta search.
    Phase 4: calls _quiescence_py at depth 0, uses MVV-LVA ordering.
    """
    if depth == 0:
        return _quiescence_py(board, alpha, beta)

    moves, count = _generate_legal_moves_py(board)
    if count == 0:
        return 0

    # MVV-LVA ordering: sort by capture priority
    move_list = [(moves[i], _mvv_lva_py(moves[i], board)) for i in range(count)]
    move_list.sort(key=lambda x: -x[1])

    best = NEG_INF
    for m, _ in move_list:
        nb = make_move(copy.copy(board), m)
        score = -_alpha_beta_py(nb, depth - 1, -beta, -alpha)
        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    return best


def _find_best_move_py(board, depth):
    """Python-mode root search. Returns (best_move, best_score)."""
    moves, count = _generate_legal_moves_py(board)
    if count == 0:
        return 0, 0

    # MVV-LVA ordering at root
    move_list = [(moves[i], _mvv_lva_py(moves[i], board)) for i in range(count)]
    move_list.sort(key=lambda x: -x[1])

    best_move  = move_list[0][0]
    best_score = NEG_INF
    alpha      = NEG_INF
    beta       = INF

    for m, _ in move_list:
        nb = make_move(copy.copy(board), m)
        score = -_alpha_beta_py(nb, depth - 1, -beta, -alpha)
        if score > best_score:
            best_score = score
            best_move  = m
        if score > alpha:
            alpha = score

    return best_move, best_score


def _iterative_deepening_py(board, max_time_ms=1000, max_depth=20):
    """
    Iterative deepening search with time limit.
    Searches depth 1, 2, 3... until time runs out.

    Always completes at least depth 1 before checking the clock.
    Outputs UCI 'info depth N score cp S time T' lines as it goes.

    Returns (best_move, best_score, completed_depth).
    """
    start_ms = time.time() * 1000
    best_move = 0
    best_score = 0

    for depth in range(1, max_depth + 1):
        # Time check before starting a new depth (after depth 1 is done)
        if depth > 1:
            elapsed = time.time() * 1000 - start_ms
            # If we've used more than half the budget, the next depth
            # is likely to exceed it — stop here.
            if elapsed > max_time_ms * 0.5:
                return best_move, best_score, depth - 1

        move, score = _find_best_move_py(board, depth)
        if move != 0:
            best_move  = move
            best_score = score

        elapsed = int(time.time() * 1000 - start_ms)
        sys.stdout.write(
            f'info depth {depth} score cp {score} time {elapsed}\n'
        )
        sys.stdout.flush()

        if elapsed >= max_time_ms:
            return best_move, best_score, depth

    return best_move, best_score, max_depth


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
        uci                         → id + uciok
        isready                     → readyok
        ucinewgame                  → reset board
        position startpos [moves …] → set up board
        go depth N                  → fixed-depth search
        go movetime N               → search for N milliseconds
        go wtime N btime N          → time-control search (uses 1/20 of clock)
        go infinite                 → search until 'stop' (uses 5s budget)
        quit                        → exit

    Phase 4: iterative deepening + time management.
    Outputs 'info depth N score cp S time T' lines during search.
    """
    board         = BoardState()
    default_depth = 5          # Fallback when no time/depth specified

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
            parts = cmd.split()

            # ── Parse go parameters ──────────────────────────────────────
            depth_param  = None
            movetime_ms  = None
            wtime_ms     = None
            btime_ms     = None
            infinite     = 'infinite' in parts

            if 'depth' in parts:
                di = parts.index('depth')
                if di + 1 < len(parts):
                    try: depth_param = int(parts[di + 1])
                    except ValueError: pass

            if 'movetime' in parts:
                mi = parts.index('movetime')
                if mi + 1 < len(parts):
                    try: movetime_ms = int(parts[mi + 1])
                    except ValueError: pass

            if 'wtime' in parts:
                wi = parts.index('wtime')
                if wi + 1 < len(parts):
                    try: wtime_ms = int(parts[wi + 1])
                    except ValueError: pass

            if 'btime' in parts:
                bi = parts.index('btime')
                if bi + 1 < len(parts):
                    try: btime_ms = int(parts[bi + 1])
                    except ValueError: pass

            # ── Select search mode ───────────────────────────────────────
            if depth_param is not None:
                # Fixed depth: iterative deepening up to depth_param
                best, score, _ = _iterative_deepening_py(
                    board,
                    max_time_ms=60_000,   # Large budget so it always finishes
                    max_depth=depth_param,
                )

            elif movetime_ms is not None:
                best, score, _ = _iterative_deepening_py(
                    board,
                    max_time_ms=movetime_ms,
                )

            elif wtime_ms is not None or btime_ms is not None:
                # Time-control: use ~1/20 of the remaining clock
                if board.white_to_move and wtime_ms is not None:
                    budget = max(50, wtime_ms // 20)
                elif (not board.white_to_move) and btime_ms is not None:
                    budget = max(50, btime_ms // 20)
                else:
                    budget = 1000
                best, score, _ = _iterative_deepening_py(
                    board,
                    max_time_ms=budget,
                )

            elif infinite:
                # 'go infinite' — use 5 second budget
                best, score, _ = _iterative_deepening_py(
                    board,
                    max_time_ms=5000,
                )

            else:
                # No time info — use default depth
                best, score, _ = _iterative_deepening_py(
                    board,
                    max_time_ms=60_000,
                    max_depth=default_depth,
                )

            # ── Output bestmove ──────────────────────────────────────────
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
