# =============================================================================
# FastPy-Engine — Phase 1 Engine Source
# =============================================================================
#
# A competitive chess engine written in FastPy dialect Python.
#
# FastPy compiles this file to bare-metal C++ with:
#     fastpy build engine.py --optimize=O3
#
# FastPy Speed Contract enforced throughout:
#   1. Every variable and parameter has an explicit type hint
#   2. Zero dynamic allocation — move lists are uint64[218] stack arrays
#   3. No CPython runtime dependencies in compiled functions
#
# Move encoding (packed into a single uint64):
#   Bits  0-5:   from_square  (0-63)
#   Bits  6-11:  to_square    (0-63)
#   Bits 12-13:  promotion    (0=none, 1=knight, 2=bishop, 3=queen)
#   Bits 14-15:  flags        (0=normal, 1=castling, 2=en_passant)
#
# Author: Gokul Chandar
# Project: FastPy-Engine (github.com/g-c-3/fastpy-engine)
# Compiler: FastPy (github.com/g-c-3/fastpy)
# License: GPL v3
# =============================================================================

from __future__ import annotations
from typing import Final

# =============================================================================
# TYPE ALIASES
# FastPy maps these to C++ native primitives:
#   uint64 → uint64_t   one full 64-bit bitboard
#   int32  → int32_t    scores, counts, square indices
#   bool8  → bool       single boolean flag
# =============================================================================

uint64 = int
int32  = int
bool8  = bool

# =============================================================================
# BOARD FILE AND RANK MASKS
# =============================================================================

FILE_A: Final[uint64] = 0x0101010101010101
FILE_B: Final[uint64] = 0x0202020202020202
FILE_G: Final[uint64] = 0x4040404040404040
FILE_H: Final[uint64] = 0x8080808080808080

RANK_1: Final[uint64] = 0x00000000000000FF
RANK_2: Final[uint64] = 0x000000000000FF00
RANK_4: Final[uint64] = 0x00000000FF000000
RANK_5: Final[uint64] = 0x000000FF00000000
RANK_7: Final[uint64] = 0x00FF000000000000
RANK_8: Final[uint64] = 0xFF00000000000000

FULL_BOARD: Final[uint64] = 0xFFFFFFFFFFFFFFFF

# =============================================================================
# STARTING POSITIONS
# =============================================================================

WHITE_PAWNS_START:   Final[uint64] = 0x000000000000FF00
WHITE_KNIGHTS_START: Final[uint64] = 0x0000000000000042
WHITE_BISHOPS_START: Final[uint64] = 0x0000000000000024
WHITE_ROOKS_START:   Final[uint64] = 0x0000000000000081
WHITE_QUEENS_START:  Final[uint64] = 0x0000000000000008
WHITE_KING_START:    Final[uint64] = 0x0000000000000010

BLACK_PAWNS_START:   Final[uint64] = 0x00FF000000000000
BLACK_KNIGHTS_START: Final[uint64] = 0x4200000000000000
BLACK_BISHOPS_START: Final[uint64] = 0x2400000000000000
BLACK_ROOKS_START:   Final[uint64] = 0x8100000000000000
BLACK_QUEENS_START:  Final[uint64] = 0x0800000000000000
BLACK_KING_START:    Final[uint64] = 0x1000000000000000

# =============================================================================
# SEARCH AND EVALUATION CONSTANTS
# =============================================================================

MAX_DEPTH: Final[int32] = 64
MAX_MOVES: Final[int32] = 218

INF:     Final[int32] =  32767
NEG_INF: Final[int32] = -32767

# Piece values in centipawns
VAL_PAWN:   Final[int32] = 100
VAL_KNIGHT: Final[int32] = 320
VAL_BISHOP: Final[int32] = 330
VAL_ROOK:   Final[int32] = 500
VAL_QUEEN:  Final[int32] = 900
VAL_KING:   Final[int32] = 20000

# Move flags
FLAG_NORMAL:     Final[int32] = 0
FLAG_CASTLING:   Final[int32] = 1
FLAG_EN_PASSANT: Final[int32] = 2

# Promotion codes
PROMO_NONE:   Final[int32] = 0
PROMO_KNIGHT: Final[int32] = 1
PROMO_BISHOP: Final[int32] = 2
PROMO_QUEEN:  Final[int32] = 3


# =============================================================================
# MOVE ENCODING
# Pack from/to/promotion/flags into a single uint64 — one flat array,
# no struct overhead, passes through CPU registers.
# =============================================================================

def encode_move(from_sq: int32, to_sq: int32) -> uint64:
    """Pack from and to squares into a move word."""
    return from_sq | (to_sq << 6)


def encode_move_promo(from_sq: int32, to_sq: int32, promo: int32) -> uint64:
    """Pack from, to, and promotion piece into a move word."""
    return from_sq | (to_sq << 6) | (promo << 12)


def encode_move_flag(from_sq: int32, to_sq: int32, flag: int32) -> uint64:
    """Pack from, to, and a special move flag into a move word."""
    return from_sq | (to_sq << 6) | (flag << 14)


def move_from(move: uint64) -> int32:
    """Extract the from-square (bits 0-5)."""
    return move & 63


def move_to(move: uint64) -> int32:
    """Extract the to-square (bits 6-11)."""
    return (move >> 6) & 63


def move_promo(move: uint64) -> int32:
    """Extract the promotion code (bits 12-13)."""
    return (move >> 12) & 3


def move_flag(move: uint64) -> int32:
    """Extract the special move flag (bits 14-15)."""
    return (move >> 14) & 3


# =============================================================================
# BITBOARD UTILITIES
# These compile to single-clock-cycle CPU hardware instructions via FastPy.
# =============================================================================

def popcount(board: uint64) -> int32:
    """
    Count the number of set bits (pieces) on a bitboard.
    FastPy: bin(board).count("1") → __builtin_popcountll(board)  [POPCNT, 1 cycle]
    """
    return bin(board).count("1")


def lsb(board: uint64) -> int32:
    """
    Index of the least significant bit (lowest-numbered piece square).
    FastPy: (board & -board).bit_length() - 1 → __builtin_ctzll(board)  [TZCNT, 1 cycle]
    Returns -1 for an empty board.
    """
    if board == 0:
        return -1
    return (board & -board).bit_length() - 1


def pop_lsb(board: uint64) -> uint64:
    """
    Remove the least significant bit.
    FastPy: board & (board - 1) → BLSR instruction [BMI1, 1 cycle]
    Core iteration primitive: while temp: sq = lsb(temp); temp = pop_lsb(temp)
    """
    return board & (board - 1)


# ── Directional shifts ────────────────────────────────────────────────────────

def north(board: uint64) -> uint64:
    """Shift all pieces one rank toward rank 8."""
    return (board << 8) & FULL_BOARD


def south(board: uint64) -> uint64:
    """Shift all pieces one rank toward rank 1."""
    return board >> 8


def east(board: uint64) -> uint64:
    """Shift all pieces one file toward the H file. Mask FILE_A to prevent wrap."""
    return (board << 1) & ~FILE_A & FULL_BOARD


def west(board: uint64) -> uint64:
    """Shift all pieces one file toward the A file. Mask FILE_H to prevent wrap."""
    return (board >> 1) & ~FILE_H


def north_east(board: uint64) -> uint64:
    """Shift diagonally north-east. Mask FILE_A to prevent wrap."""
    return (board << 9) & ~FILE_A & FULL_BOARD


def north_west(board: uint64) -> uint64:
    """Shift diagonally north-west. Mask FILE_H to prevent wrap."""
    return (board << 7) & ~FILE_H & FULL_BOARD


def south_east(board: uint64) -> uint64:
    """Shift diagonally south-east. Mask FILE_A to prevent wrap."""
    return (board >> 7) & ~FILE_A


def south_west(board: uint64) -> uint64:
    """Shift diagonally south-west. Mask FILE_H to prevent wrap."""
    return (board >> 9) & ~FILE_H


# =============================================================================
# BOARD STATE
# FastPy compiles this class to a tightly-packed C++ struct (~128 bytes).
# Fits in L1/L2 cache. No pointer indirection. No garbage collector.
# =============================================================================

class BoardState:
    """
    Complete chess board state stored as twelve 64-bit bitboards.

    FastPy output:
        struct BoardState {
            uint64_t white_pawns = 0x000000000000FF00ULL;
            ...
        };
    """

    def __init__(self):
        # White pieces
        self.white_pawns:   uint64 = WHITE_PAWNS_START
        self.white_knights: uint64 = WHITE_KNIGHTS_START
        self.white_bishops: uint64 = WHITE_BISHOPS_START
        self.white_rooks:   uint64 = WHITE_ROOKS_START
        self.white_queens:  uint64 = WHITE_QUEENS_START
        self.white_king:    uint64 = WHITE_KING_START

        # Black pieces
        self.black_pawns:   uint64 = BLACK_PAWNS_START
        self.black_knights: uint64 = BLACK_KNIGHTS_START
        self.black_bishops: uint64 = BLACK_BISHOPS_START
        self.black_rooks:   uint64 = BLACK_ROOKS_START
        self.black_queens:  uint64 = BLACK_QUEENS_START
        self.black_king:    uint64 = BLACK_KING_START

        # Game state
        self.white_to_move:     bool8  = True
        self.castling_rights:   int32  = 15     # 0b1111 = KQkq
        self.en_passant_square: uint64 = 0
        self.halfmove_clock:    int32  = 0
        self.fullmove_number:   int32  = 1

    def white_pieces(self) -> uint64:
        """All white pieces as a single occupancy bitboard."""
        return (self.white_pawns   | self.white_knights |
                self.white_bishops | self.white_rooks   |
                self.white_queens  | self.white_king)

    def black_pieces(self) -> uint64:
        """All black pieces as a single occupancy bitboard."""
        return (self.black_pawns   | self.black_knights |
                self.black_bishops | self.black_rooks   |
                self.black_queens  | self.black_king)

    def all_pieces(self) -> uint64:
        """All occupied squares."""
        return self.white_pieces() | self.black_pieces()

    def empty_squares(self) -> uint64:
        """All empty squares."""
        return ~self.all_pieces() & FULL_BOARD


# =============================================================================
# MOVE GENERATORS
# Output-parameter pattern: generators fill a pre-allocated stack array
# and return the updated move count. Zero heap allocation — ever.
#
# C++ output:
#     uint64_t moves[218];   // stack-allocated, never touches the heap
#     int32_t count = 0;
#     count = generate_white_pawns(board, moves, count);
#     count = generate_white_knights(board, moves, count);
# =============================================================================

def generate_white_pawns(
    board: BoardState,
    moves: uint64[218],
    count: int32,
) -> int32:
    """
    Generate all white pawn moves into moves[count..].
    Returns the updated count.

    Covers: single push, double push, captures (east/west),
            en passant captures, and promotions.
    """
    empty: uint64 = board.empty_squares()
    pawns: uint64 = board.white_pawns
    black: uint64 = board.black_pieces()

    # ── Single pushes ─────────────────────────────────────────────────────────
    single: uint64 = north(pawns) & empty

    # Promotions: pawns reaching rank 8
    promo_single: uint64 = single & RANK_8
    temp: uint64 = promo_single
    while temp:
        to_sq: int32 = lsb(temp)
        from_sq: int32 = to_sq - 8
        moves[count] = encode_move_promo(from_sq, to_sq, PROMO_QUEEN)
        count += 1
        moves[count] = encode_move_promo(from_sq, to_sq, PROMO_KNIGHT)
        count += 1
        moves[count] = encode_move_promo(from_sq, to_sq, PROMO_BISHOP)
        count += 1
        temp = pop_lsb(temp)

    # Non-promotion single pushes
    quiet_single: uint64 = single & ~RANK_8
    temp = quiet_single
    while temp:
        to_sq = lsb(temp)
        from_sq = to_sq - 8
        moves[count] = encode_move(from_sq, to_sq)
        count += 1
        temp = pop_lsb(temp)

    # ── Double pushes (from rank 2 only) ──────────────────────────────────────
    double_push: uint64 = north(quiet_single) & empty & RANK_4
    temp = double_push
    while temp:
        to_sq = lsb(temp)
        from_sq = to_sq - 16
        moves[count] = encode_move(from_sq, to_sq)
        count += 1
        temp = pop_lsb(temp)

    # ── East captures ─────────────────────────────────────────────────────────
    cap_east: uint64 = north_east(pawns) & black

    # Promotion captures east
    promo_cap_east: uint64 = cap_east & RANK_8
    temp = promo_cap_east
    while temp:
        to_sq = lsb(temp)
        from_sq = to_sq - 9
        moves[count] = encode_move_promo(from_sq, to_sq, PROMO_QUEEN)
        count += 1
        moves[count] = encode_move_promo(from_sq, to_sq, PROMO_KNIGHT)
        count += 1
        moves[count] = encode_move_promo(from_sq, to_sq, PROMO_BISHOP)
        count += 1
        temp = pop_lsb(temp)

    # Non-promotion east captures
    quiet_cap_east: uint64 = cap_east & ~RANK_8
    temp = quiet_cap_east
    while temp:
        to_sq = lsb(temp)
        from_sq = to_sq - 9
        moves[count] = encode_move(from_sq, to_sq)
        count += 1
        temp = pop_lsb(temp)

    # ── West captures ─────────────────────────────────────────────────────────
    cap_west: uint64 = north_west(pawns) & black

    # Promotion captures west
    promo_cap_west: uint64 = cap_west & RANK_8
    temp = promo_cap_west
    while temp:
        to_sq = lsb(temp)
        from_sq = to_sq - 7
        moves[count] = encode_move_promo(from_sq, to_sq, PROMO_QUEEN)
        count += 1
        moves[count] = encode_move_promo(from_sq, to_sq, PROMO_KNIGHT)
        count += 1
        moves[count] = encode_move_promo(from_sq, to_sq, PROMO_BISHOP)
        count += 1
        temp = pop_lsb(temp)

    # Non-promotion west captures
    quiet_cap_west: uint64 = cap_west & ~RANK_8
    temp = quiet_cap_west
    while temp:
        to_sq = lsb(temp)
        from_sq = to_sq - 7
        moves[count] = encode_move(from_sq, to_sq)
        count += 1
        temp = pop_lsb(temp)

    # ── En passant ────────────────────────────────────────────────────────────
    if board.en_passant_square:
        ep: uint64 = board.en_passant_square
        ep_east: uint64 = north_east(pawns) & ep
        if ep_east:
            to_sq = lsb(ep_east)
            from_sq = to_sq - 9
            moves[count] = encode_move_flag(from_sq, to_sq, FLAG_EN_PASSANT)
            count += 1
        ep_west: uint64 = north_west(pawns) & ep
        if ep_west:
            to_sq = lsb(ep_west)
            from_sq = to_sq - 7
            moves[count] = encode_move_flag(from_sq, to_sq, FLAG_EN_PASSANT)
            count += 1

    return count


def generate_black_pawns(
    board: BoardState,
    moves: uint64[218],
    count: int32,
) -> int32:
    """
    Generate all black pawn moves into moves[count..].
    Returns the updated count.

    Mirror of generate_white_pawns with south-facing directions.
    """
    empty: uint64 = board.empty_squares()
    pawns: uint64 = board.black_pawns
    white: uint64 = board.white_pieces()

    # ── Single pushes ─────────────────────────────────────────────────────────
    single: uint64 = south(pawns) & empty

    # Promotions: pawns reaching rank 1
    promo_single: uint64 = single & RANK_1
    temp: uint64 = promo_single
    while temp:
        to_sq: int32 = lsb(temp)
        from_sq: int32 = to_sq + 8
        moves[count] = encode_move_promo(from_sq, to_sq, PROMO_QUEEN)
        count += 1
        moves[count] = encode_move_promo(from_sq, to_sq, PROMO_KNIGHT)
        count += 1
        moves[count] = encode_move_promo(from_sq, to_sq, PROMO_BISHOP)
        count += 1
        temp = pop_lsb(temp)

    # Non-promotion single pushes
    quiet_single: uint64 = single & ~RANK_1
    temp = quiet_single
    while temp:
        to_sq = lsb(temp)
        from_sq = to_sq + 8
        moves[count] = encode_move(from_sq, to_sq)
        count += 1
        temp = pop_lsb(temp)

    # ── Double pushes (from rank 7 only) ──────────────────────────────────────
    double_push: uint64 = south(quiet_single) & empty & RANK_5
    temp = double_push
    while temp:
        to_sq = lsb(temp)
        from_sq = to_sq + 16
        moves[count] = encode_move(from_sq, to_sq)
        count += 1
        temp = pop_lsb(temp)

    # ── East captures (south-east from black's perspective) ───────────────────
    cap_east: uint64 = south_east(pawns) & white

    # Promotion captures east
    promo_cap_east: uint64 = cap_east & RANK_1
    temp = promo_cap_east
    while temp:
        to_sq = lsb(temp)
        from_sq = to_sq + 7
        moves[count] = encode_move_promo(from_sq, to_sq, PROMO_QUEEN)
        count += 1
        moves[count] = encode_move_promo(from_sq, to_sq, PROMO_KNIGHT)
        count += 1
        moves[count] = encode_move_promo(from_sq, to_sq, PROMO_BISHOP)
        count += 1
        temp = pop_lsb(temp)

    # Non-promotion east captures
    quiet_cap_east: uint64 = cap_east & ~RANK_1
    temp = quiet_cap_east
    while temp:
        to_sq = lsb(temp)
        from_sq = to_sq + 7
        moves[count] = encode_move(from_sq, to_sq)
        count += 1
        temp = pop_lsb(temp)

    # ── West captures (south-west from black's perspective) ───────────────────
    cap_west: uint64 = south_west(pawns) & white

    # Promotion captures west
    promo_cap_west: uint64 = cap_west & RANK_1
    temp = promo_cap_west
    while temp:
        to_sq = lsb(temp)
        from_sq = to_sq + 9
        moves[count] = encode_move_promo(from_sq, to_sq, PROMO_QUEEN)
        count += 1
        moves[count] = encode_move_promo(from_sq, to_sq, PROMO_KNIGHT)
        count += 1
        moves[count] = encode_move_promo(from_sq, to_sq, PROMO_BISHOP)
        count += 1
        temp = pop_lsb(temp)

    # Non-promotion west captures
    quiet_cap_west: uint64 = cap_west & ~RANK_1
    temp = quiet_cap_west
    while temp:
        to_sq = lsb(temp)
        from_sq = to_sq + 9
        moves[count] = encode_move(from_sq, to_sq)
        count += 1
        temp = pop_lsb(temp)

    # ── En passant ────────────────────────────────────────────────────────────
    if board.en_passant_square:
        ep: uint64 = board.en_passant_square
        ep_east: uint64 = south_east(pawns) & ep
        if ep_east:
            to_sq = lsb(ep_east)
            from_sq = to_sq + 7
            moves[count] = encode_move_flag(from_sq, to_sq, FLAG_EN_PASSANT)
            count += 1
        ep_west: uint64 = south_west(pawns) & ep
        if ep_west:
            to_sq = lsb(ep_west)
            from_sq = to_sq + 9
            moves[count] = encode_move_flag(from_sq, to_sq, FLAG_EN_PASSANT)
            count += 1

    return count


def generate_knights(
    knights: uint64,
    friendly: uint64,
    moves: uint64[218],
    count: int32,
) -> int32:
    """
    Generate all knight moves from a bitboard of knights.
    Works for both colours — pass the appropriate friendly-pieces mask.
    Returns the updated count.
    """
    temp: uint64 = knights
    while temp:
        from_sq: int32 = lsb(temp)
        knight: uint64 = 1 << from_sq

        attacks: uint64 = (
            ((knight << 17) & ~FILE_A & FULL_BOARD) |
            ((knight << 15) & ~FILE_H & FULL_BOARD) |
            ((knight << 10) & ~FILE_A & ~FILE_B & FULL_BOARD) |
            ((knight <<  6) & ~FILE_G & ~FILE_H & FULL_BOARD) |
            ((knight >> 15) & ~FILE_A) |
            ((knight >> 17) & ~FILE_H) |
            ((knight >>  6) & ~FILE_A & ~FILE_B) |
            ((knight >> 10) & ~FILE_G & ~FILE_H)
        )
        attacks = attacks & ~friendly & FULL_BOARD

        atk: uint64 = attacks
        while atk:
            to_sq: int32 = lsb(atk)
            moves[count] = encode_move(from_sq, to_sq)
            count += 1
            atk = pop_lsb(atk)

        temp = pop_lsb(temp)

    return count


def generate_king(
    king: uint64,
    friendly: uint64,
    moves: uint64[218],
    count: int32,
) -> int32:
    """
    Generate all king moves (non-castling) from a king bitboard.
    Castling is handled separately.
    Returns the updated count.
    """
    from_sq: int32 = lsb(king)

    attacks: uint64 = (
        north(king)      |
        south(king)      |
        east(king)       |
        west(king)       |
        north_east(king) |
        north_west(king) |
        south_east(king) |
        south_west(king)
    )
    attacks = attacks & ~friendly & FULL_BOARD

    temp: uint64 = attacks
    while temp:
        to_sq: int32 = lsb(temp)
        moves[count] = encode_move(from_sq, to_sq)
        count += 1
        temp = pop_lsb(temp)

    return count


def generate_all_moves(
    board: BoardState,
    moves: uint64[218],
    count: int32,
) -> int32:
    """
    Generate all pseudo-legal moves for the side to move.
    Fills moves[0..count] and returns the total move count.

    Phase 1: pawns + knights + king only.
    Sliding pieces (bishop, rook, queen) are Phase 3.
    """
    if board.white_to_move:
        count = generate_white_pawns(board, moves, count)
        count = generate_knights(board.white_knights, board.white_pieces(), moves, count)
        count = generate_king(board.white_king, board.white_pieces(), moves, count)
    else:
        count = generate_black_pawns(board, moves, count)
        count = generate_knights(board.black_knights, board.black_pieces(), moves, count)
        count = generate_king(board.black_king, board.black_pieces(), moves, count)

    return count


# =============================================================================
# EVALUATION
# Static evaluation using material count.
# popcount() compiles to POPCNT — one clock cycle per call.
# =============================================================================

def evaluate(board: BoardState) -> int32:
    """
    Fast static evaluation — material count only.

    Returns a score in centipawns from the perspective of the side to move:
    positive means the side to move is ahead.

    FastPy compiles each popcount() call to __builtin_popcountll() [POPCNT, 1 cycle].
    """
    white_score: int32 = (
        popcount(board.white_pawns)   * VAL_PAWN   +
        popcount(board.white_knights) * VAL_KNIGHT +
        popcount(board.white_bishops) * VAL_BISHOP +
        popcount(board.white_rooks)   * VAL_ROOK   +
        popcount(board.white_queens)  * VAL_QUEEN
    )

    black_score: int32 = (
        popcount(board.black_pawns)   * VAL_PAWN   +
        popcount(board.black_knights) * VAL_KNIGHT +
        popcount(board.black_bishops) * VAL_BISHOP +
        popcount(board.black_rooks)   * VAL_ROOK   +
        popcount(board.black_queens)  * VAL_QUEEN
    )

    if board.white_to_move:
        return white_score - black_score
    else:
        return black_score - white_score


# =============================================================================
# SEARCH
# Negamax with Alpha-Beta pruning.
# Move list is a stack array — no heap allocation, no GC pause, ever.
# =============================================================================

def alpha_beta(
    board: BoardState,
    depth: int32,
    alpha: int32,
    beta: int32,
) -> int32:
    """
    Negamax alpha-beta search.

    Returns the score of the position from the perspective of the side to move.
    Positive = side to move is winning.

    At depth 0: returns static evaluation.
    Otherwise: generates all pseudo-legal moves and searches recursively.

    FastPy guarantees zero heap allocation inside this function:
    - moves[] is a C-style stack array (uint64_t moves[218])
    - No dynamic lists, no GC, no interpreter overhead
    """
    if depth == 0:
        return evaluate(board)

    moves: uint64[218]
    count: int32 = 0
    count = generate_all_moves(board, moves, count)

    if count == 0:
        return 0   # Stalemate placeholder — checkmate detection in Phase 3

    best: int32 = NEG_INF
    i: int32 = 0
    while i < count:
        # Phase 1 placeholder: make/unmake not yet implemented.
        # Phase 2 will apply moves[i] to a copy of the board, recurse, unmake.
        score: int32 = evaluate(board)

        if score > best:
            best = score

        if score > alpha:
            alpha = score

        if alpha >= beta:
            break

        i += 1

    return best


def find_best_move(
    board: BoardState,
    depth: int32,
) -> uint64:
    """
    Find the best move for the current position at the given search depth.

    Returns the best move encoded as a uint64 (bits 0-5: from, 6-11: to).
    Returns 0 if no legal moves exist.
    """
    moves: uint64[218]
    count: int32 = 0
    count = generate_all_moves(board, moves, count)

    best_move:  uint64 = 0
    best_score: int32  = NEG_INF
    alpha: int32 = NEG_INF
    beta:  int32 = INF

    i: int32 = 0
    while i < count:
        move: uint64 = moves[i]

        # Phase 2: apply move → recurse → unmake
        # Phase 1: use static evaluation as placeholder
        score: int32 = -alpha_beta(board, depth - 1, -beta, -alpha)

        if score > best_score:
            best_score = score
            best_move = move

        if score > alpha:
            alpha = score

        i += 1

    return best_move


# =============================================================================
# MAIN — UCI entry point stub
# Phase 1: compile-and-exit smoke test.
# Phase 2: full UCI loop (uci / isready / position / go / bestmove).
# =============================================================================

def main() -> int32:
    """
    Engine entry point.

    Phase 1: Initialise the starting position, run a depth-1 search, exit.
    Phase 2: Replace with a proper UCI protocol loop.

    FastPy compiles this to a standard C++ `int32_t main()`.
    """
    return 0
