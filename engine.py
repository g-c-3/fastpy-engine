# =============================================================================
# FastPy-Engine — Phase 3 Engine Source
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
# Phase 3 additions:
#   - Sliding piece move generation (bishops, rooks, queens) via ray fills
#   - Castling move generation with rights tracking
#   - Check detection via reverse attack tracing
#   - Legal move filtering (removes moves leaving king in check)
#   - Perft function for move generation correctness testing
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

# BIT_ONE is used as `BIT_ONE << sq` to produce a uint64 single-bit mask.
# Plain `1 << sq` is a 32-bit int shift in C++ — undefined for sq > 30.
BIT_ONE: Final[uint64] = 1

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
# CASTLING RIGHTS AND PATH MASKS
# Bit layout of castling_rights field:
#   bit 0 (1): White kingside   (WK)
#   bit 1 (2): White queenside  (WQ)
#   bit 2 (4): Black kingside   (BK)
#   bit 3 (8): Black queenside  (BQ)
#
# Path masks: squares that must be empty for castling.
# Safe masks: squares (king's path) that must not be attacked.
# =============================================================================

CASTLE_WK: Final[int32] = 1
CASTLE_WQ: Final[int32] = 2
CASTLE_BK: Final[int32] = 4
CASTLE_BQ: Final[int32] = 8

# f1, g1 — must be empty for white kingside
CASTLE_WK_PATH: Final[uint64] = 0x0000000000000060
# b1, c1, d1 — must be empty for white queenside
CASTLE_WQ_PATH: Final[uint64] = 0x000000000000000E
# f8, g8 — must be empty for black kingside
CASTLE_BK_PATH: Final[uint64] = 0x6000000000000000
# b8, c8, d8 — must be empty for black queenside
CASTLE_BQ_PATH: Final[uint64] = 0x0E00000000000000


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
# MOVE SCORING — MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
# Used to order moves before searching: try captures of high-value pieces first.
# =============================================================================

def piece_at_square(sq: int32, board: BoardState) -> int32:
    """
    Return the material value of the piece on the given square.
    Returns 0 if the square is empty or occupied by a king.
    Used for MVV-LVA move ordering.
    """
    sq_bb: uint64 = BIT_ONE << sq
    if (sq_bb & (board.white_pawns | board.black_pawns)) != 0:
        return VAL_PAWN
    if (sq_bb & (board.white_knights | board.black_knights)) != 0:
        return VAL_KNIGHT
    if (sq_bb & (board.white_bishops | board.black_bishops)) != 0:
        return VAL_BISHOP
    if (sq_bb & (board.white_rooks | board.black_rooks)) != 0:
        return VAL_ROOK
    if (sq_bb & (board.white_queens | board.black_queens)) != 0:
        return VAL_QUEEN
    return 0


def mvv_lva(move: uint64, board: BoardState) -> int32:
    """
    MVV-LVA capture score: victim_value * 10 - attacker_value.
    Higher scores = try first. Quiet moves score 0.
    Examples: QxP = 900*10 - 100 = 8900, PxQ = 100*10 - 900 = 100.
    """
    to_sq: int32 = move_to(move)
    from_sq: int32 = move_from(move)
    victim: int32 = piece_at_square(to_sq, board)
    if victim == 0:
        return 0
    attacker: int32 = piece_at_square(from_sq, board)
    return victim * 10 - attacker


def sort_moves(moves: uint64[218], count: int32, board: BoardState) -> None:
    """
    In-place selection sort of moves[] by MVV-LVA score (descending).
    Captures tried before quiet moves. Among captures: PxQ before QxP.
    O(n²) — acceptable for n ≤ 218.
    FastPy: moves decays to uint64_t* in C++ — in-place writes work correctly.
    """
    outer_i: int32 = 0
    while outer_i < count:
        best_j: int32 = outer_i
        best_score: int32 = mvv_lva(moves[outer_i], board)
        j: int32 = outer_i + 1
        while j < count:
            s: int32 = mvv_lva(moves[j], board)
            if s > best_score:
                best_score = s
                best_j = j
            j += 1
        tmp: uint64 = moves[outer_i]
        moves[outer_i] = moves[best_j]
        moves[best_j] = tmp
        outer_i += 1


# =============================================================================
# BITBOARD UTILITIES
# These compile to single-clock-cycle CPU hardware instructions via FastPy.
# =============================================================================

def popcount(board: uint64) -> int32:
    """
    Count set bits on a bitboard.
    FastPy: bin(board).count("1") → __builtin_popcountll(board)  [POPCNT, 1 cycle]
    """
    return bin(board).count("1")


def lsb(board: uint64) -> int32:
    """
    Index of the least significant set bit.
    FastPy: (board & -board).bit_length() - 1 → __builtin_ctzll(board)  [TZCNT, 1 cycle]
    Returns -1 for an empty board.
    """
    if board == 0:
        return -1
    return (board & -board).bit_length() - 1


def pop_lsb(board: uint64) -> uint64:
    """
    Remove the least significant bit.
    FastPy: board & (board - 1) → BLSR instruction  [BMI1, 1 cycle]
    """
    return board & (board - 1)


# ── Directional one-square shifts ─────────────────────────────────────────────

def north(board: uint64) -> uint64:
    """Shift all pieces one rank toward rank 8."""
    return (board << 8) & FULL_BOARD


def south(board: uint64) -> uint64:
    """Shift all pieces one rank toward rank 1."""
    return board >> 8


def east(board: uint64) -> uint64:
    """Shift one file toward H. Mask FILE_A to prevent wrap."""
    return (board << 1) & ~FILE_A & FULL_BOARD


def west(board: uint64) -> uint64:
    """Shift one file toward A. Mask FILE_H to prevent wrap."""
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
# RAY GENERATORS
# Each function extends a single-square bitboard along one ray direction
# until it hits an occupied square (inclusive — captures are included).
#
# FastPy compiles the inner loop to tightly-packed shifts with early exit.
# All ray attacks together cover sliding piece moves for bishops, rooks,
# and queens.
# =============================================================================

def ray_north(sq_bb: uint64, occupied: uint64) -> uint64:
    """North ray from sq_bb until (and including) the first occupied square."""
    attacks: uint64 = 0
    ray: uint64 = north(sq_bb)
    while ray:
        attacks = attacks | ray
        if ray & occupied:
            break
        ray = north(ray)
    return attacks


def ray_south(sq_bb: uint64, occupied: uint64) -> uint64:
    """South ray from sq_bb until (and including) the first occupied square."""
    attacks: uint64 = 0
    ray: uint64 = south(sq_bb)
    while ray:
        attacks = attacks | ray
        if ray & occupied:
            break
        ray = south(ray)
    return attacks


def ray_east(sq_bb: uint64, occupied: uint64) -> uint64:
    """East ray from sq_bb until (and including) the first occupied square."""
    attacks: uint64 = 0
    ray: uint64 = east(sq_bb)
    while ray:
        attacks = attacks | ray
        if ray & occupied:
            break
        ray = east(ray)
    return attacks


def ray_west(sq_bb: uint64, occupied: uint64) -> uint64:
    """West ray from sq_bb until (and including) the first occupied square."""
    attacks: uint64 = 0
    ray: uint64 = west(sq_bb)
    while ray:
        attacks = attacks | ray
        if ray & occupied:
            break
        ray = west(ray)
    return attacks


def ray_north_east(sq_bb: uint64, occupied: uint64) -> uint64:
    """North-east diagonal ray until (and including) first occupied square."""
    attacks: uint64 = 0
    ray: uint64 = north_east(sq_bb)
    while ray:
        attacks = attacks | ray
        if ray & occupied:
            break
        ray = north_east(ray)
    return attacks


def ray_north_west(sq_bb: uint64, occupied: uint64) -> uint64:
    """North-west diagonal ray until (and including) first occupied square."""
    attacks: uint64 = 0
    ray: uint64 = north_west(sq_bb)
    while ray:
        attacks = attacks | ray
        if ray & occupied:
            break
        ray = north_west(ray)
    return attacks


def ray_south_east(sq_bb: uint64, occupied: uint64) -> uint64:
    """South-east diagonal ray until (and including) first occupied square."""
    attacks: uint64 = 0
    ray: uint64 = south_east(sq_bb)
    while ray:
        attacks = attacks | ray
        if ray & occupied:
            break
        ray = south_east(ray)
    return attacks


def ray_south_west(sq_bb: uint64, occupied: uint64) -> uint64:
    """South-west diagonal ray until (and including) first occupied square."""
    attacks: uint64 = 0
    ray: uint64 = south_west(sq_bb)
    while ray:
        attacks = attacks | ray
        if ray & occupied:
            break
        ray = south_west(ray)
    return attacks


# =============================================================================
# ATTACK MASKS
# Pre-compute all squares attacked by a piece on a given square.
# Shared between move generation and check detection.
# =============================================================================

def knight_attack_mask(sq_bb: uint64) -> uint64:
    """Bitboard of all squares a knight on sq_bb attacks."""
    return (
        ((sq_bb << 17) & ~FILE_A & FULL_BOARD) |
        ((sq_bb << 15) & ~FILE_H & FULL_BOARD) |
        ((sq_bb << 10) & ~FILE_A & ~FILE_B & FULL_BOARD) |
        ((sq_bb <<  6) & ~FILE_G & ~FILE_H & FULL_BOARD) |
        ((sq_bb >> 15) & ~FILE_A) |
        ((sq_bb >> 17) & ~FILE_H) |
        ((sq_bb >>  6) & ~FILE_A & ~FILE_B) |
        ((sq_bb >> 10) & ~FILE_G & ~FILE_H)
    )


def king_attack_mask(sq_bb: uint64) -> uint64:
    """Bitboard of all squares a king on sq_bb attacks (8 surrounding squares)."""
    return (
        north(sq_bb)      |
        south(sq_bb)      |
        east(sq_bb)       |
        west(sq_bb)       |
        north_east(sq_bb) |
        north_west(sq_bb) |
        south_east(sq_bb) |
        south_west(sq_bb)
    )


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
# MAKE MOVE
# Apply a packed move word to a board copy and return the new position.
#
# Value semantics: BoardState is passed by value (copied on entry in C++).
# The caller's original board is untouched — no explicit unmake needed.
#
# Handles: quiet moves, captures, en passant, double-pawn push (sets ep
# square), promotions (queen/knight/bishop), and castling (rook movement
# + rights update).
# =============================================================================

def make_move(board: BoardState, move: uint64) -> BoardState:
    """
    Return a new BoardState with `move` applied.

    The caller keeps its original board unchanged. Recursive search:
        new_board: BoardState = make_move(board, moves[i])
        score: int32 = -alpha_beta(new_board, depth - 1, -beta, -alpha)
    """
    from_sq: int32 = move_from(move)
    to_sq:   int32 = move_to(move)
    promo:   int32 = move_promo(move)
    flag:    int32 = move_flag(move)

    from_bb: uint64 = BIT_ONE << from_sq
    to_bb:   uint64 = BIT_ONE << to_sq

    if board.white_to_move:
        # ── Clear captured black piece from destination ────────────────────────
        board.black_pawns   = board.black_pawns   & ~to_bb
        board.black_knights = board.black_knights & ~to_bb
        board.black_bishops = board.black_bishops & ~to_bb
        board.black_rooks   = board.black_rooks   & ~to_bb
        board.black_queens  = board.black_queens  & ~to_bb
        board.black_king    = board.black_king    & ~to_bb

        # ── En passant: remove captured black pawn one rank south of to_sq ────
        if flag == FLAG_EN_PASSANT:
            ep_capture: uint64 = to_bb >> 8
            board.black_pawns = board.black_pawns & ~ep_capture

        # ── Move the white piece ───────────────────────────────────────────────
        if board.white_pawns & from_bb:
            board.white_pawns = board.white_pawns & ~from_bb
            if promo == PROMO_NONE:
                board.white_pawns = board.white_pawns | to_bb
                if to_sq - from_sq == 16:
                    board.en_passant_square = BIT_ONE << (from_sq + 8)
                else:
                    board.en_passant_square = 0
            elif promo == PROMO_QUEEN:
                board.white_queens  = board.white_queens  | to_bb
                board.en_passant_square = 0
            elif promo == PROMO_KNIGHT:
                board.white_knights = board.white_knights | to_bb
                board.en_passant_square = 0
            else:
                board.white_bishops = board.white_bishops | to_bb
                board.en_passant_square = 0
        elif board.white_knights & from_bb:
            board.white_knights = (board.white_knights & ~from_bb) | to_bb
            board.en_passant_square = 0
        elif board.white_bishops & from_bb:
            board.white_bishops = (board.white_bishops & ~from_bb) | to_bb
            board.en_passant_square = 0
        elif board.white_rooks & from_bb:
            board.white_rooks = (board.white_rooks & ~from_bb) | to_bb
            board.en_passant_square = 0
        elif board.white_queens & from_bb:
            board.white_queens = (board.white_queens & ~from_bb) | to_bb
            board.en_passant_square = 0
        elif board.white_king & from_bb:
            board.white_king = (board.white_king & ~from_bb) | to_bb
            board.en_passant_square = 0
            # ── Castling: also move the rook ──────────────────────────────────
            if flag == FLAG_CASTLING:
                if to_sq == 6:   # Kingside: rook h1 (sq 7) → f1 (sq 5)
                    board.white_rooks = (board.white_rooks & ~(BIT_ONE << 7)) | (BIT_ONE << 5)
                else:            # Queenside: rook a1 (sq 0) → d1 (sq 3)
                    board.white_rooks = (board.white_rooks & ~BIT_ONE) | (BIT_ONE << 3)
        else:
            board.en_passant_square = 0

    else:
        # ── Clear captured white piece from destination ────────────────────────
        board.white_pawns   = board.white_pawns   & ~to_bb
        board.white_knights = board.white_knights & ~to_bb
        board.white_bishops = board.white_bishops & ~to_bb
        board.white_rooks   = board.white_rooks   & ~to_bb
        board.white_queens  = board.white_queens  & ~to_bb
        board.white_king    = board.white_king    & ~to_bb

        # ── En passant: remove captured white pawn one rank north of to_sq ────
        if flag == FLAG_EN_PASSANT:
            ep_capture = to_bb << 8
            board.white_pawns = board.white_pawns & ~ep_capture

        # ── Move the black piece ───────────────────────────────────────────────
        if board.black_pawns & from_bb:
            board.black_pawns = board.black_pawns & ~from_bb
            if promo == PROMO_NONE:
                board.black_pawns = board.black_pawns | to_bb
                if from_sq - to_sq == 16:
                    board.en_passant_square = BIT_ONE << (from_sq - 8)
                else:
                    board.en_passant_square = 0
            elif promo == PROMO_QUEEN:
                board.black_queens  = board.black_queens  | to_bb
                board.en_passant_square = 0
            elif promo == PROMO_KNIGHT:
                board.black_knights = board.black_knights | to_bb
                board.en_passant_square = 0
            else:
                board.black_bishops = board.black_bishops | to_bb
                board.en_passant_square = 0
        elif board.black_knights & from_bb:
            board.black_knights = (board.black_knights & ~from_bb) | to_bb
            board.en_passant_square = 0
        elif board.black_bishops & from_bb:
            board.black_bishops = (board.black_bishops & ~from_bb) | to_bb
            board.en_passant_square = 0
        elif board.black_rooks & from_bb:
            board.black_rooks = (board.black_rooks & ~from_bb) | to_bb
            board.en_passant_square = 0
        elif board.black_queens & from_bb:
            board.black_queens = (board.black_queens & ~from_bb) | to_bb
            board.en_passant_square = 0
        elif board.black_king & from_bb:
            board.black_king = (board.black_king & ~from_bb) | to_bb
            board.en_passant_square = 0
            # ── Castling: also move the rook ──────────────────────────────────
            if flag == FLAG_CASTLING:
                if to_sq == 62:  # Kingside: rook h8 (sq 63) → f8 (sq 61)
                    board.black_rooks = (board.black_rooks & ~(BIT_ONE << 63)) | (BIT_ONE << 61)
                else:            # Queenside: rook a8 (sq 56) → d8 (sq 59)
                    board.black_rooks = (board.black_rooks & ~(BIT_ONE << 56)) | (BIT_ONE << 59)
        else:
            board.en_passant_square = 0

    # ── Update castling rights ─────────────────────────────────────────────────
    # King moves clear both rights for that side.
    # Rook moves or captures on rook start squares clear the relevant right.
    # Using positive masks avoids signed-integer issues with ~ in C++.
    if from_sq == 4:    # e1 — white king moved
        board.castling_rights = board.castling_rights & 12   # keep BK,BQ
    if from_sq == 60:   # e8 — black king moved
        board.castling_rights = board.castling_rights & 3    # keep WK,WQ
    if from_sq == 7 or to_sq == 7:    # h1 rook moved or captured
        board.castling_rights = board.castling_rights & 14   # clear WK
    if from_sq == 0 or to_sq == 0:    # a1 rook moved or captured
        board.castling_rights = board.castling_rights & 13   # clear WQ
    if from_sq == 63 or to_sq == 63:  # h8 rook moved or captured
        board.castling_rights = board.castling_rights & 11   # clear BK
    if from_sq == 56 or to_sq == 56:  # a8 rook moved or captured
        board.castling_rights = board.castling_rights & 7    # clear BQ

    board.white_to_move = not board.white_to_move
    board.halfmove_clock = board.halfmove_clock + 1
    return board


# =============================================================================
# MOVE GENERATORS
# Output-parameter pattern: fill a pre-allocated stack array, return count.
# Zero heap allocation — ever.
# =============================================================================

def generate_white_pawns(
    board: BoardState,
    moves: uint64[218],
    count: int32,
) -> int32:
    """
    Generate all white pawn moves into moves[count..].
    Covers: single push, double push, captures, en passant, promotions.
    """
    empty: uint64 = board.empty_squares()
    pawns: uint64 = board.white_pawns
    black: uint64 = board.black_pieces()

    # ── Single pushes ─────────────────────────────────────────────────────────
    single: uint64 = north(pawns) & empty

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
    Mirror of generate_white_pawns with south-facing directions.
    """
    empty: uint64 = board.empty_squares()
    pawns: uint64 = board.black_pawns
    white: uint64 = board.white_pieces()

    # ── Single pushes ─────────────────────────────────────────────────────────
    single: uint64 = south(pawns) & empty

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

    # ── East captures (south-east from black's view) ──────────────────────────
    cap_east: uint64 = south_east(pawns) & white
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

    quiet_cap_east: uint64 = cap_east & ~RANK_1
    temp = quiet_cap_east
    while temp:
        to_sq = lsb(temp)
        from_sq = to_sq + 7
        moves[count] = encode_move(from_sq, to_sq)
        count += 1
        temp = pop_lsb(temp)

    # ── West captures (south-west from black's view) ──────────────────────────
    cap_west: uint64 = south_west(pawns) & white
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
    Generate all knight moves. Works for both colours.
    Pass the appropriate friendly-pieces mask to exclude self-captures.
    """
    temp: uint64 = knights
    while temp:
        from_sq: int32 = lsb(temp)
        sq_bb: uint64 = BIT_ONE << from_sq
        attacks: uint64 = knight_attack_mask(sq_bb) & ~friendly & FULL_BOARD
        atk: uint64 = attacks
        while atk:
            to_sq: int32 = lsb(atk)
            moves[count] = encode_move(from_sq, to_sq)
            count += 1
            atk = pop_lsb(atk)
        temp = pop_lsb(temp)
    return count


def generate_bishops(
    bishops: uint64,
    friendly: uint64,
    occupied: uint64,
    moves: uint64[218],
    count: int32,
) -> int32:
    """
    Generate all bishop moves using diagonal ray fills.
    FastPy compiles the inner ray loops to tightly-packed shift sequences.
    """
    temp: uint64 = bishops
    while temp:
        from_sq: int32 = lsb(temp)
        sq_bb: uint64 = BIT_ONE << from_sq
        attacks: uint64 = (
            ray_north_east(sq_bb, occupied) |
            ray_north_west(sq_bb, occupied) |
            ray_south_east(sq_bb, occupied) |
            ray_south_west(sq_bb, occupied)
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


def generate_rooks(
    rooks: uint64,
    friendly: uint64,
    occupied: uint64,
    moves: uint64[218],
    count: int32,
) -> int32:
    """
    Generate all rook moves using horizontal and vertical ray fills.
    """
    temp: uint64 = rooks
    while temp:
        from_sq: int32 = lsb(temp)
        sq_bb: uint64 = BIT_ONE << from_sq
        attacks: uint64 = (
            ray_north(sq_bb, occupied) |
            ray_south(sq_bb, occupied) |
            ray_east(sq_bb, occupied)  |
            ray_west(sq_bb, occupied)
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


def generate_queens(
    queens: uint64,
    friendly: uint64,
    occupied: uint64,
    moves: uint64[218],
    count: int32,
) -> int32:
    """
    Generate all queen moves: union of bishop rays and rook rays.
    """
    temp: uint64 = queens
    while temp:
        from_sq: int32 = lsb(temp)
        sq_bb: uint64 = BIT_ONE << from_sq
        attacks: uint64 = (
            ray_north_east(sq_bb, occupied) |
            ray_north_west(sq_bb, occupied) |
            ray_south_east(sq_bb, occupied) |
            ray_south_west(sq_bb, occupied) |
            ray_north(sq_bb, occupied)      |
            ray_south(sq_bb, occupied)      |
            ray_east(sq_bb, occupied)       |
            ray_west(sq_bb, occupied)
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
    Generate all king moves (non-castling).
    Castling is handled separately by generate_castling().
    """
    from_sq: int32 = lsb(king)
    attacks: uint64 = king_attack_mask(king) & ~friendly & FULL_BOARD
    temp: uint64 = attacks
    while temp:
        to_sq: int32 = lsb(temp)
        moves[count] = encode_move(from_sq, to_sq)
        count += 1
        temp = pop_lsb(temp)
    return count


# =============================================================================
# CHECK DETECTION
# Determine if a given square is attacked by pieces of a specific colour,
# using reverse attack tracing (apply the attacker's pattern from the target
# square and test if it overlaps with any of that piece type).
# =============================================================================

def is_sq_attacked(sq: int32, board: BoardState, by_black: bool8) -> bool8:
    """
    Return True if square `sq` is attacked by the specified side.

    `by_black=True`  → is sq attacked by black pieces?
    `by_black=False` → is sq attacked by white pieces?

    Uses reverse attack tracing: apply each piece's attack pattern from sq
    and check if it overlaps with any piece of that type.

    Called by is_in_check() and generate_castling() for legality checks.
    """
    sq_bb: uint64 = BIT_ONE << sq
    occupied: uint64 = board.all_pieces()

    # Compute all attack sets from sq once (used by both branches)
    natk: uint64 = knight_attack_mask(sq_bb)
    diag: uint64 = (
        ray_north_east(sq_bb, occupied) |
        ray_north_west(sq_bb, occupied) |
        ray_south_east(sq_bb, occupied) |
        ray_south_west(sq_bb, occupied)
    )
    straight: uint64 = (
        ray_north(sq_bb, occupied) |
        ray_south(sq_bb, occupied) |
        ray_east(sq_bb, occupied)  |
        ray_west(sq_bb, occupied)
    )
    # Black pawns attack downward → from sq's perspective, look north for black attackers
    north_pawn: uint64 = north_east(sq_bb) | north_west(sq_bb)
    # White pawns attack upward → from sq's perspective, look south for white attackers
    south_pawn: uint64 = south_east(sq_bb) | south_west(sq_bb)
    katk: uint64 = king_attack_mask(sq_bb)

    if by_black:
        if natk & board.black_knights:
            return True
        if diag & (board.black_bishops | board.black_queens):
            return True
        if straight & (board.black_rooks | board.black_queens):
            return True
        if north_pawn & board.black_pawns:
            return True
        if katk & board.black_king:
            return True
    else:
        if natk & board.white_knights:
            return True
        if diag & (board.white_bishops | board.white_queens):
            return True
        if straight & (board.white_rooks | board.white_queens):
            return True
        if south_pawn & board.white_pawns:
            return True
        if katk & board.white_king:
            return True

    return False


def is_in_check(board: BoardState) -> bool8:
    """
    Return True if the side that JUST MOVED has left their king in check.

    After make_move(), white_to_move is already flipped. So:
    - white_to_move=True  now → black just moved → check if black king attacked by white
    - white_to_move=False now → white just moved → check if white king attacked by black

    Used by generate_legal_moves() to filter pseudo-legal moves.
    """
    white_king_sq: int32 = lsb(board.white_king)
    black_king_sq: int32 = lsb(board.black_king)
    if board.white_to_move:
        # Black just moved — is black's king attacked by white?
        return is_sq_attacked(black_king_sq, board, False)
    else:
        # White just moved — is white's king attacked by black?
        return is_sq_attacked(white_king_sq, board, True)


# =============================================================================
# CASTLING MOVE GENERATION
# Checked separately from normal king moves:
#   1. Castling right must be set
#   2. Squares between king and rook must be empty
#   3. King's path (start, transit, landing) must not be attacked
# =============================================================================

def generate_castling(
    board: BoardState,
    moves: uint64[218],
    count: int32,
) -> int32:
    """
    Generate legal castling moves for the side to move.

    Castling is only generated when:
    - The relevant castling right bit is set in board.castling_rights
    - The path between king and rook is clear (no pieces)
    - The king does not start on, pass through, or land on an attacked square

    Returns the updated move count.
    """
    occupied: uint64 = board.all_pieces()

    if board.white_to_move:
        # ── White kingside: e1→g1 ─────────────────────────────────────────────
        if board.castling_rights & CASTLE_WK:
            if not (occupied & CASTLE_WK_PATH):
                if not is_sq_attacked(4, board, True):
                    if not is_sq_attacked(5, board, True):
                        if not is_sq_attacked(6, board, True):
                            moves[count] = encode_move_flag(4, 6, FLAG_CASTLING)
                            count += 1

        # ── White queenside: e1→c1 ────────────────────────────────────────────
        if board.castling_rights & CASTLE_WQ:
            if not (occupied & CASTLE_WQ_PATH):
                if not is_sq_attacked(4, board, True):
                    if not is_sq_attacked(3, board, True):
                        if not is_sq_attacked(2, board, True):
                            moves[count] = encode_move_flag(4, 2, FLAG_CASTLING)
                            count += 1
    else:
        # ── Black kingside: e8→g8 ─────────────────────────────────────────────
        if board.castling_rights & CASTLE_BK:
            if not (occupied & CASTLE_BK_PATH):
                if not is_sq_attacked(60, board, False):
                    if not is_sq_attacked(61, board, False):
                        if not is_sq_attacked(62, board, False):
                            moves[count] = encode_move_flag(60, 62, FLAG_CASTLING)
                            count += 1

        # ── Black queenside: e8→c8 ────────────────────────────────────────────
        if board.castling_rights & CASTLE_BQ:
            if not (occupied & CASTLE_BQ_PATH):
                if not is_sq_attacked(60, board, False):
                    if not is_sq_attacked(59, board, False):
                        if not is_sq_attacked(58, board, False):
                            moves[count] = encode_move_flag(60, 58, FLAG_CASTLING)
                            count += 1

    return count


# =============================================================================
# PSEUDO-LEGAL MOVE GENERATION
# Generates all moves that are structurally valid but may leave the king
# in check. Used as the first stage of legal move generation.
# =============================================================================

def generate_all_moves(
    board: BoardState,
    moves: uint64[218],
    count: int32,
) -> int32:
    """
    Generate all pseudo-legal moves for the side to move.

    Includes: pawns, knights, bishops, rooks, queens, king, castling.
    Does NOT filter moves that leave the king in check — use
    generate_legal_moves() for that.

    Phase 3: all piece types + castling now included.
    """
    occupied: uint64 = board.all_pieces()

    if board.white_to_move:
        wp: uint64 = board.white_pieces()
        count = generate_white_pawns(board, moves, count)
        count = generate_knights(board.white_knights, wp, moves, count)
        count = generate_bishops(board.white_bishops, wp, occupied, moves, count)
        count = generate_rooks(board.white_rooks,   wp, occupied, moves, count)
        count = generate_queens(board.white_queens,  wp, occupied, moves, count)
        count = generate_king(board.white_king, wp, moves, count)
    else:
        bp: uint64 = board.black_pieces()
        count = generate_black_pawns(board, moves, count)
        count = generate_knights(board.black_knights, bp, moves, count)
        count = generate_bishops(board.black_bishops, bp, occupied, moves, count)
        count = generate_rooks(board.black_rooks,   bp, occupied, moves, count)
        count = generate_queens(board.black_queens,  bp, occupied, moves, count)
        count = generate_king(board.black_king, bp, moves, count)

    count = generate_castling(board, moves, count)
    return count


# =============================================================================
# LEGAL MOVE GENERATION
# Filters pseudo-legal moves by verifying the king is not in check after
# each move. This is the correct set of moves for search and perft.
# =============================================================================

def generate_legal_moves(
    board: BoardState,
    moves: uint64[218],
    count: int32,
) -> int32:
    """
    Generate all strictly legal moves for the side to move.

    Generates pseudo-legal moves, then removes any that leave the moving
    side's king in check. Returns the updated count.

    This is the function used by perft() and alpha_beta() for correctness.
    """
    pseudo: uint64[218]
    pcount: int32 = 0
    pcount = generate_all_moves(board, pseudo, pcount)

    i: int32 = 0
    while i < pcount:
        new_board: BoardState = make_move(board, pseudo[i])
        if not is_in_check(new_board):
            moves[count] = pseudo[i]
            count += 1
        i += 1

    return count
# =============================================================================
# PIECE-SQUARE TABLES (PST)
# Positional bonuses in centipawns for each piece type at each square.
#
# Implemented as rank/file functions — no lookup arrays, pure arithmetic.
# FastPy compiles these to straight-line C++ with branch prediction hints.
#
# Convention:
#   rank = sq >> 3    (0 = rank 1 ... 7 = rank 8)
#   file = sq & 7     (0 = file a ... 7 = file h)
#   is_white: True uses rank as-is; False mirrors rank (7 - rank).
#
# Values derived from Tomasz Michniewski's Simplified Evaluation Function
# (public domain), approximated as separable rank+file components.
# =============================================================================

def pst_pawn_sq(rank: int32, file: int32) -> int32:
    """
    Pawn PST bonus for a square given rank and file (white perspective).
    rank 0 = rank 1 (impossible for pawns in play), rank 6 = rank 7 (promotion zone).
    """
    rank_b: int32 = 0
    if rank == 6:
        rank_b = 50
    elif rank == 5:
        rank_b = 30
    elif rank == 4:
        rank_b = 20
    elif rank == 3:
        rank_b = 15
    elif rank == 2:
        rank_b = 5

    file_b: int32 = 0
    if file == 3 or file == 4:
        file_b = 10
    elif file == 2 or file == 5:
        file_b = 5
    elif file == 0 or file == 7:
        file_b = -10

    return rank_b + file_b


def pst_knight_sq(rank: int32, file: int32) -> int32:
    """
    Knight PST bonus. Knights are strongest in the centre, weakest on edges.
    Separable rank+file bonuses approximate the standard knight table.
    """
    rank_b: int32 = 0
    if rank == 0 or rank == 7:
        rank_b = -20
    elif rank == 1 or rank == 6:
        rank_b = -10
    elif rank == 3 or rank == 4:
        rank_b = 10

    file_b: int32 = 0
    if file == 0 or file == 7:
        file_b = -20
    elif file == 1 or file == 6:
        file_b = -10
    elif file == 3 or file == 4:
        file_b = 10

    return rank_b + file_b


def pst_bishop_sq(rank: int32, file: int32) -> int32:
    """
    Bishop PST bonus. Bishops reward central placement and long diagonals.
    Penalise edge squares where the bishop controls fewer squares.
    """
    center_b: int32 = 0
    if (rank == 3 or rank == 4) and (file == 3 or file == 4):
        center_b = 15
    elif (rank == 2 or rank == 5) and (file == 2 or file == 5):
        center_b = 10
    elif (rank == 3 or rank == 4) and (file == 2 or file == 5):
        center_b = 5
    elif (rank == 2 or rank == 5) and (file == 3 or file == 4):
        center_b = 5

    edge_b: int32 = 0
    if rank == 0 or rank == 7 or file == 0 or file == 7:
        edge_b = -10

    # Main diagonal bonus (a1-h8: rank==file; a8-h1: rank+file==7)
    diag_b: int32 = 0
    if rank == file or rank + file == 7:
        diag_b = 5

    return center_b + edge_b + diag_b


def pst_rook_sq(rank: int32, file: int32) -> int32:
    """
    Rook PST bonus. 7th rank dominance is the biggest positional asset;
    central files are slightly preferable to edge files.
    """
    rank_b: int32 = 0
    if rank == 6:
        rank_b = 20    # 7th rank — controls enemy pawn rank
    elif rank == 7:
        rank_b = 10    # 8th rank — behind promotion zone

    file_b: int32 = 0
    if file == 3 or file == 4:
        file_b = 5     # Central open files

    return rank_b + file_b


def pst_king_sq(rank: int32, file: int32, is_white: bool8) -> int32:
    """
    King PST bonus (middlegame). Strongly rewards castled positions,
    penalises kings stuck in the centre.

    rank is received pre-mirrored (i.e. always from the piece's own perspective:
    rank 0 = back rank, rank 7 = opponent's back rank). pst_sum performs the
    mirroring before calling this function.
    """
    if is_white:
        if rank != 0:
            return 0    # King should stay on back rank in MG
        if file == 6 or file == 7:
            return 30   # Kingside castle (g1/h1)
        if file == 1 or file == 2:
            return 20   # Queenside castle (b1/c1)
        if file == 3 or file == 4:
            return -30  # Exposed on d1/e1
        return 0
    else:
        # rank is already mirrored in pst_sum: black's rank 7 → 0
        if rank != 0:
            return 0    # Black back rank (mirrored rank 0 = original rank 7)
        if file == 6 or file == 7:
            return 30   # g8/h8
        if file == 1 or file == 2:
            return 20   # b8/c8
        if file == 3 or file == 4:
            return -30  # d8/e8
        return 0


def pst_sum(pieces: uint64, is_white: bool8, ptype: int32) -> int32:
    """
    Sum PST bonuses for all set bits in `pieces`.
    ptype: 0=pawn, 1=knight, 2=bishop, 3=rook, 4=king

    Iterates with lsb/pop_lsb — compiles to TZCNT/BLSR per iteration.
    FastPy: temp is a C-style stack variable — zero allocation.
    """
    score: int32 = 0
    temp: uint64 = pieces
    while temp:
        sq: int32 = lsb(temp)
        rank: int32 = sq >> 3
        file: int32 = sq & 7
        if not is_white:
            rank = 7 - rank

        bonus: int32 = 0
        if ptype == 0:
            bonus = pst_pawn_sq(rank, file)
        elif ptype == 1:
            bonus = pst_knight_sq(rank, file)
        elif ptype == 2:
            bonus = pst_bishop_sq(rank, file)
        elif ptype == 3:
            bonus = pst_rook_sq(rank, file)
        elif ptype == 4:
            bonus = pst_king_sq(rank, file, is_white)

        score = score + bonus
        temp = pop_lsb(temp)

    return score

# =============================================================================
# EVALUATION
# Static evaluation using material count.
# popcount() compiles to POPCNT — one clock cycle per call.
# =============================================================================

def evaluate(board: BoardState) -> int32:
    """
    Fast static evaluation — material count only (Phase 1/2).

    Returns centipawns from the perspective of the side to move.
    Positive = side to move is winning.

    FastPy compiles each popcount() call to __builtin_popcountll()  [POPCNT, 1 cycle].
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
# QUIESCENCE SEARCH
# At leaf nodes, keep searching captures until the position is "quiet".
# Prevents the horizon effect — engine won't hang pieces at the search edge.
# =============================================================================

def generate_captures(board: BoardState, moves: uint64[218], count: int32) -> int32:
    """
    Generate legal capture moves only (for quiescence search).
    Runs generate_all_moves then filters for moves landing on enemy squares
    (or en passant captures), verifying legality via is_in_check.

    FastPy: moves decays to uint64_t* — pointer fill pattern, zero allocation.
    """
    all_moves: uint64[218]
    all_count: int32 = 0
    all_count = generate_all_moves(board, all_moves, all_count)

    enemy: uint64 = 0
    if board.white_to_move:
        enemy = board.black_pieces()
    else:
        enemy = board.white_pieces()

    i: int32 = 0
    while i < all_count:
        move: uint64 = all_moves[i]
        to_sq: int32 = move_to(move)
        to_bb: uint64 = BIT_ONE << to_sq
        is_ep: bool8 = move_flag(move) == FLAG_EN_PASSANT
        if ((to_bb & enemy) != 0) or is_ep:
            new_board: BoardState = make_move(board, move)
            if not is_in_check(new_board):
                moves[count] = move
                count = count + 1
        i += 1

    return count


def quiescence(board: BoardState, alpha: int32, beta: int32) -> int32:
    """
    Quiescence search — extend the search at leaf nodes until the position
    is quiet (no captures left). This prevents the horizon effect where the
    engine evaluates a position just before losing a piece.

    Stand-pat: if the static evaluation already beats beta, prune immediately.
    Only captures and en passant are searched (no quiet moves).

    FastPy: captures[] is a C-style stack array — zero heap allocation.
    """
    stand_pat: int32 = evaluate(board)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    captures: uint64[218]
    cap_count: int32 = 0
    cap_count = generate_captures(board, captures, cap_count)

    i: int32 = 0
    while i < cap_count:
        new_board: BoardState = make_move(board, captures[i])
        score: int32 = -quiescence(new_board, -beta, -alpha)
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
        i += 1

    return alpha


# =============================================================================
# SEARCH
# Negamax with Alpha-Beta pruning.
# Move list is a stack array — no heap allocation, no GC pause, ever.
#
# Phase 4: quiescence search at depth 0, MVV-LVA move ordering.
# =============================================================================

def alpha_beta(
    board: BoardState,
    depth: int32,
    alpha: int32,
    beta: int32,
) -> int32:
    """
    Negamax alpha-beta search with quiescence extension and MVV-LVA ordering.

    Phase 4 improvements:
    - depth == 0: enters quiescence search instead of static eval
    - Moves are sorted by MVV-LVA score before the search loop
      so that captures of high-value pieces are tried first

    FastPy guarantees zero heap allocation:
    - moves[] is a C-style stack array  (uint64_t moves[218])
    - quiescence captures[] is also stack-allocated
    - No dynamic lists, no GC, no interpreter overhead
    """
    if depth == 0:
        return quiescence(board, alpha, beta)

    moves: uint64[218]
    count: int32 = 0
    count = generate_legal_moves(board, moves, count)

    if count == 0:
        # No legal moves. Stalemate or checkmate — return 0 (draw placeholder).
        return 0

    sort_moves(moves, count, board)

    best: int32 = NEG_INF
    i: int32 = 0
    while i < count:
        new_board: BoardState = make_move(board, moves[i])
        score: int32 = -alpha_beta(new_board, depth - 1, -beta, -alpha)

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
    Returns the best move encoded as a uint64. Returns 0 if no legal moves.
    """
    moves: uint64[218]
    count: int32 = 0
    count = generate_legal_moves(board, moves, count)

    best_move:  uint64 = 0
    best_score: int32  = NEG_INF
    alpha: int32 = NEG_INF
    beta:  int32 = INF

    i: int32 = 0
    while i < count:
        move: uint64 = moves[i]
        new_board: BoardState = make_move(board, move)
        score: int32 = -alpha_beta(new_board, depth - 1, -beta, -alpha)

        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score

        i += 1

    return best_move


# =============================================================================
# PERFT — Move Generation Correctness Benchmark
# Count leaf nodes at a given depth to verify move generation is correct.
#
# Correctness targets from starting position:
#   perft(1) = 20
#   perft(2) = 400
#   perft(3) = 8,902
#   perft(4) = 197,281
#   perft(5) = 4,865,609   ← primary correctness benchmark
#
# Note: perft uses int32 which is correct for depth ≤ 6 (max 119,060,324).
# Depth ≥ 7 requires int64 (3,195,901,860 exceeds INT32_MAX).
# =============================================================================

def perft(board: BoardState, depth: int32) -> int32:
    """
    Count all legal leaf nodes at `depth` plies from `board`.

    Used to verify move generation correctness before benchmarking NPS.
    A wrong perft count means a bug in move generation or make_move.
    """
    moves: uint64[218]
    count: int32 = 0
    count = generate_legal_moves(board, moves, count)

    if depth == 1:
        return count

    nodes: int32 = 0
    i: int32 = 0
    while i < count:
        new_board: BoardState = make_move(board, moves[i])
        nodes = nodes + perft(new_board, depth - 1)
        i += 1

    return nodes


# =============================================================================
# MAIN — FastPy-compiled entry point (stub)
# =============================================================================

def main() -> int32:
    """
    Engine entry point (FastPy-compiled stub). Returns 0.
    UCI loop runs in Python mode only via run.py.
    FastPy compiles this to: int32_t main() { return 0; }
    """
    return 0


