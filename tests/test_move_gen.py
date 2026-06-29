"""
FastPy-Engine — Phase 3 Move Generation Tests
==============================================
Tests for:
  - Ray generators (sliding piece attacks)
  - Sliding piece move generation (bishops, rooks, queens)
  - Attack/check detection (is_sq_attacked, is_in_check)
  - Castling move generation
  - Legal move filtering (generate_legal_moves)
  - Perft correctness benchmarks (depths 1-4 from start position)

Run with: pytest test_move_gen.py -v

Perft correctness reference (starting position):
  perft(1) = 20
  perft(2) = 400
  perft(3) = 8,902
  perft(4) = 197,281
  perft(5) = 4,865,609   ← primary benchmark (compiled only, ~0ms)
"""

import copy
import sys
import os

# Add the repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("engine", os.path.join(os.path.dirname(os.path.dirname(__file__)), "engine.py"))
mod = importlib.util.module_from_spec(spec)
mod.__name__ = "engine"
spec.loader.exec_module(mod)

from engine import (
    BoardState, make_move, evaluate,
    encode_move, encode_move_promo, encode_move_flag,
    move_from, move_to, move_promo, move_flag,
    popcount, lsb, pop_lsb,
    BIT_ONE, FULL_BOARD,
    FILE_A, FILE_H, RANK_1, RANK_2, RANK_4, RANK_7, RANK_8,
    north, south, east, west, north_east, north_west, south_east, south_west,
    ray_north, ray_south, ray_east, ray_west,
    ray_north_east, ray_north_west, ray_south_east, ray_south_west,
    knight_attack_mask, king_attack_mask,
    generate_white_pawns, generate_black_pawns,
    generate_knights, generate_bishops, generate_rooks, generate_queens, generate_king,
    generate_all_moves, generate_castling,
    is_sq_attacked, is_in_check,
    FLAG_CASTLING, FLAG_EN_PASSANT,
    CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ,
    PROMO_QUEEN, PROMO_KNIGHT, PROMO_BISHOP,
    NEG_INF, INF, VAL_PAWN, VAL_KNIGHT,
    WHITE_PAWNS_START, BLACK_PAWNS_START,
    WHITE_KING_START, BLACK_KING_START,
)


# =============================================================================
# Python-mode helpers (copy semantics fix)
# =============================================================================

def _gen_legal(board):
    """Generate legal moves as a Python list (board unchanged)."""
    pseudo = [0] * 218
    pcount = generate_all_moves(board, pseudo, 0)
    out = []
    for i in range(pcount):
        nb = make_move(copy.copy(board), pseudo[i])
        if not is_in_check(nb):
            out.append(pseudo[i])
    return out


def _perft_py(board, depth):
    """Recursive perft using Python list move generation."""
    moves = _gen_legal(board)
    if depth == 1:
        return len(moves)
    nodes = 0
    for m in moves:
        nodes += _perft_py(make_move(copy.copy(board), m), depth - 1)
    return nodes


def _make(board, from_sq, to_sq, flag=0, promo=0):
    """Convenience: encode and apply a move."""
    if promo:
        m = encode_move_promo(from_sq, to_sq, promo)
    elif flag:
        m = encode_move_flag(from_sq, to_sq, flag)
    else:
        m = encode_move(from_sq, to_sq)
    return make_move(copy.copy(board), m)


def _sq(name):
    """Convert algebraic square name to index: 'e4' → 28."""
    col = ord(name[0]) - ord('a')
    row = int(name[1]) - 1
    return row * 8 + col


def _bb(name):
    """Convert algebraic square name to single-bit bitboard."""
    return BIT_ONE << _sq(name)


# =============================================================================
# RAY GENERATOR TESTS
# =============================================================================

class TestRayGenerators:
    """Verify that directional ray fills are correct for all 8 directions."""

    def test_ray_north_empty_board(self):
        """A rook on a1 facing north on an empty board hits all of a2-a8."""
        sq_bb = _bb('a1')
        result = ray_north(sq_bb, 0)
        expected = (
            _bb('a2') | _bb('a3') | _bb('a4') | _bb('a5') |
            _bb('a6') | _bb('a7') | _bb('a8')
        )
        assert result == expected

    def test_ray_north_blocked(self):
        """Ray stops at (and includes) the first blocker."""
        sq_bb = _bb('a1')
        blocker = _bb('a4')
        result = ray_north(sq_bb, blocker)
        expected = _bb('a2') | _bb('a3') | _bb('a4')
        assert result == expected

    def test_ray_south_empty(self):
        sq_bb = _bb('d8')
        result = ray_south(sq_bb, 0)
        expected = (
            _bb('d7') | _bb('d6') | _bb('d5') | _bb('d4') |
            _bb('d3') | _bb('d2') | _bb('d1')
        )
        assert result == expected

    def test_ray_east_empty(self):
        sq_bb = _bb('a4')
        result = ray_east(sq_bb, 0)
        expected = (
            _bb('b4') | _bb('c4') | _bb('d4') | _bb('e4') |
            _bb('f4') | _bb('g4') | _bb('h4')
        )
        assert result == expected

    def test_ray_west_blocked_adjacent(self):
        """Ray blocked immediately (adjacent blocker)."""
        sq_bb = _bb('e4')
        blocker = _bb('d4')
        result = ray_west(sq_bb, blocker)
        assert result == _bb('d4')

    def test_ray_north_east_corner(self):
        """Ray from a1 north-east covers the main diagonal."""
        sq_bb = _bb('a1')
        result = ray_north_east(sq_bb, 0)
        expected = (
            _bb('b2') | _bb('c3') | _bb('d4') | _bb('e5') |
            _bb('f6') | _bb('g7') | _bb('h8')
        )
        assert result == expected

    def test_ray_south_west_corner(self):
        sq_bb = _bb('h8')
        result = ray_south_west(sq_bb, 0)
        expected = (
            _bb('g7') | _bb('f6') | _bb('e5') | _bb('d4') |
            _bb('c3') | _bb('b2') | _bb('a1')
        )
        assert result == expected

    def test_ray_includes_capture_square(self):
        """The blocker square IS included in the attack set (capture available)."""
        sq_bb = _bb('d4')
        blocker = _bb('d7')
        result = ray_north(sq_bb, blocker)
        assert result & _bb('d7'), "Capture square d7 must be included"
        assert not (result & _bb('d8')), "Square behind blocker must NOT be included"

    def test_ray_empty_board_popcount(self):
        """From center of empty board, each diagonal ray has 7 squares."""
        sq_bb = _bb('a1')
        # North from a1: 7 squares
        assert popcount(ray_north(sq_bb, 0)) == 7
        # East from a1: 7 squares
        assert popcount(ray_east(sq_bb, 0)) == 7


# =============================================================================
# ATTACK MASK TESTS
# =============================================================================

class TestAttackMasks:

    def test_knight_center_attacks_8_squares(self):
        sq_bb = _bb('d4')
        mask = knight_attack_mask(sq_bb)
        assert popcount(mask) == 8

    def test_knight_corner_a1_attacks_2_squares(self):
        sq_bb = _bb('a1')
        mask = knight_attack_mask(sq_bb)
        assert popcount(mask) == 2
        assert mask & _bb('b3')
        assert mask & _bb('c2')

    def test_knight_no_wrap_around_file_h(self):
        """Knight on h1 must not wrap to a2/a3."""
        sq_bb = _bb('h1')
        mask = knight_attack_mask(sq_bb)
        assert not (mask & FILE_A), "Knight on h-file must not wrap to a-file"

    def test_king_center_attacks_8_squares(self):
        sq_bb = _bb('e4')
        mask = king_attack_mask(sq_bb)
        assert popcount(mask) == 8

    def test_king_corner_attacks_3_squares(self):
        sq_bb = _bb('a1')
        mask = king_attack_mask(sq_bb)
        assert popcount(mask) == 3
        assert mask & _bb('a2')
        assert mask & _bb('b1')
        assert mask & _bb('b2')


# =============================================================================
# SLIDING PIECE GENERATOR TESTS
# =============================================================================

class TestSlidingPieces:

    def _count_bishop_moves(self, from_sq_name, occupied=0, friendly=0):
        sq_bb = _bb(from_sq_name)
        buf = [0] * 218
        cnt = generate_bishops(sq_bb, friendly, occupied, buf, 0)
        return cnt

    def _count_rook_moves(self, from_sq_name, occupied=0, friendly=0):
        sq_bb = _bb(from_sq_name)
        buf = [0] * 218
        cnt = generate_rooks(sq_bb, friendly, occupied, buf, 0)
        return cnt

    def test_bishop_center_empty_board(self):
        """Bishop on d4 of an empty board has 13 moves."""
        assert self._count_bishop_moves('d4') == 13

    def test_bishop_corner_empty_board(self):
        """Bishop on a1 has only one diagonal: 7 squares."""
        assert self._count_bishop_moves('a1') == 7

    def test_bishop_blocked_by_friendly(self):
        """Friendly pieces reduce available squares."""
        sq_bb = _bb('d4')
        friendly = _bb('f6')   # Block one diagonal
        buf = [0] * 218
        cnt = generate_bishops(sq_bb, friendly, _bb('f6'), buf, 0)
        # f6 is excluded (friendly), e5 still reachable
        for i in range(cnt):
            assert move_to(buf[i]) != _sq('f6'), "Cannot land on friendly f6"

    def test_bishop_captures_enemy(self):
        """Enemy pieces are included in attack set."""
        sq_bb = _bb('d4')
        enemy = _bb('f6')
        occupied = enemy
        buf = [0] * 218
        cnt = generate_bishops(sq_bb, 0, occupied, buf, 0)
        targets = {move_to(buf[i]) for i in range(cnt)}
        assert _sq('f6') in targets, "Should be able to capture f6"
        assert _sq('g7') not in targets, "Cannot go past f6"

    def test_rook_center_empty_board(self):
        """Rook on d4 of an empty board has 14 moves."""
        assert self._count_rook_moves('d4') == 14

    def test_rook_edge_empty_board(self):
        """Rook on a1 empty board: 7 north + 7 east = 14."""
        assert self._count_rook_moves('a1') == 14

    def test_rook_blocked_both_axes(self):
        """Rook with blockers on all 4 axes."""
        sq_bb = _bb('d4')
        occupied = _bb('d6') | _bb('d2') | _bb('b4') | _bb('f4')
        buf = [0] * 218
        cnt = generate_rooks(sq_bb, 0, occupied, buf, 0)
        # Should reach d5,d6 north; d3,d2 south; e4,f4 east; c4,b4 west = 2+2+2+2 = 8
        assert cnt == 8

    def test_queen_center_empty_board(self):
        """Queen on d4 empty board = bishop moves + rook moves = 13 + 14 = 27."""
        sq_bb = _bb('d4')
        buf = [0] * 218
        cnt = generate_queens(sq_bb, 0, 0, buf, 0)
        assert cnt == 27

    def test_queen_in_starting_position_blocked(self):
        """White queen on d1 (starting position) has 0 moves (all blocked)."""
        b = BoardState()
        buf = [0] * 218
        # White queen on d1 is sq 3
        occupied = b.all_pieces()
        friendly = b.white_pieces()
        cnt = generate_queens(_bb('d1'), friendly, occupied, buf, 0)
        assert cnt == 0, f"Queen on d1 in start pos should have 0 moves, got {cnt}"


# =============================================================================
# CHECK DETECTION TESTS
# =============================================================================

class TestCheckDetection:

    def test_starting_position_no_check(self):
        """Neither king is in check at the start."""
        b = BoardState()
        # White king not attacked by black
        assert not is_sq_attacked(_sq('e1'), b, True), "e1 not attacked by black"
        # Black king not attacked by white
        assert not is_sq_attacked(_sq('e8'), b, False), "e8 not attacked by white"

    def test_pawn_attacks_white(self):
        """White pawns on rank 2 attack their diagonal squares."""
        b = BoardState()
        # by_black=False means "attacked by white pieces"
        assert is_sq_attacked(_sq('d3'), b, False), "d3 attacked by e2 pawn"
        assert is_sq_attacked(_sq('f3'), b, False), "f3 attacked by e2 pawn"
        # e3 IS attacked — by the d2 and f2 pawns (south_pawn reverse trace)
        # is_sq_attacked reverse-traces: south_east(e3)|south_west(e3) = f2|d2, both have white pawns
        assert is_sq_attacked(_sq('e3'), b, False), "e3 IS attacked by d2 and f2 white pawns"
        # e2 IS also attacked — by the g1 white knight (g1→e2 is a valid knight move)
        assert is_sq_attacked(_sq('e2'), b, False), "e2 attacked by g1 knight"
        # Rank 5-8 are not reachable by any white piece from start position
        assert not is_sq_attacked(_sq('e5'), b, False), "e5 not attacked by white from start"

    def test_pawn_attacks_black(self):
        """Black pawns on rank 7 attack rank 6 squares."""
        b = BoardState()
        # Black pawn on e7 attacks d6 and f6
        assert is_sq_attacked(_sq('d6'), b, True), "d6 attacked by e7 black pawn"
        assert is_sq_attacked(_sq('f6'), b, True), "f6 attacked by e7 black pawn"

    def test_knight_attacks_from_start(self):
        """White knights on b1/g1 attack certain squares."""
        b = BoardState()
        # b1 knight attacks a3 and c3
        assert is_sq_attacked(_sq('a3'), b, False), "a3 attacked by b1 knight"
        assert is_sq_attacked(_sq('c3'), b, False), "c3 attacked by b1 knight"

    def test_bishop_reveals_attack_after_pawn_move(self):
        """After e2e4, bishop on f1 diagonal opens up, but f1 bishop still blocked."""
        b = _make(BoardState(), _sq('e2'), _sq('e4'))
        # f1 bishop now sees d3 (diagonal through e2 which is empty)
        assert is_sq_attacked(_sq('d3'), b, False), "d3 attacked by f1 bishop via open diagonal"

    def test_is_in_check_after_legal_move(self):
        """After any legal starting-position move, neither king should be in check."""
        b = BoardState()
        moves = _gen_legal(b)
        for m in moves:
            nb = make_move(copy.copy(b), m)
            # White just moved; black to move now. White king should not be in check.
            assert not is_in_check(nb), (
                f"is_in_check fired after legal white move from={move_from(m)} to={move_to(m)}"
            )

    def test_scholar_mate_check(self):
        """White queen on e5 attacks e8 via north ray — checks black king."""
        b = BoardState()
        b.white_pawns = 0; b.white_knights = 0; b.white_bishops = 0
        b.white_rooks = 0; b.white_queens = _bb('e5')
        b.black_pawns = 0; b.black_knights = 0; b.black_bishops = 0
        b.black_rooks = 0; b.black_queens = 0
        b.black_king  = _bb('e8')   # King on e8, path e6-e7 is clear
        b.castling_rights = 0
        # is_in_check semantics: white_to_move=True means black JUST moved →
        # checks whether BLACK king is attacked by white
        b.white_to_move = True
        assert is_sq_attacked(_sq('e8'), b, False), "e8 attacked by queen on e5 (north ray)"
        assert is_in_check(b), "is_in_check: black king on e8 in check from white queen on e5"


# =============================================================================
# CASTLING TESTS
# =============================================================================

class TestCastling:

    def _clear_kingside_path_white(self, board):
        """Remove pieces between king and kingside rook for white."""
        board.white_knights = board.white_knights & ~_bb('g1')
        board.white_bishops = board.white_bishops & ~_bb('f1')
        return board

    def _clear_queenside_path_white(self, board):
        """Remove pieces between king and queenside rook for white."""
        board.white_queens  = board.white_queens  & ~_bb('d1')
        board.white_bishops = board.white_bishops & ~_bb('c1')
        board.white_knights = board.white_knights & ~_bb('b1')
        return board

    def test_no_castling_starting_position(self):
        """Castling not available from start because paths are blocked."""
        b = BoardState()
        buf = [0] * 218
        cnt = generate_castling(b, buf, 0)
        assert cnt == 0, "No castling possible with pieces blocking the paths"

    def test_white_kingside_castling_available(self):
        """After clearing path, white kingside castling is generated."""
        b = self._clear_kingside_path_white(BoardState())
        buf = [0] * 218
        cnt = generate_castling(b, buf, 0)
        assert cnt == 1
        m = buf[0]
        assert move_from(m) == _sq('e1')
        assert move_to(m)   == _sq('g1')
        assert move_flag(m) == FLAG_CASTLING

    def test_white_queenside_castling_available(self):
        """After clearing queenside path, castling is generated."""
        b = self._clear_queenside_path_white(BoardState())
        buf = [0] * 218
        cnt = generate_castling(b, buf, 0)
        assert cnt == 1
        m = buf[0]
        assert move_from(m) == _sq('e1')
        assert move_to(m)   == _sq('c1')
        assert move_flag(m) == FLAG_CASTLING

    def test_castling_rights_cleared_after_king_move(self):
        """Moving the white king clears both white castling rights."""
        b = self._clear_kingside_path_white(BoardState())
        # Move king e1→d1
        b2 = _make(b, _sq('e1'), _sq('d1'))
        assert not (b2.castling_rights & CASTLE_WK), "WK right cleared after king move"
        assert not (b2.castling_rights & CASTLE_WQ), "WQ right cleared after king move"

    def test_castling_rights_cleared_after_rook_move(self):
        """Moving the h1 rook clears white kingside castling right."""
        b = self._clear_kingside_path_white(BoardState())
        # Move h1 rook h1→h2 (need h2 to be empty; in start pos it's not — use h1→h3 after pawn)
        # Simplest: directly manipulate
        b.white_pawns = b.white_pawns & ~_bb('h2')   # clear h2 pawn
        b2 = _make(b, _sq('h1'), _sq('h2'))
        assert not (b2.castling_rights & CASTLE_WK), "WK right cleared after h1 rook move"
        assert b2.castling_rights & CASTLE_WQ, "WQ right still set"

    def test_castling_rights_cleared_when_rook_captured(self):
        """When the h1 rook is captured, white kingside castling right clears."""
        b = self._clear_kingside_path_white(BoardState())
        # Place a black rook on g1 to capture the h1 rook via g1→h1
        b.black_rooks = _bb('g1')
        b.black_pawns = 0; b.black_knights = 0; b.black_bishops = 0
        b.white_to_move = False
        b2 = _make(b, _sq('g1'), _sq('h1'))
        assert not (b2.castling_rights & CASTLE_WK), "WK right cleared when h1 rook captured"

    def test_castling_executes_rook_move(self):
        """After white kingside castling, king is on g1 and rook is on f1."""
        b = self._clear_kingside_path_white(BoardState())
        b2 = _make(b, _sq('e1'), _sq('g1'), flag=FLAG_CASTLING)
        assert b2.white_king  == _bb('g1'), "King should be on g1"
        assert b2.white_rooks & _bb('f1'),  "Rook should be on f1"
        assert not (b2.white_rooks & _bb('h1')), "Rook should not still be on h1"

    def test_queenside_castling_executes_rook_move(self):
        """After white queenside castling, king is on c1 and rook is on d1."""
        b = self._clear_queenside_path_white(BoardState())
        b2 = _make(b, _sq('e1'), _sq('c1'), flag=FLAG_CASTLING)
        assert b2.white_king  == _bb('c1'), "King should be on c1"
        assert b2.white_rooks & _bb('d1'),  "Rook should be on d1"
        assert not (b2.white_rooks & _bb('a1')), "Rook should not still be on a1"

    def test_cannot_castle_through_check(self):
        """Castling is illegal if the king passes through an attacked square."""
        b = self._clear_kingside_path_white(BoardState())
        # Clear f2 pawn so the black rook on f8 has a clear line to f1
        b.white_pawns = b.white_pawns & ~_bb('f2')
        b.black_rooks = _bb('f8')
        b.black_pawns = 0; b.black_knights = 0; b.black_bishops = 0
        b.black_queens = 0
        # Verify f1 is actually attacked before asserting no castling
        assert is_sq_attacked(_sq('f1'), b, True), "f1 must be attacked by black rook for this test"
        buf = [0] * 218
        cnt = generate_castling(b, buf, 0)
        assert cnt == 0, "Cannot castle through f1 when f1 is attacked"

    def test_cannot_castle_when_in_check(self):
        """Cannot castle when the king's starting square is under attack."""
        b = self._clear_kingside_path_white(BoardState())
        # Black rook on e8 attacks e1
        b.black_rooks = _bb('e8')
        b.black_pawns = 0; b.black_knights = 0; b.black_bishops = 0
        b.black_queens = 0
        # Need to clear ranks between so rook can see e1
        b.white_pawns = b.white_pawns & ~_bb('e2')
        buf = [0] * 218
        cnt = generate_castling(b, buf, 0)
        assert cnt == 0, "Cannot castle when in check"


# =============================================================================
# LEGAL MOVE GENERATION TESTS
# =============================================================================

class TestLegalMoves:

    def test_starting_position_20_legal_moves(self):
        """From the starting position, white has exactly 20 legal moves."""
        b = BoardState()
        moves = _gen_legal(b)
        assert len(moves) == 20, f"Expected 20, got {len(moves)}"

    def test_black_starting_position_20_legal_moves(self):
        """Black also has exactly 20 legal moves from the start."""
        b = BoardState()
        b.white_to_move = False
        moves = _gen_legal(b)
        assert len(moves) == 20, f"Expected 20, got {len(moves)}"

    def test_legal_moves_exclude_king_exposing_moves(self):
        """A pinned piece cannot move (would expose king)."""
        # White king on e1, black rook on e8, white rook on e4 (pinned)
        b = BoardState()
        b.white_pawns = 0; b.white_knights = 0; b.white_bishops = 0
        b.white_queens = 0
        b.white_rooks = _bb('e4')
        b.white_king  = _bb('e1')
        b.black_pawns = 0; b.black_knights = 0; b.black_bishops = 0
        b.black_queens = 0
        b.black_rooks = _bb('e8')
        b.black_king  = _bb('a8')
        b.castling_rights = 0
        moves = _gen_legal(b)
        # White rook on e4 is pinned on the e-file by black rook on e8
        # The pinned rook can only move along the e-file (e2,e3,e5,e6,e7,e8)
        rook_moves = [m for m in moves if move_from(m) == _sq('e4')]
        for m in rook_moves:
            # All legal rook moves must stay on the e-file (sq column == 4)
            assert move_to(m) % 8 == 4, (
                f"Pinned rook moved off e-file: to sq {move_to(m)}"
            )

    def test_legal_moves_in_check_are_limited(self):
        """When in check, only moves that resolve the check are legal."""
        b = BoardState()
        # White king on e1, black queen on e4 giving check through empty e-file
        b.white_pawns = 0; b.white_knights = 0; b.white_bishops = 0
        b.white_queens = 0; b.white_rooks = 0
        b.white_king  = _bb('e1')
        b.black_pawns = 0; b.black_knights = 0; b.black_bishops = 0
        b.black_rooks = 0
        b.black_queens = _bb('e5')
        b.black_king   = _bb('a8')
        b.castling_rights = 0
        # Is e1 really in check? Queen on e5 attacks e1 via north ray going south
        assert is_sq_attacked(_sq('e1'), b, True), "e1 should be attacked by black queen on e5"
        moves = _gen_legal(b)
        # King must move off the e-file or block
        for m in moves:
            nb = make_move(copy.copy(b), m)
            # After each legal move, the white king must not be in check
            assert not is_sq_attacked(lsb(nb.white_king), nb, True), (
                f"Legal move left king in check: from={move_from(m)} to={move_to(m)}"
            )

    def test_no_legal_moves_returns_zero(self):
        """A completely stalemated position returns 0 legal moves."""
        # Classic stalemate: black king on a8, white queen on b6, white king on c6
        b = BoardState()
        b.white_pawns = 0; b.white_knights = 0; b.white_bishops = 0
        b.white_rooks = 0; b.white_queens = _bb('b6')
        b.white_king  = _bb('c6')
        b.black_pawns = 0; b.black_knights = 0; b.black_bishops = 0
        b.black_rooks = 0; b.black_queens = 0
        b.black_king  = _bb('a8')
        b.castling_rights = 0
        b.white_to_move = False
        moves = _gen_legal(b)
        assert len(moves) == 0, f"Stalemate: expected 0 legal moves, got {len(moves)}"


# =============================================================================
# MAKE MOVE TESTS
# =============================================================================

class TestMakeMove:

    def test_pawn_double_push_sets_ep(self):
        """After e2e4, en_passant_square is set to e3."""
        b = _make(BoardState(), _sq('e2'), _sq('e4'))
        assert b.en_passant_square == _bb('e3'), "EP square should be e3 after e2e4"

    def test_single_pawn_push_clears_ep(self):
        """After e2e3, no en passant square is set."""
        b = _make(BoardState(), _sq('e2'), _sq('e3'))
        assert b.en_passant_square == 0

    def test_make_move_does_not_modify_original(self):
        """make_move must not modify the caller's board (value-copy semantics)."""
        b = BoardState()
        original_pawns = b.white_pawns
        b_copy = copy.copy(b)
        _ = make_move(b_copy, encode_move(_sq('e2'), _sq('e4')))
        assert b.white_pawns == original_pawns, "Original board was modified!"

    def test_promotion_queen(self):
        """A pawn on rank 7 promotes to queen."""
        b = BoardState()
        b.white_pawns = _bb('e7')
        b.black_pawns = 0; b.black_pieces
        b.castling_rights = 0
        # Clear the destination square
        b.black_pawns = 0; b.black_knights = 0; b.black_bishops = 0
        b.black_rooks = 0; b.black_queens = 0
        b2 = _make(b, _sq('e7'), _sq('e8'), promo=PROMO_QUEEN)
        assert b2.white_queens & _bb('e8'), "Queen should appear on e8"
        assert not (b2.white_pawns & _bb('e7')), "Pawn should be gone from e7"

    def test_en_passant_capture_removes_pawn(self):
        """En passant capture removes the captured pawn from the board."""
        b = BoardState()
        b.white_pawns = _bb('e5')
        b.black_pawns = _bb('d5')
        b.en_passant_square = _bb('d6')
        b.castling_rights = 0
        b2 = _make(b, _sq('e5'), _sq('d6'), flag=FLAG_EN_PASSANT)
        assert not (b2.black_pawns & _bb('d5')), "Captured black pawn should be removed from d5"
        assert b2.white_pawns & _bb('d6'), "White pawn should be on d6"

    def test_side_to_move_flips(self):
        """white_to_move alternates after each move."""
        b = BoardState()
        assert b.white_to_move
        b2 = _make(b, _sq('e2'), _sq('e4'))
        assert not b2.white_to_move
        b3 = _make(b2, _sq('e7'), _sq('e5'))
        assert b3.white_to_move

    def test_capture_removes_piece(self):
        """Capturing a piece removes it from its bitboard."""
        b = BoardState()
        # Move white pawn e2e4, then d7d5, then exd5
        b2 = _make(b,  _sq('e2'), _sq('e4'))
        b3 = _make(b2, _sq('d7'), _sq('d5'))
        b4 = _make(b3, _sq('e4'), _sq('d5'))
        assert not (b4.black_pawns & _bb('d5')), "Black pawn on d5 should be captured"
        assert b4.white_pawns & _bb('d5'), "White pawn should be on d5"


# =============================================================================
# PERFT CORRECTNESS
# The gold standard for move generation verification.
# =============================================================================

class TestPerft:

    def test_perft_1(self):
        assert _perft_py(BoardState(), 1) == 20

    def test_perft_2(self):
        assert _perft_py(BoardState(), 2) == 400

    def test_perft_3(self):
        assert _perft_py(BoardState(), 3) == 8902

    def test_perft_4(self):
        assert _perft_py(BoardState(), 4) == 197281
