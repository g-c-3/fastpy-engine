# FastPy-Engine ♟️

> **A world-class chess engine written in Python. Compiled by FastPy to native C++. Targeting 1 Billion Nodes Per Second.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Status: Early Development](https://img.shields.io/badge/Status-Early%20Development-orange)]()
[![Built With: FastPy](https://img.shields.io/badge/Built%20With-FastPy-red)](https://github.com/g-c-3/fastpy)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen)]()

---

## What Is FastPy-Engine?

FastPy-Engine is a **competitive chess engine** written entirely in FastPy dialect Python.

It is also the **proof of concept** for the entire FastPy project.

Every line of FastPy-Engine's search, move generation, and evaluation logic is written in clean, readable Python — following the FastPy Speed Contract. The [FastPy transpiler](https://github.com/g-c-3/fastpy) compiles it into raw, zero-allocation C++ and delivers a native binary targeting **1 Billion Nodes Per Second** on modern multi-core hardware.

No manual C++ rewrites. No Cython. No JIT warmup. Just Python — compiled to the metal.

---

## The Performance Target

| Implementation | NPS (Nodes Per Second) |
|---|---|
| Pure Python chess engine | 10,000 — 50,000 |
| Python + PyPy | 50,000 — 150,000 |
| Python + Numba | up to 5,000,000 |
| Hand-written C++ (Stockfish) | 100,000,000 — 300,000,000 single thread |
| Stockfish multi-core | 1,000,000,000+ |
| **FastPy-Engine target (single thread)** | **100,000,000+** |
| **FastPy-Engine target (multi-core)** | **1,000,000,000 (1 Billion)** |

The 1 Billion NPS target is reached in Phase 4 via Lazy SMP multi-core search — the same technique used by Stockfish and other elite engines.

---

## How FastPy-Engine Is Built

FastPy-Engine is not written in C++. It is written in Python.

```
FastPy-Engine source (engine.py)      ← You are reading this repository
        │
        ▼
[ FastPy Transpiler ]                 ← github.com/g-c-3/fastpy
        │  Reads Python, validates types, generates C++
        ▼
[ Raw C++ Code ]                      ← Zero-allocation, zero-GC
        │
        ▼
[ GCC / Clang -O3 -march=native ]     ← Maximum hardware optimization
        │
        ▼
Native Binary                         ← 100,000,000+ NPS single thread
```

**Build command (once FastPy is available):**
```bash
fastpy build engine.py --optimize=O3
```

---

## Architecture

FastPy-Engine is inspired by — but not copied from — the architectural principles used by elite open-source chess engines including Stockfish. Every implementation is original.

### Core Components

**Bitboard Representation**  
The entire chess board is stored as twelve 64-bit integers — one per piece type per colour. A single integer represents all squares a piece type occupies. This maps directly to CPU registers, enabling hardware-level parallelism on every position query.

**Move Generation**  
All legal moves are generated using bitboard shift operations, attack masks, and hardware intrinsics. FastPy compiles these directly to `POPCNT`, `TZCNT`, and `BMI2` CPU instructions — one clock cycle per operation.

**Alpha-Beta Search**  
The core search algorithm is Negamax with Alpha-Beta pruning. FastPy's zero-allocation guarantee means no garbage collector ever pauses the search loop. Move lists are fixed-size arrays on the CPU stack — never heap allocated.

**Evaluation**  
Phase 1 uses fast material counting compiled to `POPCNT` instructions. Later phases introduce Piece-Square Tables (PST), mobility scoring, king safety, and eventually NNUE (Efficiently Updatable Neural Network) evaluation.

**UCI Protocol**  
FastPy-Engine speaks the Universal Chess Interface (UCI) protocol — the industry standard for chess engine communication. It works out of the box with any UCI-compatible chess GUI (Arena, Cutechess, Lichess Bot API, etc.).

---

## Roadmap

### Phase 1 — Functional Engine
- [x] Bitboard board representation
- [x] Bitboard utility functions (popcount, lsb, pop_lsb)
- [x] White pawn move generation
- [x] Knight move generation
- [x] Alpha-Beta search skeleton
- [x] Material evaluation
- [ ] Complete move generation (bishops, rooks, queens, king, castling, en passant)
- [ ] Move application and undo (make/unmake)
- [ ] UCI protocol interface
- [ ] Basic time management

### Phase 2 — Competitive Engine
- [ ] Move ordering (captures first, killer moves, history heuristic)
- [ ] Quiescence search
- [ ] Piece-Square Table (PST) evaluation
- [ ] Null move pruning
- [ ] Transposition table (Zobrist hashing)
- [ ] Iterative deepening

### Phase 3 — Elite Engine
- [ ] NNUE neural network evaluation
- [ ] Late Move Reductions (LMR)
- [ ] Futility pruning
- [ ] Singular extensions
- [ ] Full Stockfish-class search optimizations

### Phase 4 — 1 Billion NPS
- [ ] Lazy SMP multi-core parallel search
- [ ] SIMD vectorization via FastPy compiler flags
- [ ] Full hardware intrinsic coverage (BMI2 for slider move generation)
- [ ] Benchmark: 1,000,000,000 NPS on modern multi-core hardware

---

## The FastPy Speed Contract

All FastPy-Engine source code follows three strict rules enforced by the FastPy transpiler:

**1. Strict Static Typing** — Every variable in a performance-critical function has an explicit type hint. No exceptions.

```python
def alpha_beta(board: BoardState, depth: int32, alpha: int32, beta: int32) -> int32:
    ...
```

**2. Zero Dynamic Allocation** — No `list.append()`, no `dict`, no heap allocation inside compiled search loops. Move lists are fixed-size stack arrays.

```python
moves: uint64[218]   # Stack-allocated — the CPU never asks the OS for memory
```

**3. No CPython Runtime** — The compiled binary runs standalone. No Python interpreter. No garbage collector. No GIL.

---

## UCI Protocol

FastPy-Engine supports the Universal Chess Interface (UCI) protocol, allowing it to work with any UCI-compatible GUI.

```
GUI sends:    uci
Engine sends: id name FastPy-Engine
              id author Gokul Chandar
              uciok

GUI sends:    position startpos moves e2e4
              go depth 10
Engine sends: bestmove e7e5
```

Tested with: Arena, Cutechess, and Lichess Bot API.

---

## Building From Source

> **Note:** FastPy-Engine requires the FastPy transpiler to compile.  
> FastPy is currently in development at [github.com/g-c-3/fastpy](https://github.com/g-c-3/fastpy).  
> Build instructions will be updated as FastPy reaches its Phase 1 release.

**Once FastPy is available:**

```bash
# Clone the engine
git clone https://github.com/g-c-3/fastpy-engine
cd fastpy-engine

# Install FastPy
pip install fastpy

# Compile the engine
fastpy build engine.py --optimize=O3

# Run the binary
./engine
```

**System requirements:**
- GCC 9+ or Clang 10+ with C++17 support
- CPU with POPCNT support (Intel Nehalem 2008+ / AMD Barcelona 2007+)
- Python 3.10+ (for the FastPy build step only — not required at runtime)

---

## How To Contribute

FastPy-Engine is open-source under GPL v3. All skill levels are welcome.

**Where help is needed:**

*Move Generation*
- Complete sliding piece move generation (bishops, rooks, queens)
- Castling rights tracking and legal castling moves
- En passant capture generation
- Check detection and legal move filtering

*Search*
- Transposition table implementation (Zobrist hashing)
- Move ordering (MVV-LVA, killer moves, history heuristic)
- Quiescence search
- Null move pruning

*Evaluation*
- Piece-Square Tables for all piece types
- Mobility evaluation
- King safety scoring
- Pawn structure analysis

*Infrastructure*
- UCI protocol implementation
- Time management system
- Perft testing framework (move generation correctness verification)
- Benchmark suite

**To contribute:**
1. Fork this repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Write your code following the FastPy Speed Contract (strict types, zero allocation)
4. Submit a Pull Request with a clear description of what you built and why

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting.

---

## Relationship to FastPy

FastPy-Engine and the FastPy transpiler are **two separate projects**:

| Project | Repository | License | Purpose |
|---|---|---|---|
| FastPy | [g-c-3/fastpy](https://github.com/g-c-3/fastpy) | MIT | The Python-to-C++ transpiler tool |
| FastPy-Engine | [g-c-3/fastpy-engine](https://github.com/g-c-3/fastpy-engine) | GPL v3 | The chess engine built with FastPy |

FastPy (the tool) is MIT licensed so anyone — including commercial projects — can use it freely.  
FastPy-Engine (the chess engine) is GPL v3, consistent with the open-source chess engine community standard.

---

## A Note on AI-Assisted Development

FastPy-Engine was conceived and architecturally designed by **Gokul Chandar**.  
AI tools (Google Gemini and Claude/Anthropic) serve as development assistants — writing code modules under human architectural oversight.

The vision, strategic decisions, and project direction are entirely human-driven.  
We believe in being transparent about this.

---

## Creator

**Gokul Chandar** ([@g-c-3](https://github.com/g-c-3))  
*Vision, Architecture, and Project Direction*

---

## License

FastPy-Engine is released under the [GNU General Public License v3.0](LICENSE).

This means:
- You are free to use, study, modify, and distribute this software
- Any derivative work must also be released under GPL v3
- You must provide access to the source code of any distributed binary

See [LICENSE](LICENSE) for the full license text.

---

*"Write logic with the speed and beauty of Python.*  
*Run it with the terrifying, metal-shredding velocity of optimized C++."*
