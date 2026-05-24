# Chess Engine with Learned Evaluation

A chess engine in Python that searches with alpha-beta minimax and supports two interchangeable evaluation functions: a hand-crafted classical evaluator and a neural network trained on Stockfish-labelled positions. Built to compare learned vs hand-crafted evaluation at equal search depth.

## Results

| Metric | Before | After |
|---|---|---|
| NN evaluation MAE | 290 cp | **82 cp** |
| NN evaluation MSE | 84,306 | **18,646** |
| NN vs Classical (10 games, 1s/move) | — | 5-5 (50%) |
| Training data | 100 games, 5 features | 5,000 games, 29 features |

After improving the training pipeline, the NN's evaluation error dropped ~3.5× on MAE and ~4.5× on MSE. In head-to-head play, the two evaluators draw at equal time control — the NN learned chess evaluation from scratch but doesn't yet outperform a well-tuned classical baseline.

## How it works

**Search.** Alpha-beta minimax with several standard refinements:
- Quiescence search at leaf nodes to avoid the horizon effect
- Move ordering by MVV-LVA (most valuable victim, least valuable attacker) to maximise pruning
- Transposition table with Zobrist hashing to avoid re-searching repeated positions
- Iterative deepening with a time budget instead of fixed depth

**Classical evaluation.** Piece values plus piece-square tables, in centipawns. The standard hand-crafted approach used by chess engines for decades before NNUE.

**Neural network evaluation.** A feed-forward network (128 → 64 → 32 → 1) trained to predict Stockfish's depth-10 evaluation. Inputs are 29 normalised features:
- Per-piece-type material counts for each side
- Piece-square table sums per piece type
- Mobility for both sides
- King safety (castling rights, in check)
- Centre control, side to move

## Project history

Started as a first deep learning project, motivated by curiosity about how game-playing AIs actually work (and wanting to beat my brother at chess). The initial version was unusable — moves took minutes due to inefficient code structure (a 13MB CSV loading on every import, `model.predict()` called hundreds of times per search with massive Python overhead) and the NN was undertrained (100 games, 5 features, MAE of 290 centipawns).

Returned to the project in May 2026 with more programming experience and rebuilt it properly:
- Profiled and fixed the bottlenecks → moves now take seconds, not minutes
- Added quiescence search, move ordering, transposition tables → engine plays competently
- Retrained NN with 50× more data and richer features → MAE dropped from 290 to 82 cp
- Built a tournament harness to compare evaluators rigorously

## Usage

```bash
# Train the neural network (slow; uses Stockfish to label positions)
python game.py

# Play against the AI
python ai.py

# Run the NN vs Classical tournament
python compare_engines.py
```

Requires: `python-chess`, `tensorflow`, `scikit-learn`, `pandas`, `numpy`, and a Stockfish binary at `stockfish/stockfish.exe`.

## Possible future work

- Scale training data 10× and see whether NN pulls ahead
- Deploy to Lichess for a public Elo rating
- Replace MLP with a small CNN over the 8×8 board (proper chess-aware architecture)
- Add killer-move heuristic and null-move pruning for deeper search
