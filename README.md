# Chess Engine with Learned Evaluation

A chess engine in Python that searches with alpha-beta minimax and supports two interchangeable evaluation functions: a hand-crafted classical evaluator and a neural network trained on Stockfish-labelled positions. Built to compare learned vs hand-crafted evaluation at equal search depth.

## Results

| Metric | Value |
|---|---|
| NN evaluation MAE (offline) | 81 cp (down from 290) |
| NN evaluation MSE (offline) | 17,932 (down from 84,306) |
| Training data | ~200,000 positions from 5,000 games |
| Features per position | 29 |
| NN vs Classical (10 games at 1s/move) | 0-10 |
| Search depth reached at 1s/move | Classical: ~4, NN: ~2-3 |

The neural network achieved strong offline accuracy — predicting Stockfish's depth-6 evaluation to within ~81 centipawns on held-out positions — but lost decisively in head-to-head tournament play.

The cause is the speed-accuracy tradeoff: TensorFlow inference takes ~5-10ms per call, while the classical evaluator runs in microseconds. At the same time budget, the classical engine searches roughly one ply deeper, and that extra depth dominates the NN's per-position accuracy advantage. This mirrors the broader history of chess engines: classical evaluation was state-of-the-art for decades, and modern neural approaches (NNUE) only succeeded by specifically designing tiny, fast networks for inference speed within alpha-beta search.

## How it works

**Search.** Alpha-beta minimax with several standard refinements:
- Quiescence search at leaf nodes to avoid the horizon effect
- Move ordering by MVV-LVA (most valuable victim, least valuable attacker) to maximise pruning
- Transposition table with Zobrist hashing to avoid re-searching repeated positions
- Iterative deepening with a time budget instead of fixed depth
- Exception-safe board mutation (try/finally around push/pop) to handle search timeouts safely

**Classical evaluation.** Piece values plus piece-square tables, in centipawns. The standard hand-crafted approach used by chess engines for decades before NNUE.

**Neural network evaluation.** A feed-forward network (128 → 64 → 32 → 1) trained to predict Stockfish's depth-6 evaluation from White's perspective. Inputs are 29 normalised features:
- Per-piece-type material counts for each side
- Piece-square table sums per piece type
- Mobility for both sides
- King safety (castling rights, in check)
- Centre control, side to move

## Project history

Started as a first deep learning project, motivated by curiosity about how game-playing AIs work (and wanting to beat my brother at chess). The initial version was unusable — moves took minutes due to inefficient code structure (a 13MB CSV loading on every import, `model.predict()` called hundreds of times per search with massive Python overhead) and the NN was undertrained (100 games, 5 features, MAE of 290 centipawns).

Returned to the project and rebuilt it properly:

- Profiled and fixed the bottlenecks → moves now take seconds, not minutes
- Added quiescence search, move ordering, transposition tables → engine plays competently
- Retrained NN with 50× more data and richer features → MAE dropped from 290 to 81 cp
- Built a tournament harness to compare evaluators rigorously
- Discovered the speed-accuracy tradeoff through head-to-head play

The most interesting debugging finding came from the tournament: an early implementation showed the NN winning 5-5 against classical, which turned out to be an artifact of how the eval function was being swapped between turns (monkey-patching that didn't actually propagate). After fixing the comparison harness properly, the true result emerged — and was both more decisive and more informative than the original fake one.

## Usage

```bash
# Train the neural network (slow; uses Stockfish to label positions)
python game.py

# Play against the AI (uses classical evaluation by default)
python ai.py

# Run the NN vs Classical tournament
python compare_engines.py
```

Requires: `python-chess`, `tensorflow`, `scikit-learn`, `pandas`, `numpy`, and a Stockfish binary at `stockfish/stockfish.exe`.

## Possible future work

- Implement an NNUE-style architecture (small, fast networks designed for inference speed within search)
- Scale training data 10× and add tactical positions to teach the NN about non-quiet positions
- Deploy to Lichess for a public Elo rating
- Add killer-move heuristic and null-move pruning for deeper search
