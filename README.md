# BeamForge

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) ![Rust](https://img.shields.io/badge/Rust-000000?style=flat&logo=rust&logoColor=white) [![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=flat&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/infinition)

An experimental text generation engine written in Rust from scratch. Instead of neural networks, BeamForge uses a stochastic semantic graph paired with a Beam Search algorithm. The goal is to explore what you can get from classical probabilistic methods with fast native tokenization and online learning.

---

## How it works

**Tokenization**: Raw text is converted to `u32` identifiers. During generation the model works with integer arrays rather than strings, which keeps memory and CPU usage low.

**Beam Search**: Rather than greedy next-token selection, the algorithm explores multiple candidate sequences in parallel (`BEAM_WIDTH = 5`) and picks the globally highest-scoring completion. Scores are normalized by length to prevent the model from padding endlessly.

**Online learning**: Every declarative sentence typed by the user updates synaptic weights in real time. No training epochs, no restart.

**Binary persistence**: The brain state saves and loads via `bincode`. Fast enough to use between sessions.

---

## Internal architecture

The `SemanticMesh` brain uses three learning structures:

1. **Trigrams (Synapses)**: `w1 + w2 -> target`. Strict two-word context to a third.
2. **Bigrams (Fallback)**: `w2 -> target`. Used when the strict context is unknown.
3. **Fast-Forward**: Skips concepts to restart generation when user input is too short to match.

---

## Building

Requires Rust (stable).

**Windows:**

```powershell
winget install --id Rustlang.Rustup -e
rustup default stable
```

**Linux / macOS:**

```bash
curl https://sh.rustup.rs -sSf | sh
source "$HOME/.cargo/env"
```

**Run:**

```bash
cargo run --bin beamforge
```

**Release build:**

```bash
cargo build --release --bin beamforge
# target/release/beamforge
```

---

## Usage

Once running, a `YOU >` prompt appears.

- **Learning**: type a normal sentence. The model integrates it into its weights instantly.
- **Generation**: end your input with `?`. The model runs Beam Search and prints a response prefixed with `BEAMFORGE:`.

**Commands:**

| Command | Effect |
|---------|--------|
| `/train [folder]` | Ingest all `.txt` and `.md` files in a folder |
| `/save` | Save the current brain to `beamforge.brain` |
| `/load` | Reload a previously saved brain |
| `/quit` | Exit |

---

## Star History

<a href="https://www.star-history.com/?repos=infinition%2FBeamForge&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=infinition/BeamForge&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=infinition/BeamForge&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=infinition/BeamForge&type=date&legend=top-left" />
 </picture>
</a>

---

## License

MIT.
