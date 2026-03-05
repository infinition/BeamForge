# BeamForge: Tokenized Beam Search Cognitive Engine

[![Rust](https://img.shields.io/badge/Rust-1.70%2B-orange?logo=rust)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)]()

**BeamForge** (formerly NeoGenesis) is an experimental text generation engine coded in Rust from scratch. Unlike modern LLMs that use extremely demanding deep neural networks, BeamForge relies on a stochastic semantic graph coupled with a **Beam Search** algorithm.

It is designed to be ultra-fast, lightweight in memory (thanks to its native tokenization), and capable of **continuous real-time learning** (Online Learning).

## Key Features

* **Native Tokenization (`u32`)**: Raw text is converted into numeric identifiers. During generation, the model clones arrays of integers (`Vec<u32>`) rather than strings, ensuring maximum CPU and RAM performance.
* **Deep Thought (Beam Search)**: Instead of blindly choosing the next word (Greedy Search), the algorithm explores multiple scenarios in parallel (`BEAM_WIDTH = 5`) and keeps the sentence that makes the most sense globally.
* **Length Penalty**: Scenario scores are normalized by their length, preventing the model from generating infinite sentences to artificially inflate its score.
* **Live Online Learning**: Each declarative sentence typed by the user updates the synaptic weights in real-time.
* **Binary Persistence**: Ultra-fast saving and loading of the "brain" via binary serialization (`bincode`).

## Internal Architecture (Semantic Mesh)

The brain (`SemanticMesh`) relies on 3 main learning structures:

1. **Synapses (Trigrams)**: Link a strict context of 2 words to a 3rd target word (`w1 + w2 -> target`).
2. **Simple Links (Bigrams / Fallback)**: If the strict context is unknown, the model falls back on the probability of the previous word (`w2 -> target`).
3. **Fast-Forward**: Allows skipping concepts to restart generation if the user input is too short.

## Installation & Usage

### 1. Prerequisites (Rust)
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

### 2. Execution
From the root of the project, you can launch or compile the model as follows:

**Development Mode:**
```powershell
cargo run --bin beamforge
```

**Optimized Build (Release):**
```powershell
cargo build --release --bin beamforge
# The executable will be available in target/release/
```

## CLI Commands & Interaction

Once the program is running, an interactive prompt `YOU >` appears.

### Learning vs Generation Mode

* **Learning**: If you type a normal sentence (e.g., `The sky is blue this morning.`), the model integrates it instantly into its synaptic weights.
* **Generation**: If your sentence ends with a question mark `?` (e.g., `How is the sky ?`), the model launches the Beam Search and generates an answer (prefixed by `BEAMFORGE:`).

### System Commands

* `/train [folder]` : Scans a local folder and ingests all `.txt` and `.md` files to build a massive knowledge base quickly.
* *Example:* `/train ./data/books`
* `/save` : Compiles and saves the current state of the model to a `beamforge.brain` file.
* `/load` : Reloads a previously saved brain.
* `/quit` : Exits the program.

---

*Project developed for algorithmic exploration of optimized stochastic chain language generation.*
