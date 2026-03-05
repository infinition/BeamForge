# README.main.md

## Fichier
- `main.rs`

## Version
NEOGENESIS V10 (Deep Thought Beam Search)

## Architecture
Structures:
- `synapses`: contexte `(w1, w2)` vers candidats suivants,
- `simple_links`: fallback `w2 -> next`,
- `fast_forward`: acceleration de demarrage,
- `reverse_index`: acces inverse vers contextes sources.

## Entrainement
- `learn(input)` met a jour les poids triades par gravite.
- `ingest_folder(path)` lit un corpus entier (`.txt`, `.md`) via `WalkDir`.

## Generation
- `generate(prompt)` prepare `(w1, w2)` de demarrage.
- `generate_beam()` explore un arbre de suites avec beam search.
- un bonus est donne aux hypotheses qui finissent sur ponctuation forte.

## Lancement
```powershell
cargo run --bin neogenesis_v10
```
