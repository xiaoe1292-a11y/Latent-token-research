# Sources and attribution

- Official model and tokenizer: [hyp1231/Latte](https://github.com/hyp1231/Latte), fixed commit `05e4e6d983225bcb7172f148a076890e80c524d1`. This repository does not vendor the official model repository. Consult its license before redistributing upstream code.
- Dataset: Amazon Reviews 2023, Video_Games. Original interactions, user identifiers, item metadata and checkpoints are not bundled. Dataset use remains subject to its original terms.
- Semantic IDs use Sentence-T5-base representations, PCA and residual k-means. Model weights are not distributed here; their respective upstream licenses still apply.
- `reference/` contains this project's experimental harness and metric helpers, preserved for source review. These wrappers import the official model and external runtime dependencies; they are not a replacement for upstream Latte.
- `scripts/` provides CPU analysis and verification of the published study. The MIT license here covers this repository's contributed code, not third-party data or weights.

No claim is made that the underlying Latte architecture is original to this study, that these fixed-budget runs reproduce the paper's final performance, or that the local trends establish a Scaling Law.
