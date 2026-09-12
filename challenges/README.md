# Challenges

All four challenge programs keep their existing executable names and algorithms.
Build them with `cargo build --release --locked --features challenge-tools` and
select a target with `--bin NAME`:

| Executable/source | Purpose |
| --- | --- |
| `block_cipher` / `block_cipher.rs` | Block-cipher comparison and randomness experiments; retains AES and entropy tools. |
| `point_function` / `point_function.rs` | Point-function challenge construction, including the historical compression workflow. |
| `poly_canon` / `poly_canon.rs` | Polynomial graph-canonicalization comparison. |
| `newton_feistal` / `newton_feistal.rs` | Newton/Feistel construction experiments. The existing spelling is retained. |

`challenge-tools` enables the comparison support required by these programs;
ordinary GSS does not compile those implementations. Existing challenge data and
outputs retain their local paths and are not moved by the source cleanup.
