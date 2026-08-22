# Local changes

- [PR #1](https://github.com/antirez/h3.c/pull/1) — `2373d6eb7d4987afc81400d1fe8fe84eb37879a3`
- [PR #4](https://github.com/antirez/h3.c/pull/4) — `a2f2982ea893ca130828d96622025183c7555bf1`
- [PR #7](https://github.com/antirez/h3.c/pull/7) — `b73a02bb0006d953b6e6659821cd0f85403ea18a`
- [PR #14](https://github.com/antirez/h3.c/pull/14) — `733ca94dd07abcd739d73e2e574df0487c3d5976`, `16b623555c3af37bf3951e978e4a366bdf4009b9`, `784d1d712a64feea0b792086add9a597590032ef`, `d5948beea221009b9b11c24136394f7348ae0eba`, `bb89c68e1ba8df4861149982c6ce9dcd75b93d1a`
- [PR #34](https://github.com/antirez/h3.c/pull/34) — `281d15b4a1ff4ad84ff1aaf57dce9b92a34c7db5`
- [PR #35](https://github.com/antirez/h3.c/pull/35) — `caddf2f063db9aeed8efc45ed00862c1436f2557`
- [PR #44](https://github.com/antirez/h3.c/pull/44) — `1e0bdcbf10c4ee981ffee96d929c2e2160c8e3a4`

## Local performance and diagnostics

- GPU-resident RES sampler — `5594400`
- Configurable slow/lossless encoding diagnostics — `772a3a7`

## Local behavior

- Added an explicit `--sampler res|euler` selector and made RES the local
  default after the paired Segment B review. Euler remains available for live
  denoising previews and denoiser reuse; RES requires `--reuse 1`.
