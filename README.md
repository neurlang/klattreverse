# klattreverse

CNN to invert the Klatt speech synthesizer: given a phase-preserving spectrogram
(12×836×2), recover the 40 Klatt control parameters per frame (11 frames × 10ms).

## Project Structure

```
klattreverse/
├── README.md
├── goklatt/ -> ../goklatt          # Symlink to reference Klatt synth
├── klatt/
│   └── klatt.py                     # Copied from goklatt/klatt.py
├── model/
│   ├── __init__.py
│   ├── cnn.py                       # KlattReverseCNN
│   └── loss.py                      # Weighted MSE with silence masking
├── config.py                        # Hyperparameters
├── train.py                         # Training loop
├── eval.py                          # Inference from audio
└── requirements.txt
```

## CNN

```
Input:  (batch, 12, 836, 2)   12 time cols, 836 freq bins, 2 channels (real/imag)
Output: (batch, 11, 40)       11 Klatt frames, 40 params each
```

### Open Design Questions

1. **Architecture** — 1D convs over frequency axis vs full 2D convs over (time×freq)? Framework preference (PyTorch assumed)?

2. **Data generation** — Random sample-and-hold ramps between valid ranges? Specific trajectories like vowels or CV transitions?

3. **Real speech compatibility** — Run on real speech (must handle phase mismatch between Klatt-synthetic and natural audio) or only Klatt-generated?

4. **Output range** — Raw unbounded regression or squash through sigmoid/tanh to valid param ranges?

## Loss

Weighted MSE with silence masking. Silence = AVdb=ASP=AF=0 (zero contribution).

Per-parameter weights (vibed, will be tuned):
F0:10, AV:5, F1:3, B1:1, F2:3, B2:1, F3:2, B3:0.5, F4:1, B4:0.5, ASP:0.5, AF:0.5, TLT:0.5, others:0.1

## Quickstart

```bash
pip install -r requirements.txt
python train.py
python eval.py path/to/speech.wav
```

## References

- Klatt synth: `goklatt/klatt/klatt.go` (Go), `goklatt/klatt.py` (Python)
- Phase spectrogram: `pip install phase-spectrogram`
