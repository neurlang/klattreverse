import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from klatt.klatt import parse_par_file, write_wav
from klatt.diffklatt import DiffKlattWave
import torch
import numpy as np

par_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '../goklatt/cmd/klatt/example1.par')
frames = parse_par_file(par_path)
nspfr = 480  # 10ms at 48kHz

frame_tensors = []
for fr in frames:
    vec = []
    for attr in ['F0hz10','AVdb','F1hz','B1hz','F2hz','B2hz','F3hz','B3hz',
                 'F4hz','B4hz','F5hz','B5hz','F6hz','B6hz','FNZhz','BNZhz',
                 'FNPhz','BNPhz','ASP','Kopen','Aturb','TLTdb','AF','Kskew',
                 'A1','B1phz','A2','B2phz','A3','B3phz','A4','B4phz',
                 'A5','B5phz','A6','B6phz','ANP','AB','AVpdb','Gain0']:
        vec.append(getattr(fr, attr))
    frame_tensors.append(vec)

frames_t = torch.tensor(frame_tensors, dtype=torch.float32).unsqueeze(0)
print(f"Input shape: {frames_t.shape}, num_frames={frames_t.shape[1]}")

model = DiffKlattWave(samrate=48000)
model.eval()

with torch.no_grad():
    audio = model(frames_t, nspfr, glsource=0)

print(f"Output shape: {audio.shape}")
print(f"Output min/max: {audio.min():.2f} / {audio.max():.2f}")

audio_np = audio.squeeze(0).numpy()
max_abs = np.max(np.abs(audio_np))
print(f"Max abs before int16 cast: {max_abs:.2f}")
audio_np = (audio_np * (32767.0 / max_abs if max_abs > 0 else 1.0)).astype(np.int16)
write_wav('/tmp/opencode/klattcompare/example1_diff.wav', audio_np, 48000)
print("Wrote diff output")
