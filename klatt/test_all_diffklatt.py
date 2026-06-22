import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from klatt.klatt import parse_par_file, write_wav
from klatt.diffklatt import DiffKlattWave, NATURAL
import torch
import numpy as np

examples = ['example1.par', 'example2.par', 'example3.par']
nspfr = 480

for ex in examples:
    par_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f'../goklatt/cmd/klatt/{ex}')
    frames = parse_par_file(par_path)

    vecs = []
    for fr in frames:
        vec = []
        for attr in ['F0hz10','AVdb','F1hz','B1hz','F2hz','B2hz','F3hz','B3hz',
                     'F4hz','B4hz','F5hz','B5hz','F6hz','B6hz','FNZhz','BNZhz',
                     'FNPhz','BNPhz','ASP','Kopen','Aturb','TLTdb','AF','Kskew',
                     'A1','B1phz','A2','B2phz','A3','B3phz','A4','B4phz',
                     'A5','B5phz','A6','B6phz','ANP','AB','AVpdb','Gain0']:
            vec.append(getattr(fr, attr))
        vecs.append(vec)

    frames_t = torch.tensor(vecs, dtype=torch.float32).unsqueeze(0)

    model = DiffKlattWave(samrate=48000)
    model.eval()
    with torch.no_grad():
        audio = model(frames_t, nspfr, glsource=NATURAL)

    audio_np = audio.squeeze(0).numpy()
    max_abs = np.max(np.abs(audio_np))
    audio_int = (audio_np * (32767.0 / max_abs if max_abs > 0 else 1.0)).astype(np.int16)

    out_path = f'/tmp/opencode/klattcompare/{ex.replace(".par","")}_diff.wav'
    write_wav(out_path, audio_int, 48000)
    print(f'{ex}: frames={len(frames)}, output_len={len(audio_np)}, max_abs={max_abs:.1f} -> {out_path}')
