import subprocess
import sys
import tempfile
import os
import time
import datetime
import numpy as np
import torch

from klatt.klatt import Klatt, KlattFrame, ALL_PARALLEL, NATURAL
from data.random_params import random_frame_batch

GO_KLATT = "goklatt/cmd/klatt/klatt"

PARAM_NAMES = [
    "F0hz10", "AVdb", "F1hz", "B1hz", "F2hz", "B2hz", "F3hz", "B3hz",
    "F4hz", "B4hz", "F5hz", "B5hz", "F6hz", "B6hz", "FNZhz", "BNZhz",
    "FNPhz", "BNPhz", "ASP", "Kopen", "Aturb", "TLTdb", "AF", "Kskew",
    "A1", "B1phz", "A2", "B2phz", "A3", "B3phz", "A4", "B4phz",
    "A5", "B5phz", "A6", "B6phz", "ANP", "AB", "AVpdb", "Gain0",
]


def tensor_to_frames(params):
    T = params.shape[0]
    frames = []
    for t in range(T):
        f = KlattFrame()
        for i in range(40):
            setattr(f, PARAM_NAMES[i], int(params[t, i].item()))
        frames.append(f)
    return frames


def synth_ref(frames):
    k = Klatt(samrate=48000)
    k.Nspfr = 480
    k.SynthesisModel = ALL_PARALLEL
    k.Glsource = NATURAL
    out = []
    for f in frames:
        audio = k.Parwave(f)
        out.append(audio)
    return np.concatenate(out).astype(np.int16)


def synth_go(frames):
    nframes = len(frames)
    lines = []
    for f in frames:
        row = " ".join(str(getattr(f, name)) for name in PARAM_NAMES)
        lines.append(row)
    text = "\n".join(lines) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".par", delete=False
    ) as par_f:
        par_f.write(text)
        par_path = par_f.name
    raw_path = par_path.replace(".par", ".raw")
    try:
        proc = subprocess.run(
            [GO_KLATT, "-i", par_path, "-o", raw_path, "-s", "48000", "-r", "2"],
            capture_output=True,
            timeout=30,
        )
    finally:
        os.unlink(par_path)
    if proc.returncode != 0:
        print(f"Go synth failed: {proc.stderr.decode()}", file=sys.stderr)
        return np.array([], dtype=np.int16)
    expected_bytes = nframes * 480 * 2
    file_bytes = os.path.getsize(raw_path)
    if file_bytes < expected_bytes:
        print(
            f"Go synth: expected {expected_bytes} bytes, got {file_bytes}",
            file=sys.stderr,
        )
        if os.path.exists(raw_path):
            os.unlink(raw_path)
        return np.array([], dtype=np.int16)
    samples = np.fromfile(raw_path, dtype=np.int16)
    os.unlink(raw_path)
    return samples


def compare(params):
    frames = tensor_to_frames(params)
    go = synth_go(frames)
    ref = synth_ref(frames)
    if len(ref) == 0 or len(go) == 0:
        return False, "empty output", ref, go
    min_len = min(len(ref), len(go))
    ref, go = ref[:min_len], go[:min_len]
    diff = np.abs(ref.astype(np.int32) - go.astype(np.int32))
    max_diff = int(diff.max())
    rms_diff = float(np.sqrt(np.mean(diff ** 2)))
    snr = (
        20 * np.log10(np.max(np.abs(ref.astype(np.float64))) / max(1e-6, rms_diff))
        if rms_diff > 1e-6
        else 60.0
    )
    passed = max_diff <= 100
    return passed, max_diff, ref, go


def overnight_run(log_path="compare_overnight.log", duration_hours=10):
    start = time.time()
    deadline = start + duration_hours * 3600
    total = 0
    n_pass = 0
    n_clip = 0
    n_bug = 0
    n_err = 0
    bug_samples = []
    last_summary = start

    with open(log_path, "w") as log:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log.write(f"{ts}  START overnight run (duration={duration_hours}h)\n")
        log.flush()

        while time.time() < deadline:
            try:
                seed = torch.randint(0, 2**31 - 1, (1,)).item()
                torch.manual_seed(seed)
                params = random_frame_batch(1, 11).squeeze(0)
                t0 = time.time()
                passed, max_diff, ref, go = compare(params)
                dt = time.time() - t0
                total += 1

                ref_max = int(ref.max()) if len(ref) else 0
                go_max = int(go.max()) if len(go) else 0
                ref_min = int(ref.min()) if len(ref) else 0
                go_min = int(go.min()) if len(go) else 0

                if passed:
                    n_pass += 1
                    tag = "PASS"
                elif abs(ref_max) >= 32767 or abs(ref_min) >= 32767 or abs(go_max) >= 32767 or abs(go_min) >= 32767:
                    n_clip += 1
                    tag = "CLIP"
                else:
                    n_bug += 1
                    tag = "BUG"
                    if len(bug_samples) < 20:
                        bug_samples.append({
                            "iter": total,
                            "seed": seed,
                            "max_diff": max_diff,
                            "ref_max": ref_max,
                            "go_max": go_max,
                            "ref_min": ref_min,
                            "go_min": go_min,
                            "params": params.clone(),
                        })

                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if tag != "PASS":
                    log.write(f"{ts}  iter={total:<7d} {tag:4s}  max_diff={max_diff:<5d}  "
                              f"ref=[{ref_min},{ref_max}]  go=[{go_min},{go_max}]  "
                              f"seed={seed}  {dt:.3f}s\n")

                if total % 1000 == 0:
                    elapsed = time.time() - start
                    rate = total / elapsed if elapsed > 0 else 0
                    remaining = deadline - time.time()
                    log.write(f"{ts}  --- SUMMARY  iters={total}  pass={n_pass}  "
                              f"clip={n_clip}  bug={n_bug}  err={n_err}  "
                              f"rate={rate:.1f}/s  remaining={remaining:.0f}s ---\n")

                log.flush()

            except Exception as e:
                n_err += 1
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log.write(f"{ts}  ERROR iter={total}: {e}\n")
                log.flush()

        elapsed = time.time() - start
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log.write(f"{ts}  FINAL  duration={elapsed:.0f}s  iters={total}  "
                  f"pass={n_pass}  clip={n_clip}  bug={n_bug}  err={n_err}\n")
        log.write(f"{ts}  BUG EXAMPLES ({len(bug_samples)} saved):\n")
        for i, s in enumerate(bug_samples):
            f0 = int(s["params"][:, 0].float().mean().item())
            av = int(s["params"][:, 1].float().mean().item())
            g0 = int(s["params"][:, 39].float().mean().item())
            log.write(f"{ts}    bug[{i}]: iter={s['iter']} seed={s['seed']}  max_diff={s['max_diff']}  "
                      f"ref=[{s['ref_min']},{s['ref_max']}]  go=[{s['go_min']},{s['go_max']}]  "
                      f"avg_F0hz10={f0}  avg_AVdb={av}  avg_Gain0={g0}\n")
        log.flush()


if __name__ == "__main__":
    if "--overnight" in sys.argv:
        log_path = os.path.join(os.path.dirname(__file__) or ".", "..", "compare_overnight.log")
        log_path = os.path.abspath(log_path)
        overnight_run(log_path=log_path)
    else:
        print("Usage: python3 -m data.compare_klatts --overnight", file=sys.stderr)
