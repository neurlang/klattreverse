import torch
import torch.nn as nn
import math

B0 = [
    1200, 1142, 1088, 1038, 991, 948, 907, 869, 833, 799, 768, 738, 710, 683, 658,
    634, 612, 590, 570, 551, 533, 515, 499, 483, 468, 454, 440, 427, 415, 403,
    391, 380, 370, 360, 350, 341, 332, 323, 315, 307, 300, 292, 285, 278, 272,
    265, 259, 253, 247, 242, 237, 231, 226, 221, 217, 212, 208, 204, 199, 195,
    192, 188, 184, 180, 177, 174, 170, 167, 164, 161, 158, 155, 153, 150, 147,
    145, 142, 140, 137, 135, 133, 131, 128, 126, 124, 122, 120, 119, 117, 115,
    113, 111, 110, 108, 106, 105, 103, 102, 100, 99, 97, 96, 95, 93, 92, 91, 90,
    88, 87, 86, 85, 84, 83, 82, 80, 79, 78, 77, 76, 75, 75, 74, 73, 72, 71,
    70, 69, 68, 68, 67, 66, 65, 64, 64, 63, 62, 61, 61, 60, 59, 59, 58, 57,
    57, 56, 56, 55, 55, 54, 54, 53, 53, 52, 52, 51, 51, 50, 50, 49, 49, 48, 48,
    47, 47, 46, 46, 45, 45, 44, 44, 43, 43, 42, 42, 41, 41, 41, 41, 40, 40,
    39, 39, 38, 38, 38, 38, 37, 37, 36, 36, 36, 36, 35, 35, 35, 35, 34, 34, 33,
    33, 33, 33, 32, 32, 32, 32, 31, 31, 31, 31, 30, 30, 30, 30, 29, 29, 29, 29,
    28, 28, 28, 28, 27, 27,
]

AMP_TABLE = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 7.0,
    8.0, 9.0, 10.0, 11.0, 13.0, 14.0, 16.0, 18.0, 20.0, 22.0, 25.0, 28.0, 32.0,
    35.0, 40.0, 45.0, 51.0, 57.0, 64.0, 71.0, 80.0, 90.0, 101.0, 114.0, 128.0,
    142.0, 159.0, 179.0, 202.0, 227.0, 256.0, 284.0, 318.0, 359.0, 405.0,
    455.0, 512.0, 568.0, 638.0, 719.0, 811.0, 911.0, 1024.0, 1137.0, 1276.0,
    1438.0, 1622.0, 1823.0, 2048.0, 2273.0, 2552.0, 2875.0, 3244.0, 3645.0,
    4096.0, 4547.0, 5104.0, 5751.0, 6488.0, 7291.0, 8192.0, 9093.0, 10207.0,
    11502.0, 12976.0, 14582.0, 16384.0, 18350.0, 20644.0, 23429.0,
    26214.0, 29491.0, 32767,
]

CASCADE_PARALLEL = 0
ALL_PARALLEL = 1
IMPULSIVE = 0
NATURAL = 1
SAMPLED = 2


class DiffKlattWave(nn.Module):
    def __init__(self, samrate=48000):
        super().__init__()
        self.samrate = samrate
        self.FLPhz = (950 * samrate) // 10000
        self.BLPhz = (630 * samrate) // 10000

        self.register_buffer('B0_tbl', torch.tensor(B0, dtype=torch.float32))
        self.register_buffer('amp_tbl', torch.tensor(AMP_TABLE, dtype=torch.float32))
        self.register_buffer('_FLPhz_t', torch.tensor(self.FLPhz, dtype=torch.float32))
        self.register_buffer('_BLPhz_t', torch.tensor(self.BLPhz, dtype=torch.float32))

    def _db_to_lin(self, dB):
        dB = torch.clamp(dB, 0.0, 87.0)
        idx = torch.floor(dB).long()
        frac = dB - idx.float()
        idx_next = torch.clamp(idx + 1, 0, self.amp_tbl.shape[0] - 1)
        lo = torch.index_select(self.amp_tbl, 0, idx)
        hi = torch.index_select(self.amp_tbl, 0, idx_next)
        return (lo * (1.0 - frac) + hi * frac) * 0.001

    def _setabc_ret(self, f, bw):
        r = torch.exp(-math.pi / self.samrate * bw)
        C = -(r * r)
        B = r * torch.cos(2.0 * math.pi / self.samrate * f) * 2.0
        A = 1.0 - B - C
        return A, B, C

    def _setzeroabc(self, f, bw):
        r = torch.exp(-math.pi / self.samrate * bw)
        C = -(r * r)
        B = r * torch.cos(2.0 * math.pi / self.samrate * (-f)) * 2.0
        A = 1.0 - B - C
        zf = (f != 0).float()
        invA = 1.0 / (A + 1e-8)
        A = zf * invA + (1 - zf) * A
        C = zf * (-C * invA) + (1 - zf) * C
        B = zf * (-B * invA) + (1 - zf) * B
        return A, B, C

    def forward(self, frames, nspfr, glsource=NATURAL, synthesis_model=ALL_PARALLEL, f0_flutter=0):
        batch, num_frames, _ = frames.shape
        total_samples = num_frames * nspfr
        device = frames.device

        z = torch.zeros(batch, device=device)
        rst = {}
        for name in ['rRnpp','rR1p','rR2p','rR3p','rR4p','rR5p','rR6p',
                     'rR1c','rR2c','rR3c','rR4c','rR5c','rR6c','rR7c','rR8c',
                     'rRnpc','rRnz','rRgl','rRlp','rRout']:
            rst[name] = [z.clone(), z.clone()]

        # persistent scalar state
        Nper = torch.zeros(batch, dtype=torch.long, device=device)
        nlast = torch.zeros(batch, device=device)
        vlast = torch.zeros(batch, device=device)
        glotlast = torch.zeros(batch, device=device)
        skew_sign = torch.full((batch,), 1.0, device=device)
        naturalVwave = torch.zeros(batch, device=device)
        PulseShapeA = torch.zeros(batch, device=device)
        PulseShapeB = torch.zeros(batch, device=device)

        # Initialize first-frame pitch-period state so first sample starts cleanly
        f0 = frames[:, 0, :]
        T0_cur = torch.full((batch,), 1, dtype=torch.long, device=device)
        AmpVoice_cur = torch.zeros(batch, device=device)
        Nmod_cur = torch.zeros(batch, dtype=torch.long, device=device)
        AmpBreth_cur = torch.zeros(batch, device=device)
        Nopen_cur = torch.zeros(batch, dtype=torch.long, device=device)
        Decay_cur = torch.zeros(batch, device=device)
        Onemd_cur = torch.ones(batch, device=device)
        rgl_A = torch.zeros(batch, device=device)
        rgl_B = torch.zeros(batch, device=device)
        rgl_C = torch.zeros(batch, device=device)
        rgl_A[:] = 1.0

        # -- pre-generate noise --
        raw_noise = (torch.randint(0, 16383, (batch, total_samples), device=device).float() - 8191)

        # -- precompute glottal LPF coefficients (constant) --
        rlp_A, rlp_B, rlp_C = self._setabc_ret(
            torch.full((batch,), self.FLPhz, device=device).float(),
            torch.full((batch,), self.BLPhz, device=device).float()
        )

        out_samples = []
        frame_cache = {}

        for ns in range(total_samples):
            frame_idx = ns // nspfr
            ns_in_frame = ns % nspfr
            first_ever = (ns == 0)

            # -- frameInit at start of each frame --
            if ns_in_frame == 0:
                f = frames[:, frame_idx, :]
                fc = {}

                AVdb = f[:, 1] - 7
                AVdb = torch.clamp(AVdb, min=0)
                fc['AVdb'] = AVdb
                fc['AmpAspir'] = self._db_to_lin(f[:, 18]) * 0.05
                fc['AmpFrica'] = self._db_to_lin(f[:, 22]) * 0.25
                fc['ParAmpVoice'] = self._db_to_lin(f[:, 38])
                fc['AmpBypas'] = self._db_to_lin(f[:, 37]) * 0.05
                G0 = f[:, 39] - 3
                G0 = torch.where(G0 <= 0, torch.tensor(57.0, device=device), G0)
                fc['AmpGain0'] = self._db_to_lin(G0)

                # cascade
                fc['r1c'] = self._setabc_ret(f[:, 2], f[:, 3])
                fc['r2c'] = self._setabc_ret(f[:, 4], f[:, 5])
                fc['r3c'] = self._setabc_ret(f[:, 6], f[:, 7])
                fc['r4c'] = self._setabc_ret(f[:, 8], f[:, 9])
                fc['r5c'] = self._setabc_ret(f[:, 10], f[:, 11])
                fc['r6c'] = self._setabc_ret(f[:, 12], f[:, 13])
                fc['rnpc'] = self._setabc_ret(f[:, 16], f[:, 17])
                fc['rnz'] = self._setzeroabc(f[:, 14], f[:, 15])

                # parallel (amplitude-scaled)
                r1pA, r1pB, r1pC = self._setabc_ret(f[:, 2], f[:, 25])
                r1pA = r1pA * self._db_to_lin(f[:, 24]) * 0.4
                fc['r1p'] = (r1pA, r1pB, r1pC)

                rnppA, rnppB, rnppC = self._setabc_ret(f[:, 16], f[:, 17])
                rnppA = rnppA * self._db_to_lin(f[:, 36]) * 0.6
                fc['rnpp'] = (rnppA, rnppB, rnppC)

                r2pA, r2pB, r2pC = self._setabc_ret(f[:, 4], f[:, 27])
                r2pA = r2pA * self._db_to_lin(f[:, 26]) * 0.15
                fc['r2p'] = (r2pA, r2pB, r2pC)

                r3pA, r3pB, r3pC = self._setabc_ret(f[:, 6], f[:, 29])
                r3pA = r3pA * self._db_to_lin(f[:, 28]) * 0.06
                fc['r3p'] = (r3pA, r3pB, r3pC)

                r4pA, r4pB, r4pC = self._setabc_ret(f[:, 8], f[:, 31])
                r4pA = r4pA * self._db_to_lin(f[:, 30]) * 0.04
                fc['r4p'] = (r4pA, r4pB, r4pC)

                r5pA, r5pB, r5pC = self._setabc_ret(f[:, 10], f[:, 33])
                r5pA = r5pA * self._db_to_lin(f[:, 32]) * 0.022
                fc['r5p'] = (r5pA, r5pB, r5pC)

                r6pA, r6pB, r6pC = self._setabc_ret(f[:, 12], f[:, 35])
                r6pA = r6pA * self._db_to_lin(f[:, 34]) * 0.03
                fc['r6p'] = (r6pA, r6pB, r6pC)

                fc['rout'] = self._setabc_ret(
                    torch.zeros(batch, device=device),
                    torch.full((batch,), self.samrate // 2, device=device).float()
                )

                # F0hz10 (with optional flutter)
                if f0_flutter != 0:
                    raise NotImplementedError("F0 flutter not implemented in diff version")
                fc['F0hz10'] = f[:, 0].float()

                # pitch-synch helpers (recomputed once per frame; conditional reset picks them up)
                fc['Kopen'] = f[:, 19]
                fc['Kskew'] = f[:, 23]
                fc['TLTdb'] = f[:, 21]
                fc['Aturb'] = f[:, 20]

                frame_cache = fc

            # -- noise --
            nrand = raw_noise[:, ns]
            nlast = nrand + 0.75 * nlast
            noise_sample = nlast

            frics = frame_cache['AmpFrica'] * noise_sample

            # -- 4 voice sub-iterations --
            for _ in range(4):
                # ---- voice source ----
                if glsource == IMPULSIVE:
                    m0 = (Nper == 0).float()
                    m1 = (Nper == 1).float()
                    m2 = (Nper == 2).float()
                    vwave = m1 * 13000000.0 - m2 * 13000000.0
                elif glsource == NATURAL:
                    open_mask = (Nper < Nopen_cur).float()
                    PulseShapeA = PulseShapeA - open_mask * PulseShapeB
                    nat_v = naturalVwave + PulseShapeA
                    naturalVwave = open_mask * nat_v
                    vwave = naturalVwave * 0.028
                else:
                    vwave = torch.zeros(batch, device=device)

                # ---- glottal resonator (impulsive / sampled only) ----
                if glsource == IMPULSIVE:
                    voice = self._res(rst['rRgl'], rgl_A, rgl_B, rgl_C, vwave)
                elif glsource == NATURAL:
                    voice = vwave
                else:
                    voice = self._res(rst['rRgl'], rgl_A, rgl_B, rgl_C, vwave)

                # ---- pitch synch reset (compute reset values) ----
                fc = frame_cache
                F0hz10 = fc['F0hz10']

                T0_new = (40 * self.samrate) / F0hz10.clamp(min=1)
                T0_long = T0_new.long()

                voiced = (F0hz10 > 0).float()

                AmpVoice_new = voiced * self._db_to_lin(fc['AVdb'])
                Nmod_new = torch.where((voiced > 0) & (fc['AVdb'] > 0),
                                       T0_long // 2, T0_long)
                AmpBreth_new = self._db_to_lin(fc['Aturb']) * 0.1

                Nopen_new = 4 * fc['Kopen'].long()
                if glsource == IMPULSIVE:
                    Nopen_new = torch.where(Nopen_new > 263,
                                            torch.tensor(263, device=device), Nopen_new)
                Nopen_new = torch.where(Nopen_new >= (T0_long - 1),
                                        T0_long - 2, Nopen_new)
                Nopen_new = torch.where(Nopen_new < 40,
                                        torch.tensor(40, device=device), Nopen_new)

                idx = torch.clamp(Nopen_new - 40, 0, self.B0_tbl.shape[0] - 1)
                B0v = self.B0_tbl[idx]
                PulseShapeB_new = B0v.float()
                PulseShapeA_new = (PulseShapeB_new * Nopen_new.float()) * 0.333

                temp_sr = self.samrate // Nopen_new.clamp(min=1)
                rgl_A_new, rgl_B_new, rgl_C_new = self._setabc_ret(
                    torch.zeros(batch, device=device),
                    temp_sr.float()
                )
                temp1 = Nopen_new.float() * 0.00833
                rgl_A_new = rgl_A_new * temp1 * temp1

                Kskew = fc['Kskew'].long()
                Kskew_clamped = torch.where(Kskew > T0_long - Nopen_new,
                                            T0_long - Nopen_new, Kskew)
                skew_abs = skew_sign * Kskew_clamped.float()
                T0_skewed = (T0_new + skew_abs).long()
                skew_sign_new = -skew_abs

                TLTdb = fc['TLTdb'].float()
                Decay_new = 0.033 * TLTdb
                Onemd_new = torch.where(Decay_new > 0.0,
                                        1.0 - Decay_new, torch.ones_like(Decay_new))

                # ---- conditionally apply reset ----
                reset = (Nper >= T0_cur) | first_ever
                first_ever = False

                T0_cur = torch.where(reset, T0_skewed, T0_cur)
                AmpVoice_cur = torch.where(reset, AmpVoice_new, AmpVoice_cur)
                Nmod_cur = torch.where(reset, Nmod_new, Nmod_cur)
                AmpBreth_cur = torch.where(reset, AmpBreth_new, AmpBreth_cur)
                Nopen_cur = torch.where(reset, Nopen_new, Nopen_cur)
                PulseShapeA = torch.where(reset, PulseShapeA_new, PulseShapeA)
                PulseShapeB = torch.where(reset, PulseShapeB_new, PulseShapeB)
                rgl_A = torch.where(reset, rgl_A_new, rgl_A)
                rgl_B = torch.where(reset, rgl_B_new, rgl_B)
                rgl_C = torch.where(reset, rgl_C_new, rgl_C)
                Decay_cur = torch.where(reset, Decay_new, Decay_cur)
                Onemd_cur = torch.where(reset, Onemd_new, Onemd_cur)
                skew_sign = torch.where(reset, skew_sign_new, skew_sign)

                # reset Nper to 0 on reset, otherwise +1
                Nper = torch.where(reset, torch.zeros_like(Nper), Nper + 1)

                # ---- glottal LPF ----
                voice = self._res(rst['rRlp'], rlp_A, rlp_B, rlp_C, voice)

            # -- voice smoothing --
            voice = voice * Onemd_cur + vlast * Decay_cur
            vlast = voice

            # -- breathiness --
            breath_mask = (Nper < Nopen_cur).float()
            voice = voice + breath_mask * AmpBreth_cur * (nrand / 8191.0)

            glotout = AmpVoice_cur * voice
            parGlotout = frame_cache['ParAmpVoice'] * voice

            aspiration = frame_cache['AmpAspir'] * noise_sample
            glotout = glotout + aspiration
            parGlotout = parGlotout + aspiration

            # ---- cascade branch ----
            if synthesis_model != ALL_PARALLEL:
                rnz_out = self._antires(rst['rRnz'],
                    frame_cache['rnz'][0], frame_cache['rnz'][1], frame_cache['rnz'][2], glotout)
                casc_in = self._res(rst['rRnpc'],
                    frame_cache['rnpc'][0], frame_cache['rnpc'][1], frame_cache['rnpc'][2], rnz_out)
                casc_in = self._res(rst['rR6c'],
                    frame_cache['r6c'][0], frame_cache['r6c'][1], frame_cache['r6c'][2], casc_in)
                casc_in = self._res(rst['rR5c'],
                    frame_cache['r5c'][0], frame_cache['r5c'][1], frame_cache['r5c'][2], casc_in)
                casc_in = self._res(rst['rR4c'],
                    frame_cache['r4c'][0], frame_cache['r4c'][1], frame_cache['r4c'][2], casc_in)
                casc_in = self._res(rst['rR3c'],
                    frame_cache['r3c'][0], frame_cache['r3c'][1], frame_cache['r3c'][2], casc_in)
                casc_in = self._res(rst['rR2c'],
                    frame_cache['r2c'][0], frame_cache['r2c'][1], frame_cache['r2c'][2], casc_in)
                out_c = self._res(rst['rR1c'],
                    frame_cache['r1c'][0], frame_cache['r1c'][1], frame_cache['r1c'][2], casc_in)
            else:
                out_c = torch.zeros(batch, device=device)

            # ---- parallel branch ----
            r1p = frame_cache['r1p']
            out = self._res(rst['rR1p'], r1p[0], r1p[1], r1p[2], parGlotout)
            rnpp = frame_cache['rnpp']
            out = out + self._res(rst['rRnpp'], rnpp[0], rnpp[1], rnpp[2], parGlotout)

            sourc = frics + parGlotout - glotlast
            glotlast = parGlotout

            r6p = frame_cache['r6p']
            out = self._res(rst['rR6p'], r6p[0], r6p[1], r6p[2], sourc) - out
            r5p = frame_cache['r5p']
            out = self._res(rst['rR5p'], r5p[0], r5p[1], r5p[2], sourc) - out
            r4p = frame_cache['r4p']
            out = self._res(rst['rR4p'], r4p[0], r4p[1], r4p[2], sourc) - out
            r3p = frame_cache['r3p']
            out = self._res(rst['rR3p'], r3p[0], r3p[1], r3p[2], sourc) - out
            r2p = frame_cache['r2p']
            out = self._res(rst['rR2p'], r2p[0], r2p[1], r2p[2], sourc) - out
            out = frame_cache['AmpBypas'] * sourc - out

            if synthesis_model != ALL_PARALLEL:
                out = out + out_c

            rout = frame_cache['rout']
            out = self._res(rst['rRout'], rout[0], rout[1], rout[2], out)
            out = out * frame_cache['AmpGain0']

            out = torch.clamp(out, -32768.0, 32767.0)
            out_samples.append(out.unsqueeze(1))

        return torch.cat(out_samples, dim=1)

    def _res(self, st, A, B, C, x):
        p1, p2 = st
        out = A * x + B * p1 + C * p2
        st[0] = out
        st[1] = p1
        return out

    def _antires(self, st, A, B, C, x):
        p1, p2 = st
        out = A * x + B * p1 + C * p2
        st[0] = x
        st[1] = p1
        return out
