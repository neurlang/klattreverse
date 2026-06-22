import math
import argparse
import struct
import numpy as np

from klatt.hashtron import Hash

NPAR = 40
MAX_SAM = 20000
TRUE = 1
FALSE = 0

CASCADE_PARALLEL = 0
ALL_PARALLEL = 1

IMPULSIVE = 0
NATURAL = 1
SAMPLED = 2


class Resonator:
    def __init__(self):
        self.A = 0.0
        self.B = 0.0
        self.C = 0.0
        self.P1 = 0.0
        self.P2 = 0.0


class KlattFrame:
    def __init__(self):
        self.F0hz10 = 0
        self.AVdb = 0
        self.F1hz = 0
        self.B1hz = 0
        self.F2hz = 0
        self.B2hz = 0
        self.F3hz = 0
        self.B3hz = 0
        self.F4hz = 0
        self.B4hz = 0
        self.F5hz = 0
        self.B5hz = 0
        self.F6hz = 0
        self.B6hz = 0
        self.FNZhz = 0
        self.BNZhz = 0
        self.FNPhz = 0
        self.BNPhz = 0
        self.ASP = 0
        self.Kopen = 0
        self.Aturb = 0
        self.TLTdb = 0
        self.AF = 0
        self.Kskew = 0
        self.A1 = 0
        self.B1phz = 0
        self.A2 = 0
        self.B2phz = 0
        self.A3 = 0
        self.B3phz = 0
        self.A4 = 0
        self.B4phz = 0
        self.A5 = 0
        self.B5phz = 0
        self.A6 = 0
        self.B6phz = 0
        self.ANP = 0
        self.AB = 0
        self.AVpdb = 0
        self.Gain0 = 0


class KlattVector:
    def __init__(self):
        self.data = [0] * NPAR

    def ToFrame(self):
        frame = KlattFrame()
        frame.F0hz10 = self.data[0]
        frame.AVdb = self.data[1]
        frame.F1hz = self.data[2]
        frame.B1hz = self.data[3]
        frame.F2hz = self.data[4]
        frame.B2hz = self.data[5]
        frame.F3hz = self.data[6]
        frame.B3hz = self.data[7]
        frame.F4hz = self.data[8]
        frame.B4hz = self.data[9]
        frame.F5hz = self.data[10]
        frame.B5hz = self.data[11]
        frame.F6hz = self.data[12]
        frame.B6hz = self.data[13]
        frame.FNZhz = self.data[14]
        frame.BNZhz = self.data[15]
        frame.FNPhz = self.data[16]
        frame.BNPhz = self.data[17]
        frame.ASP = self.data[18]
        frame.Kopen = self.data[19]
        frame.Aturb = self.data[20]
        frame.TLTdb = self.data[21]
        frame.AF = self.data[22]
        frame.Kskew = self.data[23]
        frame.A1 = self.data[24]
        frame.B1phz = self.data[25]
        frame.A2 = self.data[26]
        frame.B2phz = self.data[27]
        frame.A3 = self.data[28]
        frame.B3phz = self.data[29]
        frame.A4 = self.data[30]
        frame.B4phz = self.data[31]
        frame.A5 = self.data[32]
        frame.B5phz = self.data[33]
        frame.A6 = self.data[34]
        frame.B6phz = self.data[35]
        frame.ANP = self.data[36]
        frame.AB = self.data[37]
        frame.AVpdb = self.data[38]
        frame.Gain0 = self.data[39]
        return frame


class Klatt:
    def __init__(self, samrate=16000):
        self.SynthesisModel = CASCADE_PARALLEL
        self.samrate = samrate
        self.FLPhz = (950 * samrate) // 10000
        self.BLPhz = (630 * samrate) // 10000
        self.Nfcascade = 6
        self.Glsource = IMPULSIVE
        self.F0Flutter = 0
        self.QuietFlag = False
        self.Nspfr = 0
        self.Nper = 0
        self.Ns = 0
        self.T0 = 0
        self.Nopen = 0
        self.Nmod = 0
        self.Nrand = 0
        self.PulseShapeA = 0.0
        self.PulseShapeB = 0.0
        self.naturalVwave = 0.0
        self.Onemd = 0.0
        self.Decay = 0.0
        self.AmpBypas = 0.0
        self.AmpVoice = 0.0
        self.ParAmpVoice = 0.0
        self.AmpAspir = 0.0
        self.AmpFrica = 0.0
        self.AmpBreth = 0.0
        self.AmpGain0 = 0.0
        self.NumSamples = 0
        self.SampleFactor = 0.0
        self.NaturalSamples = []
        self.OriginalF0 = 0

        self.rRnpp = Resonator()
        self.rR1p = Resonator()
        self.rR2p = Resonator()
        self.rR3p = Resonator()
        self.rR4p = Resonator()
        self.rR5p = Resonator()
        self.rR6p = Resonator()
        self.rR1c = Resonator()
        self.rR2c = Resonator()
        self.rR3c = Resonator()
        self.rR4c = Resonator()
        self.rR5c = Resonator()
        self.rR6c = Resonator()
        self.rR7c = Resonator()
        self.rR8c = Resonator()
        self.rRnpc = Resonator()
        self.rRnz = Resonator()
        self.rRgl = Resonator()
        self.rRlp = Resonator()
        self.rRout = Resonator()

        self.timeCount = 0
        self.skew = 0
        self.nlast = 0.0
        self.glotlast = 0.0
        self.vlast = 0.0

        setabc(self.FLPhz, self.BLPhz, self.rRlp, self)

    def resonator(self, rp, input_val):
        x = rp.A * input_val + rp.B * rp.P1 + rp.C * rp.P2
        rp.P2 = rp.P1
        rp.P1 = x
        return x

    def antiresonator(self, rp, input_val):
        x = rp.A * input_val + rp.B * rp.P1 + rp.C * rp.P2
        rp.P2 = rp.P1
        rp.P1 = input_val
        return x

    def Parwave(self, frame):
        self.frameInit(frame)

        if self.F0Flutter != 0:
            self.flutter(frame)

        output = []

        for self.Ns in range(int(self.Nspfr)):
            noise = self.genNoise()

            if self.Nper > self.Nmod:
                noise *= 0.5

            frics = self.AmpFrica * noise

            for n4 in range(4):
                if self.Glsource == IMPULSIVE:
                    voice = self.impulsiveSource()
                elif self.Glsource == NATURAL:
                    voice = self.naturalSource()
                elif self.Glsource == SAMPLED:
                    voice = self.sampledSource()
                else:
                    voice = self.impulsiveSource()

                if self.Nper >= self.T0:
                    self.Nper = 0
                    self.pitchSynchParReset(frame)

                voice = self.resonator(self.rRlp, voice)
                self.Nper += 1

            voice = (voice * self.Onemd) + (self.vlast * self.Decay)
            self.vlast = voice

            if self.Nper < self.Nopen:
                voice += self.AmpBreth * float(self.Nrand)

            glotout = self.AmpVoice * voice
            parGlotout = self.ParAmpVoice * voice

            aspiration = self.AmpAspir * noise
            glotout += aspiration
            parGlotout += aspiration

            if self.SynthesisModel != ALL_PARALLEL:
                rnzout = self.antiresonator(self.rRnz, glotout)
                cascNextIn = self.resonator(self.rRnpc, rnzout)

                if self.Nfcascade >= 8:
                    cascNextIn = self.resonator(self.rR8c, cascNextIn)
                if self.Nfcascade >= 7:
                    cascNextIn = self.resonator(self.rR7c, cascNextIn)
                if self.Nfcascade >= 6:
                    cascNextIn = self.resonator(self.rR6c, cascNextIn)
                if self.Nfcascade >= 5:
                    cascNextIn = self.resonator(self.rR5c, cascNextIn)
                if self.Nfcascade >= 4:
                    cascNextIn = self.resonator(self.rR4c, cascNextIn)
                if self.Nfcascade >= 3:
                    cascNextIn = self.resonator(self.rR3c, cascNextIn)
                if self.Nfcascade >= 2:
                    cascNextIn = self.resonator(self.rR2c, cascNextIn)
                if self.Nfcascade >= 1:
                    out = self.resonator(self.rR1c, cascNextIn)

            out = self.resonator(self.rR1p, parGlotout)
            out += self.resonator(self.rRnpp, parGlotout)

            sourc = frics + parGlotout - self.glotlast
            self.glotlast = parGlotout

            out = self.resonator(self.rR6p, sourc) - out
            out = self.resonator(self.rR5p, sourc) - out
            out = self.resonator(self.rR4p, sourc) - out
            out = self.resonator(self.rR3p, sourc) - out
            out = self.resonator(self.rR2p, sourc) - out
            out = self.AmpBypas * sourc - out

            out = self.resonator(self.rRout, out)
            out = out * self.AmpGain0

            if out < -32768:
                out = -32768
            if out > 32767:
                out = 32767

            output.append(np.int16(out))

        return np.array(output, dtype=np.int16)

    def flutter(self, frame):
        deltaF0 = 0.0
        fla = float(self.F0Flutter) / 50
        flb = float(self.OriginalF0) / 100
        flc = math.sin(2 * math.pi * 12.7 * float(self.timeCount))
        fld = math.sin(2 * math.pi * 7.1 * float(self.timeCount))
        fle = math.sin(2 * math.pi * 4.7 * float(self.timeCount))
        deltaF0 = fla * flb * (flc + fld + fle) * 10
        frame.F0hz10 = frame.F0hz10 + int(deltaF0)
        self.timeCount += 1

    def sampledSource(self):
        result = 0.0
        if self.T0 != 0:
            ftemp = float(self.Nper)
            ftemp = ftemp / float(self.T0)
            ftemp = ftemp * float(self.NumSamples)
            itemp = int(ftemp)

            tempDiff = ftemp - float(itemp)

            currentValue = self.NaturalSamples[itemp]
            nextValue = self.NaturalSamples[itemp + 1]

            diffValue = float(nextValue) - float(currentValue)
            diffValue = diffValue * tempDiff

            result = float(self.NaturalSamples[itemp]) + diffValue
            result = result * self.SampleFactor
        return result

    def frameInit(self, frame):
        self.OriginalF0 = frame.F0hz10 // 10

        frame.AVdb = frame.AVdb - 7
        if frame.AVdb < 0:
            frame.AVdb = 0

        self.AmpAspir = self.dbToLin(frame.ASP) * 0.05
        self.AmpFrica = self.dbToLin(frame.AF) * 0.25
        self.ParAmpVoice = self.dbToLin(frame.AVpdb)
        self.AmpBypas = self.dbToLin(frame.AB) * 0.05
        frame.Gain0 = frame.Gain0 - 3
        if frame.Gain0 <= 0:
            frame.Gain0 = 57
        self.AmpGain0 = self.dbToLin(frame.Gain0)

        if self.Nfcascade >= 8:
            if self.samrate >= 16000:
                setabc(7500, 600, self.rR8c, self)
            else:
                self.Nfcascade = 6

        if self.Nfcascade >= 7:
            if self.samrate >= 16000:
                setabc(6500, 500, self.rR7c, self)
            else:
                self.Nfcascade = 6

        if self.Nfcascade >= 6:
            setabc(frame.F6hz, frame.B6hz, self.rR6c, self)
        if self.Nfcascade >= 5:
            setabc(frame.F5hz, frame.B5hz, self.rR5c, self)

        setabc(frame.F4hz, frame.B4hz, self.rR4c, self)
        setabc(frame.F3hz, frame.B3hz, self.rR3c, self)
        setabc(frame.F2hz, frame.B2hz, self.rR2c, self)
        setabc(frame.F1hz, frame.B1hz, self.rR1c, self)

        setabc(frame.FNPhz, frame.BNPhz, self.rRnpc, self)
        setzeroabc(frame.FNZhz, frame.BNZhz, self.rRnz, self)

        setabc(frame.F1hz, frame.B1phz, self.rR1p, self)
        self.rR1p.A *= self.dbToLin(frame.A1) * 0.4
        setabc(frame.FNPhz, frame.BNPhz, self.rRnpp, self)
        self.rRnpp.A *= self.dbToLin(frame.ANP) * 0.6
        setabc(frame.F2hz, frame.B2phz, self.rR2p, self)
        self.rR2p.A *= self.dbToLin(frame.A2) * 0.15
        setabc(frame.F3hz, frame.B3phz, self.rR3p, self)
        self.rR3p.A *= self.dbToLin(frame.A3) * 0.06
        setabc(frame.F4hz, frame.B4phz, self.rR4p, self)
        self.rR4p.A *= self.dbToLin(frame.A4) * 0.04
        setabc(frame.F5hz, frame.B5phz, self.rR5p, self)
        self.rR5p.A *= self.dbToLin(frame.A5) * 0.022
        setabc(frame.F6hz, frame.B6phz, self.rR6p, self)
        self.rR6p.A *= self.dbToLin(frame.A6) * 0.03

        setabc(0, self.samrate // 2, self.rRout, self)

    def impulsiveSource(self):
        doublet = [0.0, 13000000.0, -13000000.0]
        if self.Nper < 3:
            vwave = doublet[self.Nper]
        else:
            vwave = 0.0
        return self.resonator(self.rRgl, vwave)

    def naturalSource(self):
        if self.Nper < self.Nopen:
            self.PulseShapeA -= self.PulseShapeB
            self.naturalVwave += self.PulseShapeA
            return self.naturalVwave * 0.028
        self.naturalVwave = 0.0
        return 0.0

    def pitchSynchParReset(self, frame):
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
            28, 28, 28, 28, 27, 27
        ]

        if frame.F0hz10 > 0:
            self.T0 = (40 * self.samrate) // frame.F0hz10

            self.AmpVoice = self.dbToLin(frame.AVdb)

            self.Nmod = self.T0
            if frame.AVdb > 0:
                self.Nmod >>= 1

            self.AmpBreth = self.dbToLin(frame.Aturb) * 0.1

            self.Nopen = 4 * frame.Kopen

            if (self.Glsource == IMPULSIVE) and (self.Nopen > 263):
                self.Nopen = 263

            if self.Nopen >= (self.T0 - 1):
                self.Nopen = self.T0 - 2

            if self.Nopen < 40:
                self.Nopen = 40

            self.PulseShapeB = float(B0[self.Nopen - 40])
            self.PulseShapeA = (self.PulseShapeB * float(self.Nopen)) * 0.333

            temp = self.samrate // self.Nopen
            setabc(0, temp, self.rRgl, self)

            temp1 = float(self.Nopen) * 0.00833
            self.rRgl.A *= temp1 * temp1

            temp = self.T0 - self.Nopen
            if frame.Kskew > temp:
                frame.Kskew = temp

            if self.skew >= 0:
                self.skew = frame.Kskew
            else:
                self.skew = -frame.Kskew

            self.T0 = self.T0 + self.skew
            self.skew = -self.skew
        else:
            self.T0 = 4
            self.AmpVoice = 0.0
            self.Nmod = self.T0
            self.AmpBreth = 0.0
            self.PulseShapeA = 0.0
            self.PulseShapeB = 0.0

        if (self.T0 != 4) or (self.Ns == 0):
            self.Decay = 0.033 * float(frame.TLTdb)
            if self.Decay > 0.0:
                self.Onemd = 1.0 - self.Decay
            else:
                self.Onemd = 1.0

    def genNoise(self):
        temp = Hash.hash(self.Ns, 42, 16383)
        temp -= 8191
        self.Nrand = temp
        self.nlast = float(self.Nrand) + (0.75 * self.nlast)
        return self.nlast

    def dbToLin(self, dB):
        ampTable = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 7.0,
            8.0, 9.0, 10.0, 11.0, 13.0, 14.0, 16.0, 18.0, 20.0, 22.0, 25.0, 28.0, 32.0,
            35.0, 40.0, 45.0, 51.0, 57.0, 64.0, 71.0, 80.0, 90.0, 101.0, 114.0, 128.0,
            142.0, 159.0, 179.0, 202.0, 227.0, 256.0, 284.0, 318.0, 359.0, 405.0,
            455.0, 512.0, 568.0, 638.0, 719.0, 811.0, 911.0, 1024.0, 1137.0, 1276.0,
            1438.0, 1622.0, 1823.0, 2048.0, 2273.0, 2552.0, 2875.0, 3244.0, 3645.0,
            4096.0, 4547.0, 5104.0, 5751.0, 6488.0, 7291.0, 8192.0, 9093.0, 10207.0,
            11502.0, 12976.0, 14582.0, 16384.0, 18350.0, 20644.0, 23429.0,
            26214.0, 29491.0, 32767
        ]

        if dB < 0 or dB > 87:
            return 0.0

        lgtemp = ampTable[dB] * 0.001
        return lgtemp


def setabc(f, bw, rp, globals_obj):
    r = math.exp(-math.pi / float(globals_obj.samrate) * float(bw))
    rp.C = -(r * r)
    rp.B = r * math.cos(2.0 * math.pi / float(globals_obj.samrate) * float(f)) * 2.0
    rp.A = 1.0 - rp.B - rp.C


def setzeroabc(f, bw, rp, globals_obj):
    r = math.exp(-math.pi / float(globals_obj.samrate) * float(bw))
    rp.C = -(r * r)
    rp.B = r * math.cos(2.0 * math.pi / float(globals_obj.samrate) * float(-f)) * 2.0
    rp.A = 1.0 - rp.B - rp.C

    if f != 0:
        rp.A = 1.0 / rp.A
        rp.C *= -rp.A
        rp.B *= -rp.A


def parse_par_file(filename):
    frames = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < NPAR:
                continue
            vec = KlattVector()
            for i in range(NPAR):
                vec.data[i] = int(float(parts[i]))
            frames.append(vec.ToFrame())
    return frames


def write_wav(filename, samples, sample_rate):
    num_channels = 1
    bytes_per_sample = 2
    byte_rate = sample_rate * num_channels * bytes_per_sample
    block_align = num_channels * bytes_per_sample
    bits_per_sample = 8 * bytes_per_sample
    data_size = len(samples) * bytes_per_sample

    with open(filename, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + data_size))
        f.write(b'WAVE')

        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))
        f.write(struct.pack('<H', 1))
        f.write(struct.pack('<H', num_channels))
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', byte_rate))
        f.write(struct.pack('<H', block_align))
        f.write(struct.pack('<H', bits_per_sample))

        f.write(b'data')
        f.write(struct.pack('<I', data_size))

        for sample in samples:
            f.write(struct.pack('<h', sample))


def main():
    parser = argparse.ArgumentParser(description='Klatt synthesizer')
    parser.add_argument('-i', '--input', required=True, help='Input par file')
    parser.add_argument('-o', '--output', required=True, help='Output WAV file')
    parser.add_argument('--ms-per-frame', type=int, default=10, help='Milliseconds per frame (default: 10)')
    args = parser.parse_args()

    sample_rate = 48000
    frames = parse_par_file(args.input)

    klatt = Klatt(samrate=sample_rate)
    ms_per_frame = args.ms_per_frame
    klatt.Nspfr = (sample_rate * ms_per_frame) // 1000

    all_samples = []

    for frame in frames:
        samples = klatt.Parwave(frame)
        all_samples.append(samples)

    if all_samples:
        final_samples = np.concatenate(all_samples)
    else:
        final_samples = np.array([], dtype=np.int16)

    write_wav(args.output, final_samples, sample_rate)


if __name__ == '__main__':
    main()
