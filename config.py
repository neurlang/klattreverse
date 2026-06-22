SAMPLE_RATE = 48000
NSPFR = 480

NUM_FRAMES = 11
NUM_FREQS = 836
NUM_CHANNELS = 2
NUM_PARAMS = 40

N_FFT = 1600
HOP_LENGTH = 200

# Each param index: (min, max) for uniform random generation
# Derived from GoDoc ranges + user constraint: max formant freq = 9000 Hz
PARAM_RANGES = [
    (0, 2000),     # 0:  F0hz10   (F0*10, 0=unvoiced)
    (0, 77),       # 1:  AVdb     (7 subtracted internally, effective 0-70)
    (200, 9000),   # 2:  F1hz
    (40, 1000),    # 3:  B1hz
    (550, 9000),   # 4:  F2hz
    (40, 1000),    # 5:  B2hz
    (1200, 9000),  # 6:  F3hz
    (40, 1000),    # 7:  B3hz
    (1200, 9000),  # 8:  F4hz
    (40, 1000),    # 9:  B4hz
    (1200, 9000),  # 10: F5hz
    (40, 1000),    # 11: B5hz
    (1200, 9000),  # 12: F6hz
    (40, 2000),    # 13: B6hz
    (0, 9000),     # 14: FNZhz   (0=disabled)
    (40, 1000),    # 15: BNZhz
    (200, 9000),   # 16: FNPhz
    (40, 1000),    # 17: BNPhz
    (0, 70),       # 18: ASP
    (10, 65),      # 19: Kopen   (hard-clamped by Klatt)
    (0, 80),       # 20: Aturb
    (0, 24),       # 21: TLTdb
    (0, 80),       # 22: AF
    (0, 40),       # 23: Kskew
    (0, 80),       # 24: A1
    (40, 1000),    # 25: B1phz
    (0, 80),       # 26: A2
    (40, 1000),    # 27: B2phz
    (0, 80),       # 28: A3
    (40, 1000),    # 29: B3phz
    (0, 80),       # 30: A4
    (40, 1000),    # 31: B4phz
    (0, 80),       # 32: A5
    (40, 1000),    # 33: B5phz
    (0, 80),       # 34: A6
    (40, 2000),    # 35: B6phz
    (0, 80),       # 36: ANP
    (0, 80),       # 37: AB
    (0, 70),       # 38: AVpdb
    (0, 60),       # 39: Gain0   (60 = unity)
]

# CNN input shape
TIME_COLS = 12
FREQ_BINS = 836
IN_CHANNELS = 2

# CNN output shape
OUT_FRAMES = 11
OUT_PARAMS = 40

TRAIN_EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

ACTIVE_PARAM_WEIGHTS = {
    0: 10.0,   # F0hz10
    1: 5.0,    # AVdb
    2: 3.0,    # F1hz
    3: 1.0,    # B1hz
    4: 3.0,    # F2hz
    5: 1.0,    # B2hz
    6: 2.0,    # F3hz
    7: 0.5,    # B3hz
    8: 1.0,    # F4hz
    9: 0.5,    # B4hz
    18: 0.5,   # ASP
    22: 0.5,   # AF
    21: 0.5,   # TLTdb
}
