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
    (400, 2000),   # 0:  F0hz10   (below 400 → T0 > 4800, pitch reset never fires)
    (20, 70),      # 1:  AVdb     (7 subtracted internally, below 20 is near-silent)
    (200, 6000),   # 2:  F1hz     (capped at 6000 to avoid extreme resonator Q)
    (100, 800),    # 3:  B1hz     (raised min BW to avoid high-Q sensitivity)
    (550, 6000),   # 4:  F2hz
    (100, 800),    # 5:  B2hz
    (1200, 6000),  # 6:  F3hz
    (100, 800),    # 7:  B3hz
    (1200, 6000),  # 8:  F4hz
    (100, 800),    # 9:  B4hz
    (1200, 6000),  # 10: F5hz
    (100, 800),    # 11: B5hz
    (1200, 6000),  # 12: F6hz
    (100, 800),    # 13: B6hz
    (0, 6000),     # 14: FNZhz   (0=disabled)
    (100, 800),    # 15: BNZhz
    (200, 6000),   # 16: FNPhz
    (100, 800),    # 17: BNPhz
    (0, 70),       # 18: ASP
    (15, 55),      # 19: Kopen   (narrowed to avoid unnatural pulse shapes)
    (0, 80),       # 20: Aturb
    (0, 24),       # 21: TLTdb
    (0, 80),       # 22: AF
    (0, 40),       # 23: Kskew
    (10, 70),      # 24: A1      (raised min to avoid silent branches)
    (100, 800),    # 25: B1phz
    (10, 70),      # 26: A2
    (100, 800),    # 27: B2phz
    (10, 70),      # 28: A3
    (100, 800),    # 29: B3phz
    (10, 70),      # 30: A4
    (100, 800),    # 31: B4phz
    (10, 70),      # 32: A5
    (100, 800),    # 33: B5phz
    (10, 70),      # 34: A6
    (100, 800),    # 35: B6phz
    (10, 70),      # 36: ANP
    (10, 70),      # 37: AB
    (10, 65),      # 38: AVpdb   (raised min to avoid silent parallel voice)
    (16, 60),      # 39: Gain0   (below 16 → dbToLin returns 0)
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
