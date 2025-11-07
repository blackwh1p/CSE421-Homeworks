# host_stream_wav.py — stream WAVs from ./recordings and save exactly-what-was-sent
import os, time, struct, csv, sys
import numpy as np
from math import gcd
from scipy.io import wavfile
from scipy.signal import resample_poly
import serial

# ---- make prints show up immediately ----
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
def p(*a, **kw):
    kw.setdefault("flush", True)
    print(*a, **kw)

# ---- config (paths are relative to this .py file) ----
SER_PORT   = "COM6"          # change if needed (e.g., "/dev/ttyACM0")
BAUD       = 115200
IN_FOLDER  = "./recordings"  # <-- your source folder with WAVs
FRAME_LEN  = 1024
FS_TARGET  = 8000            # 8 kHz MFCC pipeline
OUT_FRAMES = "./sent_frames" # per-frame raw dumps go here

# ---- helpers ----
def to_int16_any_dtype(x):
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.floating):
        m = float(np.max(np.abs(x))) if x.size else 1.0
        if m == 0.0: m = 1.0
        x = (x / m).astype(np.float32)
        return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
    if x.dtype == np.int32:
        x = np.right_shift(x, 16)
        x = np.clip(x, -32768, 32767).astype(np.int16)
        return x
    if x.dtype == np.int16:
        return x
    x = x.astype(np.float32)
    m = float(np.max(np.abs(x))) if x.size else 1.0
    if m == 0.0: m = 1.0
    x = x / m
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)

def load_wav_8k_mono(path, fs_target=FS_TARGET):
    fs, d = wavfile.read(path)
    if d.ndim > 1:
        d = d[:, 0]
    if fs != fs_target:
        g = gcd(fs, fs_target)
        d = resample_poly(d.astype(np.float32), fs_target // g, fs // g)
    return to_int16_any_dtype(d)

def frames(x, N, step=None):
    if step is None: step = N
    for i in range(0, len(x) - N + 1, step):
        yield i // step, x[i:i+N]

def rms(x):
    x = x.astype(np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0

def main():
    os.makedirs(OUT_FRAMES, exist_ok=True)
    os.makedirs(IN_FOLDER, exist_ok=True)

    # open serial
    p(f"[HOST] Opening serial {SER_PORT} @ {BAUD} ...")
    ser = serial.Serial(SER_PORT, BAUD, timeout=2, write_timeout=2)
    time.sleep(1.0)
    p(f"[HOST] Connected: {SER_PORT} @ {BAUD}")

    # list input wavs from ./recordings
    wavs = [os.path.join(IN_FOLDER, f) for f in os.listdir(IN_FOLDER) if f.lower().endswith(".wav")]
    wavs.sort()
    if not wavs:
        p(f"[HOST] No WAVs in {IN_FOLDER}")
        return

    # open logs
    mfcc_csv = open("mfcc_dump.csv", "w", newline="")
    mfcc_writer = csv.writer(mfcc_csv)
    mfcc_writer.writerow(["file", "frame"] + [f"c{i}" for i in range(13)])

    idx_csv = open("sent_frames_index.csv", "w", newline="")
    idx_writer = csv.writer(idx_csv)
    idx_writer.writerow(["file", "frame", "nsamples", "min", "max", "mean", "rms"])

    for w in wavs:
        base = os.path.basename(w)
        stem, ext = os.path.splitext(base)
        p(f"[HOST] Streaming: {base}")

        # prepare data we actually send
        d16 = load_wav_8k_mono(w, FS_TARGET)

        # save the exact audio stream we’ll send (whole file) into current py folder
        sent_wav_path = f"sent_{stem}__8k16.wav"
        wavfile.write(sent_wav_path, FS_TARGET, d16)
        p(f"        wrote: {sent_wav_path}")

        # stream in non-overlapping frames
        for fidx, fr in frames(d16, FRAME_LEN, step=FRAME_LEN):
            # log stats & save per-frame raw bytes (exact payload after header)
            idx_writer.writerow([base, fidx, len(fr), int(fr.min()), int(fr.max()), float(fr.mean()), rms(fr)])

            # also save each frame as raw s16le in ./sent_frames
            raw_name = os.path.join(OUT_FRAMES, f"sent_{stem}__frame{fidx:05d}.s16le")
            with open(raw_name, "wb") as fh:
                fh.write(fr.tobytes())

            # packetize and send
            pkt = b'W' + struct.pack('<H', FRAME_LEN) + fr.tobytes()
            ser.write(pkt)

            # read one MFCC line back (CSV)
            line = ser.readline().decode(errors='ignore').strip()
            if line:
                # keep the line and store to CSV (truncate/pad to 13 coeffs)
                try:
                    vals = [float(x) for x in line.split(",")]
                except Exception:
                    vals = []
                vals = (vals + [np.nan]*13)[:13]
                mfcc_writer.writerow([base, fidx] + vals)

    mfcc_csv.close()
    idx_csv.close()
    ser.close()

    p("[HOST] Saved:")
    p("  - mfcc_dump.csv                (MCU MFCC lines)")
    p("  - sent_frames_index.csv        (stats per frame)")
    p("  - sent_<orig>__8k16.wav        (exact stream audio per file)")
    p("  - sent_frames/sent_<orig>__frameNNNNN.s16le  (raw frames sent)")
    p("[HOST] Done.")

if __name__ == "_main_":
    main()