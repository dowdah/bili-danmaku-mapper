from pathlib import Path
import subprocess
import wave

import numpy as np
from scipy.signal import chirp


def sound(t, variant=0):
    t = np.asarray(t)
    result = chirp(t, 250 + variant * 151, 12, 2200 + variant * 271) * np.sin(np.pi * t * 1.7) ** 2
    for n in range(8):
        result += .3 * chirp(t, 500 + n * 113 + variant * 197, 12, 900 + n * 71 + variant * 127,
                            method="quadratic") * np.sin(np.pi * t * (1.1 + n * .13)) ** 2
    return result * .18


def mux_sound(video, output, duration, timeline=lambda t: t, *, rate=44100, gain=1,
              delay=0, mute=None, variant=0, metadata=None, offset=0):
    output = Path(output)
    wav = output.with_suffix(".wav")
    t = np.arange(round(duration * rate)) / rate
    original_time = np.asarray([timeline(float(v)) for v in t])
    signal = sound(original_time, variant) * gain
    if mute:
        signal[(t >= mute[0]) & (t < mute[1])] = 0
    with wave.open(str(wav), "wb") as file:
        file.setnchannels(1); file.setsampwidth(2); file.setframerate(rate)
        file.writeframes((np.clip(signal, -1, 1) * 32767).astype("<i2").tobytes())
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(video)]
    if delay:
        command += ["-itsoffset", str(delay)]
    command += ["-i", str(wav), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "96k"]
    if metadata:
        command += ["-metadata:s:a:0", f"language={metadata}"]
    if offset:
        command += ["-output_ts_offset", str(offset)]
    command += ["-shortest", str(output)]
    subprocess.run(command, check=True)
    return output
