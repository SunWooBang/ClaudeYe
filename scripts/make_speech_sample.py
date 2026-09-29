"""samples/demo_speech.mp4 = samples/demo.mp4 + Korean narration (Windows SAPI TTS) at known times.
Windows only (uses the built-in 'Microsoft Heami' voice via PowerShell)."""

from __future__ import annotations

import json
import subprocess
import tempfile
import wave
from pathlib import Path

import av
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
SR = 16000
LINES = [
    (1.0, "첫 번째 장면입니다. 파란 화면에서 흰 공이 오른쪽으로 움직입니다."),
    (11.2, "설정 화면에서 서버 주소를 입력합니다."),
    (16.2, "연결 버튼을 누르면 오류 코드 E 503이 표시됩니다."),
]


def tts(text: str, out: Path) -> None:
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "$v=$s.GetInstalledVoices()|?{$_.VoiceInfo.Culture.Name -eq 'ko-KR'}|select -First 1;"
        "if($v){$s.SelectVoice($v.VoiceInfo.Name)};"
        "$f=New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,"
        "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,[System.Speech.AudioFormat.AudioChannel]::Mono);"
        f"$s.SetOutputToWaveFile('{out}',$f);$s.Speak($env:TTS_TEXT);$s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True,
                   env={**__import__("os").environ, "TTS_TEXT": text})


def main() -> None:
    src = SAMPLES / "demo.mp4"
    dst = SAMPLES / "demo_speech.mp4"
    with av.open(str(src)) as c:
        dur = float(c.duration / 1_000_000)
    track = np.zeros(int(dur * SR), dtype=np.int16)
    spoken = []
    with tempfile.TemporaryDirectory() as td:
        for i, (t, text) in enumerate(LINES):
            wav = Path(td) / f"l{i}.wav"
            tts(text, wav)
            with wave.open(str(wav)) as w:
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            a = int(t * SR)
            b = min(len(track), a + len(pcm))
            track[a:b] = pcm[: b - a]
            spoken.append({"t": t, "end": round(t + len(pcm) / SR, 2), "text": text})

    inp = av.open(str(src))
    out = av.open(str(dst), "w")
    vin = inp.streams.video[0]
    vout = out.add_stream_from_template(vin)
    aout = out.add_stream("aac", rate=SR, layout="mono")
    for pkt in inp.demux(vin):
        if pkt.dts is None:
            continue
        pkt.stream = vout
        out.mux(pkt)
    frame_size = 1024
    for k in range(0, len(track), frame_size):
        chunk = track[k:k + frame_size]
        fr = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
        fr.sample_rate = SR
        fr.pts = k
        for p in aout.encode(fr):
            out.mux(p)
    for p in aout.encode():
        out.mux(p)
    out.close()
    inp.close()
    truth_path = SAMPLES / "truth.json"
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    truth["speech"] = {"path": "samples/demo_speech.mp4", "lines": spoken}
    truth_path.write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", dst, spoken)


if __name__ == "__main__":
    main()
