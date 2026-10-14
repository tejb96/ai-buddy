"""Convert the LibriSpeech dummy parquet into 16 kHz PCM files + a JSON manifest.

Usage: python -I extract_librispeech.py <validation.parquet> <out_dir>
"""
import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
table = pq.read_table(src)
print(table.schema)
rows = table.to_pylist()
manifest = []
for i, row in enumerate(rows):
    audio = row["audio"]
    data = audio["bytes"]
    pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-ar", "16000", "-ac", "1", "-f", "s16le",
                          "pipe:1"], input=data, capture_output=True, check=True).stdout
    name = f"ls_{i:03d}.pcm"
    (out / name).write_bytes(pcm)
    manifest.append({"file": name, "text": row["text"], "seconds": round(len(pcm) / 32000, 2),
                     "speaker": row.get("speaker_id")})
(out / "manifest.json").write_text(json.dumps(manifest, indent=1))
print(len(manifest), "utterances,", round(sum(m["seconds"] for m in manifest)), "s total")
