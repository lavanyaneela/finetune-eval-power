#!/usr/bin/env python3
"""Records a terminal demo as a replay of real captured output.

Every line in the recording is stdout from a command that actually ran, with its
real exit code, and each segment is paced by the wall time that command took. It is
a replay rather than a screen recording, and the README says so.

The segment order is the argument, not the feature list: what the suite reported,
then what it missed, then the refusal, then the receipts failing on a corrupted
figure. The last one is the point, so it goes last.

Durations are used to pace the playback and are written to the manifest. They are
not published anywhere, because a duration is a fact about the machine that measured
it. `tools/check_durations.py` enforces that over every tracked file.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "video"
COLUMNS = 104

SEGMENTS: tuple[tuple[str, list[str], int], ...] = (
    (
        "What the suite reported",
        [sys.executable, "-m", "finetune_eval_power.cli", "suite", "--arm", "vision"],
        18,
    ),
    (
        "What it missed",
        [sys.executable, "-m", "finetune_eval_power.cli", "blind", "--arm", "vision"],
        16,
    ),
    (
        "Can it see a two point regression?",
        [
            sys.executable,
            "-m",
            "finetune_eval_power.cli",
            "power",
            "--arm",
            "vision",
            "--claim",
            "2",
        ],
        14,
    ),
    ("And the receipts, with one figure corrupted", [sys.executable, "tools/check_numbers.py"], 8),
)


def capture(command: list[str]) -> dict[str, object]:
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - started
    output = (result.stdout + result.stderr).rstrip("\n")
    return {
        "command": " ".join(command).replace(sys.executable, "python"),
        "exit_code": result.returncode,
        "seconds": round(elapsed, 3),
        "lines": output.splitlines(),
    }


def corrupt_a_figure() -> tuple[Path, str]:
    """Break one published figure so the last segment shows the checker failing."""
    metrics = json.loads((ROOT / "docs" / "metrics.json").read_text(encoding="utf-8"))
    readme = ROOT / "README.md"
    original = readme.read_text(encoding="utf-8")
    for metric, phrases in metrics["anchors"].items():
        value = metrics["rendered"][metric]
        for phrase in phrases:
            quoted = phrase.replace("{}", value)
            if quoted in original:
                readme.write_text(
                    original.replace(quoted, phrase.replace("{}", "99999")), encoding="utf-8"
                )
                return readme, original
    raise SystemExit("no anchored figure found in the README to corrupt")


def build_player(segments: list[dict[str, object]]) -> str:
    payload = json.dumps(segments)
    return """<!doctype html><html><head><meta charset="utf-8"><style>
body{margin:0;background:#12141a;font-family:"SFMono-Regular",Menlo,Consolas,monospace}
#frame{padding:18px 22px}
.prompt{color:#7fd1b9}.cmd{color:#e8eaed}.out{color:#b9bec8;white-space:pre}
.label{color:#6c7380;margin:14px 0 6px;font-size:13px}
.code{color:#e06c75}
pre{margin:0;font-size:14px;line-height:1.45}
</style></head><body><div id="frame"></div><script>
const segments = __PAYLOAD__;
const frame = document.getElementById("frame");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function line(cls, text){
  const p = document.createElement("pre");
  p.className = cls; p.textContent = text; frame.appendChild(p); return p;
}
(async () => {
  for (const segment of segments) {
    const label = line("label", "# " + segment.label);
    await sleep(260);
    const typed = line("cmd", "");
    const text = "$ " + segment.command;
    for (let i = 0; i <= text.length; i++) {
      typed.textContent = text.slice(0, i); await sleep(11);
    }
    await sleep(180);
    const rows = Math.max(segment.lines.length, 1);
    const per = Math.max(14, Math.min(70, (segment.seconds * 1000) / rows));
    for (const row of segment.lines.slice(0, segment.limit)) {
      line("out", row); await sleep(per);
    }
    line("code", "exit code " + segment.exit_code);
    await sleep(520);
    if (label) { /* keep */ }
  }
  window.__done = true;
})();
</script></body></html>""".replace("__PAYLOAD__", payload)


def main() -> int:
    from playwright.sync_api import sync_playwright

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed, so no recording was produced")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Warm the import so the first segment does not pay for it and pace wrongly.
    subprocess.run([sys.executable, "-c", "import finetune_eval_power.cli"], cwd=ROOT, check=False)

    segments: list[dict[str, object]] = []
    for label, command, limit in SEGMENTS[:-1]:
        captured = capture(command)
        captured["label"] = label
        captured["limit"] = limit
        segments.append(captured)

    readme, original = corrupt_a_figure()
    try:
        label, command, limit = SEGMENTS[-1]
        captured = capture(command)
        captured["label"] = label
        captured["limit"] = limit
        segments.append(captured)
    finally:
        readme.write_text(original, encoding="utf-8")

    player = OUTPUT / "player.html"
    player.write_text(build_player(segments), encoding="utf-8")

    total = sum(float(segment["seconds"]) for segment in segments)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(
            viewport={"width": 980, "height": 620},
            record_video_dir=str(OUTPUT),
            record_video_size={"width": 980, "height": 620},
        )
        page = context.new_page()
        page.goto(f"file://{player.resolve()}")
        page.wait_for_function("() => window.__done === true", timeout=180_000)
        page.wait_for_timeout(1400)
        context.close()
        browser.close()

    recordings = sorted(OUTPUT.glob("*.webm"), key=lambda path: path.stat().st_mtime)
    if not recordings:
        raise SystemExit("playwright produced no recording")
    source = recordings[-1]

    palette = OUTPUT / "palette.png"
    gif = OUTPUT / "demo.gif"
    mp4 = OUTPUT / "demo.mp4"
    chain = "fps=9,scale=900:-1:flags=lanczos"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(source), "-vf", f"{chain},palettegen", str(palette)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-i",
            str(palette),
            "-lavfi",
            f"{chain} [x]; [x][1:v] paletteuse",
            str(gif),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-movflags",
            "faststart",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "scale=980:-2",
            str(mp4),
        ],
        check=True,
        capture_output=True,
    )
    palette.unlink(missing_ok=True)

    (OUTPUT / "manifest.json").write_text(
        json.dumps(
            {
                "note": (
                    "Every line in the recording is real captured stdout. Each segment "
                    "is paced by the wall time its command took. Durations pace the "
                    "playback and are not published as claims."
                ),
                "segments": [
                    {
                        "label": segment["label"],
                        "command": segment["command"],
                        "exit_code": segment["exit_code"],
                        "seconds": segment["seconds"],
                        "lines_captured": len(segment["lines"]),
                    }
                    for segment in segments
                ],
                "total_seconds": round(total, 3),
                "gif_bytes": gif.stat().st_size,
                "mp4_bytes": mp4.stat().st_size,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    source.unlink(missing_ok=True)
    print(f"wrote {gif} ({gif.stat().st_size} bytes) and {mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
