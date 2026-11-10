"""Do the numbers the voice says match the numbers on screen? The owner, 2026-11-07: "it says two hundred and twenty but the screen shows 222".

    python -m logit.lecture02.runs.number_audit list            # every spoken number, per scene, with its time; and the near misses in the scene's code
    python -m logit.lecture02.runs.number_audit sheets [ch,ch]  # <build>/number_audit/<id>.jpg: the frame while each number is said, captioned with the phrase

No model runs here: words -> values, and frames from the rendered clips."""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from logit.lecture.audio_util import tool_env
from logit.lecture.paths import BUILD, CONTENT

UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: (i + 2) * 10 for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
SCALES = {"hundred": 100, "thousand": 1000, "million": 10 ** 6, "billion": 10 ** 9}


def tokens(word):
    return [x for x in re.sub(r"[^a-z\- ]", "", word.lower()).replace("-", " ").split()]


def is_num_token(t):
    return t in UNITS or t in TENS or t in SCALES


def value_of(ts):
    """A run of number words as a value. 'two hundred and twenty four' -> 224. 'nineteen fifty nine' -> 1959 (a year)."""
    ts = [t for t in ts if t != "and"]
    if not ts:
        return None
    if len(ts) >= 2 and ts[0] in UNITS and 10 <= UNITS[ts[0]] <= 19 and (ts[1] in TENS or ts[1] in UNITS or ts[1] == "hundred"):
        # years and hundreds: nineteen fifty nine, twelve hundred fifty (fixed 2026-11-10: the words after
        # "hundred" were dropped, so this read 1200). Twenty twelve is below.
        rest = value_of(ts[1:]) if ts[1] != "hundred" else (value_of(ts[2:]) or 0)
        if rest is not None and rest < 100:
            return UNITS[ts[0]] * 100 + rest
    if ts[0] == "twenty" and len(ts) >= 2 and ts[1] in UNITS and UNITS[ts[1]] >= 10:
        return 2000 + UNITS[ts[1]]
    total, cur = 0, 0
    for t in ts:
        if t in UNITS:
            cur += UNITS[t]
        elif t in TENS:
            cur += TENS[t]
        elif t == "hundred":
            cur = max(cur, 1) * 100
        elif t in SCALES:
            total += max(cur, 1) * SCALES[t]
            cur = 0
    return total + cur


def spoken_numbers(words):
    """[(value, phrase, start, end)] from a scene's timing words. 'a hundred' counts."""
    out, k = [], 0
    while k < len(words):
        ts = tokens(words[k]["w"])
        if ts and (is_num_token(ts[0]) or (ts[0] == "a" and k + 1 < len(words) and tokens(words[k + 1]["w"])[:1] in (["hundred"], ["thousand"], ["million"], ["billion"]))):
            j, run = k, []
            while j < len(words):
                tj = tokens(words[j]["w"])
                ok = tj and all(is_num_token(x) or x == "and" or (x == "a" and j == k) for x in tj)
                # "and" joins only when a number word follows and a scale word came before (three and a half is handled by the caller's eye)
                if tj == ["and"]:
                    nxt = tokens(words[j + 1]["w"]) if j + 1 < len(words) else []
                    ok = bool(nxt) and is_num_token(nxt[0]) and any(x in SCALES for x in run)
                if not ok:
                    break
                run += tj
                stop = words[j]["w"][-1] in ",.;:?!" and j + 1 < len(words)
                j += 1
                if stop:
                    break
            if run:
                clean = [x for x in run if x != "a"]
                out.append((value_of(clean), " ".join(w["w"] for w in words[k:j]), words[k]["start"], words[j - 1]["end"]))
                k = j
                continue
        k += 1
    return out


def scene_strings(ch, sid):
    """The numbers written in the string literals of a scene's class (any string, not only Txt/Num arguments).
    "1,2,3" is three numbers; "13,000" is one."""
    src = (CONTENT / "scenes" / f"ch{ch:02d}.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    cls = "S" + sid.replace(".", "_")
    found = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for n in ast.walk(node):
                if isinstance(n, ast.Constant) and isinstance(n.value, str):
                    for m in re.findall(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?", n.value):
                        try:
                            found.add(float(m.replace(",", "").rstrip(".")))
                        except ValueError:
                            pass
    return found


def load():
    scenes = json.loads((CONTENT / "data" / "scenes.json").read_text(encoding="utf-8"))
    res = []
    for s in scenes:
        t = json.loads((CONTENT / "voice" / f"{s['id']}.json").read_text(encoding="utf-8"))
        res.append((s, t))
    return res


def cmd_list():
    total = 0
    for s, t in load():
        nums = spoken_numbers(t["words"])
        total += len(nums)
        lit = scene_strings(s["chapter"], s["id"]) if (CONTENT / "scenes" / f"ch{s['chapter']:02d}.py").exists() else set()
        vals = {n[0] for n in nums}
        near = []
        for x in sorted(lit):
            if x in vals or x < 20:
                continue
            for v in vals:
                if v and v >= 20 and x != v and abs(x - v) / max(v, 1) < 0.12:
                    near.append((x, v))
        print(f"[{s['id']}] " + "; ".join(f"{n[1]}={n[0]}@{n[2]:.1f}" for n in nums))
        unspoken = sorted(x for x in lit if x not in vals and x >= 20)
        if near:
            print(f"     NEAR MISS (screen, spoken): {near}")
        if unspoken:
            print(f"     on screen, not spoken: {unspoken}")
    print("spoken numbers:", total)


def font(px):
    for p in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, px)
    return ImageFont.load_default()


def cmd_sheets(chapters):
    """One sheet a scene: for each spoken number, a frame just after it was said. Scene clips start with no lead-in: the voice timing is the clip's clock (checked against timeline.json by eye)."""
    out = BUILD / "number_audit"
    out.mkdir(exist_ok=True)
    f = font(20)
    for s, t in load():
        if chapters and s["chapter"] not in chapters:
            continue
        nums = spoken_numbers(t["words"])
        if not nums:
            continue
        clip = BUILD / "final" / f"{s['id']}.mp4"
        # phrases close together share a frame: one tile per 2.5 s of voice
        tiles, last = [], -9
        for v, ph, a, b in nums:
            if tiles and a - last < 2.5:
                tiles[-1][1].append(ph)
                tiles[-1][0] = b
            else:
                tiles.append([b, [ph]])
            last = b
        ims = []
        for at, phs in tiles:
            p = out / "_f.jpg"
            p.unlink(missing_ok=True)                    # never caption the previous tile's frame (fixed 2026-11-10)
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{at + 0.45:.2f}", "-i", str(clip), "-frames:v", "1", "-vf", "scale=800:450", "-q:v", "3", str(p)], env=tool_env(), check=False)
            if not p.exists():
                print(f"  {s['id']}: no frame at {at + 0.45:.2f}s (past the clip's end?), tile skipped: {' | '.join(phs)}", flush=True)
                continue
            im = Image.open(p).convert("RGB")
            im.load()
            ims.append((im, f"{at + 0.45:.1f}s: " + " | ".join(phs)))
        if not ims:
            continue
        cols = 2 if len(ims) <= 6 else 3
        rows = -(-len(ims) // cols)
        tw, th = 800, 450 + 52
        sh = Image.new("RGB", (cols * tw, rows * th), (28, 30, 36))
        d = ImageDraw.Draw(sh)
        for i, (im, cap) in enumerate(ims):
            x, y = (i % cols) * tw, (i // cols) * th
            sh.paste(im, (x, y + 52))
            lines = [cap[k:k + 78] for k in range(0, len(cap), 78)][:2]
            for li, ln in enumerate(lines):
                d.text((x + 8, y + 3 + li * 23), ln, font=f, fill=(255, 225, 120))
        scale = min(1.0, 1800 / sh.width)
        if scale < 1:
            sh = sh.resize((int(sh.width * scale), int(sh.height * scale)), Image.LANCZOS)
        sh.save(out / f"{s['id']}.jpg", quality=85)
        print(s["id"], len(ims), "tiles", flush=True)
    (out / "_f.jpg").unlink(missing_ok=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2 or sys.argv[1] not in ("list", "sheets"):
        sys.exit(__doc__)
    if sys.argv[1] == "list":
        cmd_list()
    else:
        cmd_sheets([int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else None)
