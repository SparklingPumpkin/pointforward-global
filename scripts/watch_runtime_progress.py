#!/usr/bin/env python3
"""Show one continuously updated progress bar for global-runtime rebuild/render."""

import argparse
import json
import time
from pathlib import Path


def last_record(path: Path, required_key: str):
    if not path.exists():
        return None
    record = None
    for line in path.read_text(errors="replace").splitlines():
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and required_key in candidate:
            record = candidate
    return record


def progress_bar(done: int, total: int, width: int = 48) -> str:
    done = max(0, min(done, total))
    filled = round(width * done / total) if total else 0
    return f"[{'#' * filled}{'-' * (width - filled)}] {done:>3}/{total:<3} {100 * done / total:5.1f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--rebuild-log", type=Path, required=True)
    parser.add_argument("--rebuild-total", type=int, required=True)
    parser.add_argument("--render-log", type=Path, required=True)
    parser.add_argument("--render-total", type=int, default=156)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    while True:
        render = last_record(args.render_log, "index")
        rebuild = last_record(args.rebuild_log, "window_index")
        if args.manifest.exists():
            phase, done, total = "完成", args.render_total, args.render_total
        elif render is not None:
            phase, done, total = "最终全局 128D+UNet 渲染", int(render["index"]) + 1, args.render_total
        else:
            phase, done, total = "全局 128D Gaussian 重建", (int(rebuild["window_index"]) + 1 if rebuild else 0), args.rebuild_total

        print("\033[2J\033[H", end="")
        print(f"{args.label}\n")
        print(f"阶段: {phase}")
        print(progress_bar(done, total))
        if rebuild is not None:
            print(f"当前全局 Gaussian: {int(rebuild.get('gaussian_count', 0)):,}")
        print(f"更新时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nCtrl-b 然后 d：退出查看，任务继续运行", flush=True)
        if args.manifest.exists():
            time.sleep(10)
        else:
            time.sleep(2)


if __name__ == "__main__":
    main()
