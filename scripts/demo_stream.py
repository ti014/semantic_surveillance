"""
Simple interactive streaming demo.
- Opens camera or video file.
- Uses PPEMonitorApp for real-time visualization and key controls:
  q/ESC: quit, p: pause/resume, s: save frame.
"""

import os
import sys
import argparse

# Ensure project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import PPEMonitorApp


def parse_args():
    parser = argparse.ArgumentParser(description="PPE streaming demo")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_yolov8_production.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera index (e.g., 0) or video path",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Optional: path to save annotated video (mp4)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional: stop after N frames",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Optional: override confidence threshold",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    app = PPEMonitorApp(args.config)

    # Override config values if provided
    if args.conf is not None:
        app.config["detection"]["confidence_threshold"] = args.conf

    # Prefer CLI source; fallback to config if blank
    source = args.source if args.source is not None else app.config["video"].get("source", "0")

    app.run(source=source, max_frames=args.max_frames, save_path=args.save_path)


if __name__ == "__main__":
    main()
