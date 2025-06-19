#!/usr/bin/env python3
"""
trace_runner.py – run another Python file under a custom tracer.

Usage:
    python trace_runner.py target.py -- <target args>
"""
import sys
import os
import runpy
import argparse
import traceback

# ---------- the tracer you supplied, with a few extras ----------
def my_tracer(frame, event, arg=None):
    """Print one‐line trace records for *every* event we see."""
    # Don’t trace inside this driver itself
    if frame.f_code.co_filename == __file__:
        return my_tracer

    code      = frame.f_code
    func_name = code.co_name
    line_no   = frame.f_lineno
    file_name = os.path.relpath(code.co_filename)

    print(f"A {event} encountered in {func_name}() at {file_name}:{line_no}")
    return my_tracer
# ----------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Python script with a live execution trace."
    )
    parser.add_argument(
        "script",
        help="Path to the Python file you want to trace.",
    )
    # Everything after '--' goes straight to the target script
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments for the target script (use -- before them).",
    )
    args = parser.parse_args()

    # Rebuild sys.argv so the target sees exactly what it would
    # if you ran `python target.py ...` directly.
    sys.argv = [args.script] + args.script_args

    # Activate tracing
    sys.settrace(my_tracer)
    try:
        # Run the file as if it were the __main__ module
        runpy.run_path(args.script, run_name="__main__")
    except Exception:
        # Show the normal traceback *after* our per‑line trace
        traceback.print_exc()
    finally:
        sys.settrace(None)


if __name__ == "__main__":
    main()
