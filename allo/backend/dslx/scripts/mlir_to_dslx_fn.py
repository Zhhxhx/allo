#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert an Allo MLIR function and its call closure to DSLX."""

import argparse
from pathlib import Path
import sys


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

try:
    from allo._mlir import ir
    from allo._mlir.dialects import func as func_d
    from allo.backend.dslx.dslx import parse_mlir_text as parse_mlir
    from allo.backend.dslx.lowering.mlir_to_dslx_fn import lower_functions
except ImportError as exc:
    sys.exit(f"Allo must be installed to convert MLIR to DSLX: {exc}")


def mlir_to_dslx(module, top=None):
    """Select one MLIR entry and emit its reachable DSLX functions."""
    if top is None:
        names = [
            op.name.value
            for op in module.body.operations
            if isinstance(op, func_d.FuncOp)
        ]
        if not names:
            raise ValueError("No function found in MLIR module")
        if len(names) > 1:
            raise ValueError(
                "MLIR module contains multiple functions. "
                "Select an entry with --top NAME. Available entries: "
                + ", ".join(names)
            )
        top = names[0]
    return lower_functions(module, top)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert Allo MLIR to DSLX")
    parser.add_argument("input", nargs="?", type=Path, help="Input MLIR file")
    parser.add_argument("--stdin", action="store_true", help="Read MLIR from stdin")
    parser.add_argument("--top", metavar="NAME", help="MLIR function symbol to compile")
    parser.add_argument(
        "-o", "--output", type=Path, help="Output file or directory (default: stdout)"
    )
    parser.add_argument(
        "--name",
        default="output",
        help="Base name for stdin output when writing to a directory (default: output)",
    )
    args = parser.parse_args(argv)
    if args.stdin == (args.input is not None):
        parser.error("Specify either an input file or --stdin")

    try:
        mlir_text = (
            sys.stdin.read() if args.stdin else args.input.read_text(encoding="utf-8")
        )
        module = parse_mlir(mlir_text)
        module.operation.verify()
        source = mlir_to_dslx(module, args.top)
        if args.output is None:
            sys.stdout.write(source)
            return 0

        output = args.output
        if output.is_dir() or (not output.exists() and not output.suffix):
            name = args.name if args.stdin else args.input.stem
            output = output / f"{name}.x"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(source, encoding="utf-8")
        print(f"Generated DSLX at {output}")
        return 0
    except (OSError, ValueError, ir.MLIRError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
