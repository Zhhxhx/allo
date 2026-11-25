# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile DSLX through XLS and compare simulated RTL against independent oracles."""

import csv
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import re
import runpy
import shlex
import shutil
import subprocess
import sys

import numpy as np
import pytest

import allo
from allo._mlir import ir
from allo.backend.dslx import DslxFunctionModule, DslxProcModule
from allo.ir.types import float32, int32
from allo.utils import register_dialect


FIXTURES = Path(__file__).with_name("dslx_rtl")
ENTRIES = {"fn": DslxFunctionModule, "proc": DslxProcModule}
MASK32 = (1 << 32) - 1
FROZEN_COUNTS = {"fadd": 4135, "fmul": 4132}
FROZEN_SEED = 0x43EF46FE
RANDOM_COUNT = 4096


def subtract_i32(left: int32, right: int32) -> int32:
    return left - right


def dot_i32(left: int32[4], right: int32[4]) -> int32:
    total: int32 = 17
    for i in range(4):
        total += left[i] * right[i]
    return total


def add_f32(left: float32, right: float32) -> float32:
    return left + right


def multiply_f32(left: float32, right: float32) -> float32:
    return left * right


def _tool(name, required=False, directory=None):
    executable = shutil.which(name, path=directory)
    if executable is None:
        message = f"{name} is required for DSLX RTL simulation"
        if required:
            pytest.fail(message)
        pytest.skip(message)
    return Path(executable).resolve()


@pytest.fixture(scope="session")
def rtl_tools():
    xls_directory = os.environ.get("XLS_BIN")
    required = (
        xls_directory is not None or os.environ.get("ALLO_DSLX_RTL_REQUIRED") == "1"
    )
    tools = {
        name: _tool(name, required, xls_directory)
        for name in ("ir_converter_main", "opt_main", "codegen_main")
    }
    for name in ("iverilog", "vvp"):
        tools[name] = _tool(name, required)
    stdlib = Path(
        os.environ.get(
            "XLS_DSLX_STDLIB",
            tools["ir_converter_main"].parent / "xls" / "dslx" / "stdlib",
        )
    ).resolve()
    if not stdlib.is_dir():
        message = f"Cannot find XLS DSLX stdlib at {stdlib}; set XLS_DSLX_STDLIB"
        if required:
            pytest.fail(message)
        pytest.skip(message)
    tools["stdlib"] = stdlib
    return tools


def _mark(directory, phase):
    (directory / "phase.txt").write_text(f"{phase}\n", encoding="utf-8")


def _run(directory, phase, command, output=None):
    _mark(directory, phase)
    command = [str(arg) for arg in command]
    with (directory / "commands.log").open("a", encoding="utf-8") as log:
        log.write(f"{phase}: {shlex.join(command)}\n")
    result = subprocess.run(
        command, cwd=directory, capture_output=True, text=True, timeout=120, check=False
    )
    (directory / (output or f"{phase}.log")).write_text(result.stdout, encoding="utf-8")
    (directory / f"{phase}.stderr").write_text(result.stderr, encoding="utf-8")
    diagnostics = result.stderr
    if output is None:
        diagnostics = result.stdout + diagnostics
    assert (
        result.returncode == 0
    ), f"{phase} failed ({result.returncode}): {shlex.join(command)}\n{diagnostics}"
    return result.stdout


def _compile(directory, tools, entry, kernel):
    top = kernel.__name__
    (directory / "source.txt").write_text(
        f"{inspect.getfile(kernel)}\n{inspect.getsource(kernel)}",
        encoding="utf-8",
    )
    case = {"entry": entry, "top": top, "backend": inspect.getfile(ENTRIES[entry])}
    (directory / "case.json").write_text(json.dumps(case) + "\n", encoding="utf-8")
    with ir.Context() as context, ir.Location.unknown():
        register_dialect(context)
        _mark(directory, "allo-customize")
        module = allo.customize(kernel).module
        assert module.operation.verify()
        (directory / f"{top}.mlir").write_text(str(module), encoding="utf-8")
        _mark(directory, "dslx-emitter")
        source = ENTRIES[entry](module, top).codegen()
    (directory / f"{top}.x").write_text(source, encoding="utf-8")
    _run(
        directory,
        "dslx-to-ir",
        [
            tools["ir_converter_main"],
            f"--dslx_stdlib_path={tools['stdlib']}",
            f"--top={top}",
            f"{top}.x",
        ],
        output=f"{top}.ir",
    )
    _run(directory, "xls-opt", [tools["opt_main"], f"{top}.ir"], output=f"{top}.opt.ir")
    _run(
        directory,
        "xls-codegen",
        [
            tools["codegen_main"],
            "--generator=pipeline",
            "--pipeline_stages=3",
            "--delay_model=unit",
            "--reset=rst",
            "--use_system_verilog=false",
            f"--module_name={top}",
            f"{top}.opt.ir",
        ],
        output=f"{top}.v",
    )
    return top


def _iverilog(directory, tools, phase, *arguments):
    ivl_path = os.environ.get("IVERILOG_IVL_PATH")
    ivl = ["-B", ivl_path] if ivl_path else []
    _run(directory, phase, [tools["iverilog"], *ivl, "-g2012", *arguments])


def _check_rtl(directory, tools, entry, kernel, vectors, width=32, ieee_nan=False):
    top = _compile(directory, tools, entry, kernel)
    digits = width // 4
    (directory / "vectors.mem").write_text(
        "".join(
            f"{a:0{digits}x}{b:0{digits}x}{value:08x}\n" for a, b, value in vectors
        ),
        encoding="utf-8",
    )
    defines = [f"-DDUT={top}", f"-DCOUNT={len(vectors)}", f"-DINPUT_BITS={width}"]
    if entry == "proc":
        defines.append("-DPROC")
    if ieee_nan:
        defines.append("-DIEEE_NAN")
    _iverilog(
        directory,
        tools,
        "iverilog",
        "-s",
        "compare_tb",
        *defines,
        "-o",
        "compare.vvp",
        FIXTURES / "compare_tb.sv",
        f"{top}.v",
    )
    output = _run(directory, "rtl-sim", [tools["vvp"], "compare.vvp"])
    counters = re.search(r"^COMPARE compared=(\d+) mismatches=(\d+)$", output, re.M)
    assert counters is not None, output
    compared, mismatches = map(int, counters.groups())
    first = next(
        (line for line in output.splitlines() if line.startswith("MISMATCH ")), None
    )
    result = {
        "entry": entry,
        "top": top,
        "compared": compared,
        "mismatches": mismatches,
        "first_mismatch": first,
    }
    (directory / "comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    _mark(directory, "rtl-compare")
    assert compared == len(vectors), result
    assert mismatches == 0, result
    _mark(directory, "passed")


@pytest.mark.parametrize("entry", ENTRIES)
def test_dslx_rtl_integer_subtract(entry, rtl_tools, tmp_path):
    pairs = [(7, 24), (24, 7), (-9, 6), (6, -9), (-21, -3), (0, 0), (2**31 - 1, -1)]
    vectors = [(a & MASK32, b & MASK32, (a - b) & MASK32) for a, b in pairs]
    _check_rtl(tmp_path, rtl_tools, entry, subtract_i32, vectors)


def test_dslx_rtl_integer_dot(rtl_tools, tmp_path):
    def pack(vector):
        return sum((value & MASK32) << (32 * i) for i, value in enumerate(vector))

    pairs = [
        ([1, -2, 3, -4], [5, 6, -7, 8]),
        ([6, 8, 10, 12], [3, 0, -1, 2]),
        ([0, 0, 0, 0], [2, 3, 4, 5]),
        ([0, 0, 0, 3], [1, 2, 3, 4]),
        ([0, 0, 3, 0], [1, 2, 3, 4]),
        ([2**30, -(2**31), 1, -1], [7, 3, 4, 5]),
    ]
    vectors = [
        (
            pack(a),
            pack(b),
            (17 + sum(x * y for x, y in zip(a, b, strict=True))) & MASK32,
        )
        for a, b in pairs
    ]
    _check_rtl(tmp_path, rtl_tools, "fn", dot_i32, vectors, width=128)


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize(
    "kernel, operation", [(add_f32, np.add), (multiply_f32, np.multiply)]
)
def test_dslx_rtl_binary32(entry, kernel, operation, rtl_tools, tmp_path):
    rng = random.Random(525)
    pairs = [
        (0x3FC00000, 0x40100000),
        (0xC0600000, 0x3E800000),
        (0x3F800000, 0x33800000),
        (0x3F800001, 0x33800000),
        (0x3FC00000, 0x3F800003),
        (0x3FC00000, 0x3F800001),
        (0x3F800000, 0xBF800000),
        (0x80000000, 0x00000000),
        (0x80000000, 0x80000000),
        (0x80000000, 0x3F800000),
        (0x00800001, 0x80800000),
        (0x00800000, 0x3F000000),
        (0x00800000, 0x3F7FFFFF),
        (0x00000001, 0x3F000000),
        (0x00000003, 0x3F000000),
        (0x80000003, 0x3F000000),
        (0x007FFFFF, 0x00000001),
        (0x007FFFFF, 0x40000000),
        (0x7F7FFFFF, 0x40000000),
        (0x7F7FFFFF, 0x7F7FFFFF),
        (0x7F800000, 0xFF800000),
        (0x7F800000, 0x00000000),
        (0xFF800000, 0x3F800000),
        (0x7F812345, 0x3F800000),
        (0x3F800000, 0xFFC56789),
    ]
    pairs.extend((rng.getrandbits(32), rng.getrandbits(32)) for _ in range(64))
    for _ in range(32):
        subnormal = rng.randrange(1, 0x00800000)
        sign = rng.randrange(2) << 31
        near_one = rng.randrange(0x3F7FFFF0, 0x3F800011)
        pairs.append((subnormal | sign, near_one))
        pairs.append((0x00800000 + subnormal, 0x80800000))
    inputs = np.array(pairs, dtype=np.uint32).view(np.float32)
    with np.errstate(all="ignore"):
        expected = operation(inputs[:, 0], inputs[:, 1]).view(np.uint32)
    vectors = [
        (a, b, int(value)) for (a, b), value in zip(pairs, expected, strict=True)
    ]
    _check_rtl(tmp_path, rtl_tools, entry, kernel, vectors, ieee_nan=True)


@pytest.fixture(scope="session")
def opentpu_paths():
    keys = (
        "ALLO_OPENTPU_SNAPSHOT",
        "ALLO_OPENTPU_FP_RTL",
        "ALLO_OPENTPU_PROC_EVIDENCE",
    )
    values = [os.environ.get(key) for key in keys]
    if not any(values):
        pytest.skip("Set " + ", ".join(keys) + " to run the frozen OpenTPU RTL tests")
    assert all(values), "Set all frozen OpenTPU paths: " + ", ".join(keys)
    snapshot, rtl, evidence = (Path(value).resolve(strict=True) for value in values)
    assert (snapshot / "native_arithmetic.py").is_file()
    assert (snapshot / "arithmetic.py").is_file()
    assert rtl.is_file()
    assert evidence.is_dir()
    return snapshot, rtl, evidence


@pytest.fixture(scope="session")
def opentpu_oracle(opentpu_paths, tmp_path_factory):
    _, rtl, _ = opentpu_paths
    work = tmp_path_factory.mktemp("opentpu-oracle")
    executable = work / "obj" / "oracle"
    _run(
        work,
        "verilator",
        [
            _tool("verilator", required=True),
            "--binary",
            "-j",
            "2",
            "-Wno-fatal",
            "--top-module",
            "oracle_tb",
            "--Mdir",
            executable.parent,
            "-o",
            executable.name,
            rtl,
            FIXTURES / "oracle_tb.sv",
        ],
    )
    return executable


@pytest.fixture(scope="session", params=FROZEN_COUNTS)
def frozen_vectors(request, opentpu_paths, opentpu_oracle, tmp_path_factory):
    _, _, evidence = opentpu_paths
    operator = request.param
    count = FROZEN_COUNTS[operator]
    directory = evidence / operator / "vectors"
    raw = directory / "input.hex"
    pairs = [
        tuple(int(word, 16) for word in row.split())
        for row in raw.read_text().splitlines()
    ]
    seed = json.loads((directory / "seed.json").read_text())
    assert seed["seed"] == FROZEN_SEED
    assert len(pairs) == count
    rng = random.Random(FROZEN_SEED)
    assert pairs[-RANDOM_COUNT:] == [
        (rng.getrandbits(32), rng.getrandbits(32)) for _ in range(RANDOM_COUNT)
    ]
    work = tmp_path_factory.mktemp(f"{operator}-oracle")
    log = _run(
        work,
        "oracle",
        [opentpu_oracle, f"+op={operator[1:]}", f"+input={raw}", f"+count={count}"],
    )
    rows = [line.split()[1:] for line in log.splitlines() if line.startswith("ORACLE ")]
    assert len(rows) == count
    vectors = []
    for index, row in enumerate(rows):
        number, left, right, expected = row
        left, right, expected = (int(word, 16) for word in (left, right, expected))
        assert int(number) == index and (left, right) == pairs[index], row
        vectors.append((left, right, expected))
    with (directory / "oracle.tsv").open(encoding="utf-8") as archive:
        archived_vectors = [
            tuple(int(row[word], 16) for word in ("left", "right", "expected"))
            for row in csv.DictReader(archive, delimiter="\t")
        ]
    assert (
        vectors == archived_vectors
    ), f"{operator}: rebuilt SV oracle differs from the archived oracle"
    return operator, vectors


@pytest.mark.parametrize("entry", ENTRIES)
def test_dslx_rtl_opentpu(
    frozen_vectors, opentpu_paths, entry, rtl_tools, tmp_path, monkeypatch
):
    snapshot, _, _ = opentpu_paths
    operator, vectors = frozen_vectors
    spec = importlib.util.spec_from_file_location(
        "arithmetic", snapshot / "arithmetic.py"
    )
    helper = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "arithmetic", helper)
    spec.loader.exec_module(helper)
    native = runpy.run_path(str(snapshot / "native_arithmetic.py"))
    assert native["ftz"] is helper.ftz
    _check_rtl(tmp_path, rtl_tools, entry, native[operator], vectors)


def test_dslx_rtl_ram_binding(rtl_tools, tmp_path):
    from types import SimpleNamespace

    from allo.backend.dslx.lowering.core_ram_rtl import wrap_ram_rtl

    case = {
        "entry": "ram",
        "top": "ram_check",
        "backend": inspect.getfile(wrap_ram_rtl),
    }
    (tmp_path / "case.json").write_text(json.dumps(case) + "\n", encoding="utf-8")
    _run(
        tmp_path,
        "ram-dslx-to-ir",
        [
            rtl_tools["ir_converter_main"],
            f"--dslx_stdlib_path={rtl_tools['stdlib']}",
            "--top=ram_check",
            FIXTURES / "ram_datapath.x",
        ],
        output="ram_check.ir",
    )
    _run(
        tmp_path,
        "ram-xls-opt",
        [rtl_tools["opt_main"], "ram_check.ir"],
        output="ram_check.opt.ir",
    )
    datapath = _run(
        tmp_path,
        "ram-xls-codegen",
        [
            rtl_tools["codegen_main"],
            "--generator=pipeline",
            "--pipeline_stages=4",
            "--delay_model=unit",
            "--reset=clear",
            "--use_system_verilog=false",
            "--module_name=ram_check_datapath",
            "ram_check.opt.ir",
        ],
        output="ram_check_datapath.v",
    )
    ports = [
        SimpleNamespace(name="command", width=20, direction="in"),
        SimpleNamespace(name="answer", width=16, direction="out"),
    ]
    rams = [
        SimpleNamespace(
            name=name,
            request=f"{name}_req",
            response=f"{name}_resp",
            width=16,
            address_width=2,
            size=size,
            initial=initial,
        )
        for name, size, initial in (
            ("first", 3, (0x0123, -2, 0xBEEF)),
            ("second", 4, -1),
        )
    ]
    rtl = wrap_ram_rtl(datapath, "ram_check", ports, rams, reset="clear")
    assert rtl.startswith(datapath)
    (tmp_path / "ram_check.v").write_text(rtl, encoding="utf-8")
    _iverilog(
        tmp_path,
        rtl_tools,
        "ram-iverilog",
        "-s",
        "ram_tb",
        "-o",
        "ram.vvp",
        FIXTURES / "ram_tb.sv",
        "ram_check.v",
    )
    output = _run(tmp_path, "ram-sim", [rtl_tools["vvp"], "ram.vvp"])
    assert "RAM handshake and reset passed" in output
    _mark(tmp_path, "passed")
