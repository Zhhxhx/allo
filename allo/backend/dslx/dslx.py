# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public interfaces for the XLS DSLX and direct-IR backends."""

import os
from pathlib import Path
import shutil
import subprocess

from allo._mlir import ir as mlir_ir
from allo._mlir.dialects import func as func_d
from allo.utils import register_dialect
from .lowering import MlirToXlsIRLowerer
from .lowering.mlir_to_dslx_fn import function_name


def find_xls_stdlib():
    for name in ("ir_converter_main", "interpreter_main"):
        path = shutil.which(name)
        if path:
            for relative in ("xls/dslx/stdlib", "dslx/stdlib"):
                library = Path(path).resolve().parent / relative
                if library.is_dir():
                    return str(library)
    return None


def parse_mlir_text(mlir_text):
    with mlir_ir.Context() as ctx:
        register_dialect(ctx)
        return mlir_ir.Module.parse(mlir_text, ctx)


def extract_func_op(module, name=None):
    for op in module.body.operations:
        if isinstance(op, func_d.FuncOp) and (name is None or op.name.value == name):
            return op
    raise ValueError(f"no MLIR function named {name!r}")


class _DslxModule:
    def __init__(self, mod, top_func_name):
        self.module = parse_mlir_text(mod) if isinstance(mod, str) else mod
        self.func = top_func_name
        self.func_op = extract_func_op(self.module, self.func)
        self._out_dir = "xls_output"
        self._stdlib_path = find_xls_stdlib()

    def _path(self, suffix):
        return Path(self._out_dir) / (self.func + suffix)

    @property
    def _dslx_path(self):
        return str(self._path(".x"))

    @property
    def _ir_path(self):
        return str(self._path(".ir"))

    @property
    def _opt_ir_path(self):
        return str(self._path(".opt.ir"))

    @property
    def _verilog_path(self):
        return str(self._path(".v"))

    def _write_dslx(self, source):
        Path(self._out_dir).mkdir(parents=True, exist_ok=True)
        self._path(".x").write_text(source)
        return self._dslx_path

    def _run(self, command, verbose, output=None):
        result = subprocess.run(command, capture_output=True, text=True)
        if verbose:
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
        result.check_returncode()
        if output is not None:
            Path(output).write_text(result.stdout)
        return result

    def _lib_flags(self):
        return [f"--dslx_stdlib_path={self._stdlib_path}"] if self._stdlib_path else []

    def interpret(self, verbose=True):
        path = self._write_dslx(self.codegen())
        return self._run(["interpreter_main", *self._lib_flags(), path], verbose)

    def to_ir(self, verbose=True):
        self._write_dslx(self.codegen())
        return self._convert(verbose)

    def _convert(self, verbose):
        return self._run(
            [
                "ir_converter_main",
                *self._lib_flags(),
                f"--top={function_name(self.func)}",
                self._dslx_path,
            ],
            verbose,
            self._ir_path,
        )

    def opt(self, verbose=True, *, opt_level=None):
        options = [f"--opt_level={opt_level}"] if opt_level is not None else []
        return self._run(
            ["opt_main", *options, self._ir_path], verbose, self._opt_ir_path
        )

    def _generate(
        self,
        verbose,
        pipeline_stages,
        clock_period_ps,
        delay_model,
        reset,
        extra=(),
        module_name=None,
    ):
        if pipeline_stages is not None and clock_period_ps is not None:
            raise ValueError("cannot specify both pipeline_stages and clock_period_ps")
        timing = (
            f"--clock_period_ps={clock_period_ps}"
            if clock_period_ps is not None
            else f"--pipeline_stages={1 if pipeline_stages is None else pipeline_stages}"
        )
        return self._run(
            [
                "codegen_main",
                "--generator=pipeline",
                timing,
                f"--delay_model={delay_model}",
                f"--reset={reset}",
                f"--module_name={module_name or function_name(self.func)}",
                "--use_system_verilog=false",
                *extra,
                self._opt_ir_path,
            ],
            verbose,
            self._verilog_path,
        )

    def flow(self):
        self.to_ir(verbose=False)
        self.opt(verbose=False)
        self.to_vlog(verbose=False)

    def __str__(self):
        return self.codegen()


class DslxFunctionModule(_DslxModule):
    def codegen(self):
        from .lowering.mlir_to_dslx_fn import lower_functions

        return lower_functions(self.module, self.func)

    def to_vlog(
        self,
        verbose=True,
        pipeline_stages=None,
        clock_period_ps=None,
        delay_model="sky130",
        reset="rst",
    ):
        return self._generate(
            verbose, pipeline_stages, clock_period_ps, delay_model, reset
        )


class DslxProcModule(_DslxModule):
    def __init__(self, mod, top_func_name):
        super().__init__(mod, top_func_name)
        self.network = None

    def codegen(self):
        from .lowering.proc_dispatch import emit_proc_module

        source, self.network = emit_proc_module(self.module, self.func_op)
        self._emitted = True
        return source

    def to_vlog(
        self,
        verbose=True,
        delay_model="sky130",
        reset="rst",
        *,
        pipeline_stages=None,
        clock_period_ps=None,
        module_name=None,
        flop_inputs=True,
        flop_outputs=True,
    ):
        from .lowering.core_ram_rtl import wrap_ram_rtl

        if not hasattr(self, "_emitted"):
            self.codegen()
        name = module_name or function_name(self.func)
        bind_rams = self.network is not None and bool(self.network.rams)
        flags = ["--fifo_module=", "--materialize_internal_fifos"]
        if not flop_inputs:
            flags.append("--noflop_inputs")
        if not flop_outputs:
            flags.append("--noflop_outputs")
        result = self._generate(
            verbose,
            pipeline_stages,
            clock_period_ps,
            delay_model,
            reset,
            flags,
            name + "_datapath" if bind_rams else name,
        )
        if bind_rams:
            result.stdout = wrap_ram_rtl(
                result.stdout, name, self.network.ports, self.network.rams, reset
            )
            self._path(".v").write_text(result.stdout)
        return result


# interface for direct ir lowering (mlir -> ir -> verilog, bypassing dslx)
class XlsIRModule:
  def __init__(self, mod, top_func_name):
    # accept either mlir string or module object
    if isinstance(mod, str):
      self.module = parse_mlir_text(mod)
    else:
      self.module = mod

    self.func = top_func_name
    self._out_dir = "xls_output"

    # extract function op for lowering
    self.func_op = extract_func_op(self.module)

  @property
  def _ir_path(self):
    return os.path.join(self._out_dir, f"{self.func}.ir")

  @property
  def _opt_ir_path(self):
    return os.path.join(self._out_dir, f"{self.func}.opt.ir")

  @property
  def _verilog_path(self):
    return os.path.join(self._out_dir, f"{self.func}.v")

  # generate xls ir directly from mlir (bypassing dslx)
  def to_ir(self, verbose=True):
    lowerer = MlirToXlsIRLowerer(self.func_op)
    ir_code = lowerer.lower()

    os.makedirs(self._out_dir, exist_ok=True)
    with open(self._ir_path, "w") as f:
      f.write(ir_code)

    if verbose:
      print(ir_code)

    return ir_code

  # optimize xls ir
  def opt(self, verbose=True):
    if not os.path.isfile(self._ir_path):
      raise FileNotFoundError(f"please run to_ir() first")
    result = subprocess.run(["opt_main", self._ir_path],
                             capture_output=True, text=True)
    with open(self._opt_ir_path, "w") as f:
      f.write(result.stdout)
    if verbose:
      print(result.stdout)
      if result.stderr:
        print(result.stderr)
    return result

  # generate verilog from optimized ir
  def to_vlog(self, verbose=True, pipeline_stages=None, clock_period_ps=None,
              delay_model="sky130", reset="rst"):
    if not os.path.isfile(self._opt_ir_path):
      raise FileNotFoundError(f"please run opt() first")

    if pipeline_stages is not None and clock_period_ps is not None:
      raise ValueError("cannot specify both pipeline_stages and clock_period_ps")

    # default to 1 stage for function-based lowering
    if pipeline_stages is None and clock_period_ps is None:
      pipeline_stages = 1

    cmd = ["codegen_main", f"--delay_model={delay_model}", f"--reset={reset}"]
    if pipeline_stages is not None:
      cmd.append(f"--pipeline_stages={pipeline_stages}")
    elif clock_period_ps is not None:
      cmd.append(f"--clock_period_ps={clock_period_ps}")
    cmd.append(self._opt_ir_path)

    if verbose:
      print(" ".join(cmd))
      print()

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode == 0:
      with open(self._verilog_path, "w") as f:
        f.write(result.stdout)
      if verbose:
        print(result.stdout)
        if result.stderr:
          print(result.stderr)
    else:
      print(f"codegen_main failed with exit code {result.returncode}")
      if result.stderr:
        print("stderr:")
        print(result.stderr)
      if result.stdout:
        print("stdout:")
        print(result.stdout)
      raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    return result

  # run full pipeline: mlir -> ir -> optimize -> verilog
  def flow(self):
    self.to_ir(verbose=False)
    self.opt(verbose=False)
    self.to_vlog(verbose=False)

  def __str__(self):
    return self.to_ir(verbose=False)
