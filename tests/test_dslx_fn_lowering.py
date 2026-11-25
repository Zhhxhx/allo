# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from allo._mlir import ir
from allo.backend.dslx.lowering.mlir_to_dslx_fn import (
    DslxFunctionError,
    MlirToDslxLowerer,
    function_name,
    lower_functions,
)
from allo.utils import register_dialect


@pytest.fixture
def parse_mlir():
    with ir.Context() as context:
        register_dialect(context)

        def parse(source):
            module = ir.Module.parse(source)
            module.operation.verify()
            return module

        yield parse


def test_named_top_emits_only_its_call_closure(parse_mlir):
    module = parse_mlir(
        """
        module {
          func.func @unreachable(%a: f64) -> f64 { return %a : f64 }
          func.func @helper.ssa(%a: i8, %b: i8) -> i8 {
            %diff = arith.subi %a, %b : i8
            return %diff : i8
          }
          func.func @update(%a: i8, %b: i8) -> i8 {
            %r = func.call @helper.ssa(%a, %b) : (i8, i8) -> i8
            return %r : i8
          }
        }
        """
    )
    helper = function_name("helper.ssa")
    top = function_name("update")
    result = lower_functions(module, "update")
    assert "unreachable" not in result
    assert f"fn {helper}(arg0: u8, arg1: u8) -> u8" in result
    assert f"{helper}(arg0, arg1)" in result
    assert f"fn {top}(arg0: u8, arg1: u8) -> u8" in result
    assert MlirToDslxLowerer(module.body.operations[2]).lower() == result

    renamed = lower_functions(module, "update", emitted_top_name="wrapped")
    assert "fn wrapped(arg0: u8, arg1: u8) -> u8" in renamed
    assert f"fn {top}(" not in renamed


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            """
            func.func @kernel(%a: f32, %b: f32) -> f32 {
              %r = arith.divf %a, %b : f32 loc("kernel.py":3:2)
              return %r : f32
            }
            """,
            r'@kernel: arith.divf at loc\("kernel.py":3:2\): unsupported operation',
        ),
        (
            """
            func.func @kernel(%x: memref<1xi16, "stream:4">) {
              return
            }
            """,
            "@kernel: func.func.*unsupported memory space",
        ),
        (
            """
            func.func @kernel(%x: memref<?xi16>) {
              return
            }
            """,
            "@kernel: func.func.*expected a nonempty, static memref shape",
        ),
        (
            """
            func.func @kernel(%x: memref<i16>, %value: i16) {
              affine.store %value, %x[] : memref<i16>
              return
            }
            """,
            "@kernel: affine.store.*writes to memref arguments",
        ),
        (
            """
            func.func @kernel() -> i16 {
              %x = memref.alloc() : memref<i16>
              %r = affine.load %x[] : memref<i16>
              return %r : i16
            }
            """,
            "@kernel: affine.load.*unbound or uninitialized value",
        ),
        (
            """
            func.func @kernel(%x: i8) -> i8 {
              %r = func.call @kernel(%x) : (i8) -> i8
              return %r : i8
            }
            """,
            "@kernel: func.call.*recursive call to @kernel",
        ),
    ],
)
def test_rejects_operations_without_pure_semantics(parse_mlir, source, message):
    module = parse_mlir("module {" + source + "}")
    with pytest.raises(DslxFunctionError, match=message):
        lower_functions(module, "kernel")


def test_explicit_loop_and_if_values(parse_mlir):
    module = parse_mlir(
        """
        module {
          func.func @kernel(%seed: i16, %cond: i1) -> (i16, i16) {
            %lo = arith.constant -2 : index
            %hi = arith.constant 5 : index
            %step = arith.constant 2 : index
            %r = scf.for %i = %lo to %hi step %step iter_args(%old = %seed) -> (i16) {
              %inc = arith.index_cast %i : index to i16
              %v = arith.addi %old, %inc : i16
              scf.yield %v : i16
            }
            %out = scf.if %cond -> (i16) {
              scf.yield %r : i16
            } else {
              scf.yield %seed : i16
            }
            return %r, %out : i16, i16
          }
        }
        """
    )

    result = lower_functions(module, "kernel")
    assert "fn kernel(arg0: u16, arg1: u1) -> (u16, u16)" in result
    assert "in u32:0..u32:4" in result
    assert "s64:-2 +" in result
    assert "* s64:2" in result
    assert "}(arg0);" in result
    assert "if arg1" in result
