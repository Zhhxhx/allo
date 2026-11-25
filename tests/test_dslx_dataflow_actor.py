# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess

import pytest

from allo._mlir import ir
from allo.utils import register_dialect
from allo.backend.dslx.lowering.core_actor import (
    ActorLowerer,
    ActorPort,
    prepare_actor_module,
)


_ACTOR = """
module {
  memref.global "private" @counter : memref<1xi32> = dense<7>
  func.func @transform(%scratch: memref<4xi32>, %index: index, %x: i32) -> i32 {
    %one = arith.constant 1 : i32
    %old = memref.load %scratch[%index] : memref<4xi32>
    %odd = arith.andi %x, %one : i32
    %do_subtract = arith.cmpi eq, %odd, %one : i32
    %updated = scf.if %do_subtract -> (i32) {
      %result = arith.subi %old, %x : i32
      scf.yield %result : i32
    } else {
      %result = arith.addi %old, %x : i32
      scf.yield %result : i32
    }
    memref.store %updated, %scratch[%index] : memref<4xi32>
    return %updated : i32
  }
  func.func @actor(%input: memref<2xi32, "stream:2;S">,
                   %output: memref<1xi32, "stream:2">) {
    %zero_i = arith.constant 0 : index
    %one_i = arith.constant 1 : index
    %four_i = arith.constant 4 : index
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %true = arith.constant true
    %counter = memref.get_global @counter : memref<1xi32>
    %local = memref.alloc() : memref<4xi32>
    scf.for %i = %zero_i to %four_i step %one_i {
      %i32 = arith.index_cast %i : index to i32
      memref.store %i32, %local[%i] : memref<4xi32>
    }
    scf.while : () -> () {
      scf.condition(%true)
    } do {
      %previous = memref.load %counter[%zero_i] : memref<1xi32>
      %low = arith.andi %previous, %one : i32
      %lane = arith.index_cast %low : i32 to index
      %x = memref.load %input[%lane] : memref<2xi32, "stream:2;S">
      %sum = scf.for %i = %zero_i to %four_i step %one_i
          iter_args(%acc = %zero) -> (i32) {
        %value = func.call @transform(%local, %i, %x) : (memref<4xi32>, index, i32) -> i32
        %next_sum = arith.addi %acc, %value : i32
        scf.yield %next_sum : i32
      }
      %result = arith.addi %sum, %previous : i32
      memref.store %result, %output[%zero_i] : memref<1xi32, "stream:2">
      %next = arith.addi %previous, %one : i32
      memref.store %next, %counter[%zero_i] : memref<1xi32>
      scf.yield
    }
    return
  }
}
"""

_TESTBENCH = """
module testbench;
  reg clk = 0;
  always #5 clk = !clk;
  reg rst = 1;
  reg [31:0] source_0 = 0, source_1 = 0;
  reg valid_0 = 0, valid_1 = 0, result_ready = 0;
  wire ready_0, ready_1, result_valid;
  wire [31:0] result;
  actor dut(.clk(clk), .rst(rst),
    ._source_0(source_0), ._source_0_vld(valid_0), ._source_0_rdy(ready_0),
    ._source_1(source_1), ._source_1_vld(valid_1), ._source_1_rdy(ready_1),
    ._result(result), ._result_vld(result_valid), ._result_rdy(result_ready));

  task push(input integer lane, input [31:0] value);
    begin
      @(negedge clk);
      if (lane == 0) begin
        source_0 = value;
        valid_0 = 1;
        @(posedge clk);
        while (!ready_0) @(posedge clk);
      end else begin
        source_1 = value;
        valid_1 = 1;
        @(posedge clk);
        while (!ready_1) @(posedge clk);
      end
      @(negedge clk);
      valid_0 = 0;
      valid_1 = 0;
    end
  endtask

  task check_result(input [31:0] expected);
    begin
      wait(result_valid);
      repeat (5) begin
        @(negedge clk);
        if (!result_valid || result !== expected)
          $fatal(1, "Stalled output %h, expected %h", result, expected);
      end
      result_ready = 1;
      @(posedge clk);
      @(negedge clk);
      result_ready = 0;
    end
  endtask

  initial begin
    repeat (4) @(negedge clk);
    rst = 0;
    push(1, 3);
    check_result(1);   // Counter 7 and scratch sum 0+1+2+3 - 4*3 = -6.
    push(0, 10);
    check_result(42);  // Counter 8 and scratch sum -6 + 4*10 = 34.
    push(1, 5);
    check_result(23);  // Counter 9 and scratch sum 34 - 4*5 = 14.
    @(negedge clk);
    rst = 1;
    repeat (4) @(negedge clk);
    rst = 0;
    push(1, 2);
    check_result(21);  // Counter 7 and scratch sum 0+1+2+3 + 4*2 = 14.
    $finish;
  end
  initial begin
    #12000;
    $fatal(1, "Timed out waiting for an actor transfer");
  end
endmodule
"""


def test_dataflow_actor_state_and_backpressure(tmp_path):
    tools = ("ir_converter_main", "opt_main", "codegen_main", "iverilog", "vvp")
    for tool in tools:
        if shutil.which(tool) is None:
            pytest.skip(f"{tool} is required")
    with ir.Context() as context:
        register_dialect(context)
        module = prepare_actor_module(ir.Module.parse(_ACTOR))
        actor = next(
            op
            for op in module.body.operations
            if op.operation.name == "func.func"
            and op.attributes["sym_name"].value == "actor"
        )
        lowerer = ActorLowerer(
            module,
            actor,
            "actor",
            {0: ActorPort(("source_0", "source_1")), 1: ActorPort(("result",))},
        )
        source = lowerer.lower()
        assert lowerer.directions == {0: "in", 1: "out"}

    def run(args, output=None):
        result = subprocess.run(
            args,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        if output is not None:
            (tmp_path / output).write_text(result.stdout, encoding="utf-8")

    (tmp_path / "actor.x").write_text(source, encoding="utf-8")
    (tmp_path / "testbench.v").write_text(_TESTBENCH, encoding="utf-8")
    run(["ir_converter_main", "--top=actor", "actor.x"], "actor.ir")
    run(["opt_main", "actor.ir"], "actor.opt.ir")
    run(
        [
            "codegen_main",
            "--module_name=actor",
            "--generator=pipeline",
            "--pipeline_stages=1",
            "--use_system_verilog=false",
            "--delay_model=unit",
            "--reset=rst",
            "actor.opt.ir",
        ],
        "actor.v",
    )
    run(
        [
            "iverilog",
            "-g2012",
            "-s",
            "testbench",
            "-o",
            "simulate",
            "actor.v",
            "testbench.v",
        ]
    )
    run(["vvp", "simulate"])
