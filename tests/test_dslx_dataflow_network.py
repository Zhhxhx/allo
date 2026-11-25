# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess

import pytest

from allo.backend.dslx import DslxProcModule


_NETWORK = """
module {
  func.func @owner(%control: memref<1xi32, "stream:2">,
                   %request: memref<1xi32, "stream:2">,
                   %reply: memref<1xi32, "stream:2">,
                   %output: memref<1xi32, "stream:2">) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : i32
    %input = memref.load %control[%zero] : memref<1xi32, "stream:2">
    memref.store %input, %request[%zero] : memref<1xi32, "stream:2">
    %value = memref.load %reply[%zero] : memref<1xi32, "stream:2">
    %result = arith.addi %value, %one : i32
    memref.store %result, %output[%zero] : memref<1xi32, "stream:2">
    return
  }
  func.func @worker(%request: memref<1xi32, "stream:2">,
                    %reply: memref<1xi32, "stream:2">) {
    %zero = arith.constant 0 : index
    %three = arith.constant 3 : i32
    %high_bit = arith.constant -2147483648 : i32
    %input = memref.load %request[%zero] : memref<1xi32, "stream:2">
    %product = arith.muli %input, %three : i32
    %result = arith.xori %product, %high_bit : i32
    memref.store %result, %reply[%zero] : memref<1xi32, "stream:2">
    return
  }
  func.func @pipeline(%control: memref<1xi32, "stream:2">,
                      %output: memref<1xi32, "stream:2">)
                   attributes {dataflow, inputs="control,output_stream"} {
    %request = memref.alloc() : memref<1xi32, "stream:2">
    %reply = memref.alloc() : memref<1xi32, "stream:2">
    func.call @owner(%control, %request, %reply, %output)
      : (memref<1xi32, "stream:2">, memref<1xi32, "stream:2">,
         memref<1xi32, "stream:2">, memref<1xi32, "stream:2">) -> ()
    func.call @worker(%request, %reply)
      : (memref<1xi32, "stream:2">, memref<1xi32, "stream:2">) -> ()
    return
  }
}
"""

_TESTBENCH = """
module tb;
  reg clk = 0;
  always #5 clk = !clk;
  reg rst = 1;
  reg [31:0] command = 0;
  reg command_valid = 0, result_ready = 0;
  wire command_ready, result_valid;
  wire [31:0] result;
  pipeline dut(.clk(clk), .rst(rst),
    ._control(command), ._control_vld(command_valid), ._control_rdy(command_ready),
    ._output_stream(result), ._output_stream_vld(result_valid), ._output_stream_rdy(result_ready));

  task transfer(input [31:0] input_word, input [31:0] expected);
    begin
      @(negedge clk);
      command = input_word;
      command_valid = 1;
      @(posedge clk);
      while (!command_ready) @(posedge clk);
      @(negedge clk);
      command_valid = 0;
      wait(result_valid);
      repeat (7) begin
        @(negedge clk);
        if (!result_valid || result !== expected)
          $fatal(1, "Expected %h while stalled, got %h with valid=%b", expected, result, result_valid);
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
    transfer(32'h00000003, 32'h8000000a);
    transfer(32'h80000000, 32'h00000001);
    transfer(32'hffffffff, 32'h7ffffffe);
    $finish;
  end
  initial begin
    #5000;
    $fatal(1, "Feedback channel failed to make progress");
  end
endmodule
"""


def test_registered_feedback_network(tmp_path):
    for tool in ("ir_converter_main", "opt_main", "codegen_main", "iverilog", "vvp"):
        if shutil.which(tool) is None:
            pytest.skip(f"{tool} is required")
    pipeline = DslxProcModule(_NETWORK, "pipeline")
    pipeline._out_dir = tmp_path
    ir = pipeline.to_ir(verbose=False).stdout
    channels = [line.strip() for line in ir.splitlines() if "fifo_depth=" in line]
    assert len(channels) == 2
    for channel in channels:
        assert "fifo_depth=2" in channel
        assert "bypass=false" in channel
        assert "register_push_outputs=true" in channel
        assert "register_pop_outputs=true" in channel
    pipeline.opt(verbose=False)
    pipeline.to_vlog(verbose=False, delay_model="unit")

    (tmp_path / "tb.v").write_text(_TESTBENCH, encoding="utf-8")
    compile_sim = subprocess.run(
        ["iverilog", "-g2012", "-s", "tb", "-o", "sim", "pipeline.v", "tb.v"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=45,
        check=False,
    )
    assert compile_sim.returncode == 0, compile_sim.stdout + compile_sim.stderr
    simulate = subprocess.run(
        ["vvp", "sim"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=45,
        check=False,
    )
    assert simulate.returncode == 0, simulate.stdout + simulate.stderr
