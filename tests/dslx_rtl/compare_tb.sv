// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

module compare_tb;
  timeunit 1ns;
  timeprecision 1ps;
  localparam int WIDTH = `INPUT_BITS;
  localparam int COUNT = `COUNT;

  reg clk = 0, rst = 1;
  always #5 clk = ~clk;

  reg [2*WIDTH+31:0] rows [0:COUNT-1];
  reg [WIDTH-1:0] left = 0, right = 0;
  wire [31:0] result;
  int compared = 0, mismatches = 0;

`ifdef PROC
  reg left_valid = 0, right_valid = 0, out_ready = 0;
  wire left_ready, right_ready, out_valid;
  `DUT dut(.clk(clk), .rst(rst),
           ._in0(left), ._in0_vld(left_valid), ._in0_rdy(left_ready),
           ._in1(right), ._in1_vld(right_valid), ._in1_rdy(right_ready),
           ._out0(result), ._out0_vld(out_valid), ._out0_rdy(out_ready));
  int cycle = 0, left_sent = 0, right_sent = 0, stalled = 0;
  reg left_accepted = 0, right_accepted = 0, previous_stalled = 0;
  reg [31:0] held_result;
`else
  `DUT dut(.clk(clk), .rst(rst), .arg0(left), .arg1(right), .out(result));
`endif

  task check_output;
    reg [31:0] expected;
    reg equal_to_expected;
    expected = rows[compared][31:0];
    equal_to_expected = (result === expected);
`ifdef IEEE_NAN
    if (expected[30:23] == 8'hff && expected[22:0] != 0)
      equal_to_expected = (result[30:22] === 9'h1ff);
`endif
    if (!equal_to_expected) begin
      $display("MISMATCH index=%0d left=%h right=%h expected=%08h actual=%08h",
               compared, rows[compared][2*WIDTH+31:WIDTH+32],
               rows[compared][WIDTH+31:32], expected, result);
      mismatches++;
    end
    compared++;
  endtask

  initial begin
`ifdef PROC
    if ($bits(dut._in0) != WIDTH || $bits(dut._in1) != WIDTH || $bits(dut._out0) != 32)
      $fatal(1, "incorrect proc port widths");
`else
    if ($bits(dut.arg0) != WIDTH || $bits(dut.arg1) != WIDTH || $bits(dut.out) != 32)
      $fatal(1, "incorrect function port widths");
`endif
    $readmemh("vectors.mem", rows);
    repeat (4) @(negedge clk);
    rst = 0;
`ifdef PROC
    while (compared < COUNT && cycle < 16 * COUNT + 100) begin
      if (!left_valid || left_accepted) begin
        left_valid = (left_sent < COUNT) && (cycle % 11 != 5);
        if (left_valid) left = rows[left_sent][2*WIDTH+31:WIDTH+32];
      end
      if (!right_valid || right_accepted) begin
        right_valid = (right_sent < COUNT) && (cycle % 7 > 1);
        if (right_valid) right = rows[right_sent][WIDTH+31:32];
      end
      out_ready = (cycle >= 12) && (cycle % 19 < 13);

      @(posedge clk);
      left_accepted = left_valid && left_ready;
      right_accepted = right_valid && right_ready;
      if (left_accepted) left_sent++;
      if (right_accepted) right_sent++;

      if (previous_stalled && (out_valid !== 1'b1 || result !== held_result))
        $fatal(1, "output changed under backpressure at cycle %0d", cycle);
      previous_stalled = out_valid && !out_ready;
      if (previous_stalled) begin
        stalled++;
        held_result = result;
      end

      if (out_valid && out_ready) begin
        if (compared >= COUNT || compared >= left_sent || compared >= right_sent)
          $fatal(1, "unexpected output at cycle %0d, index %0d", cycle, compared);
        check_output();
      end
      cycle++;
      @(negedge clk);
    end
    if (left_sent != COUNT || right_sent != COUNT || compared != COUNT)
      $fatal(1, "incomplete transactions: left=%0d right=%0d compared=%0d mismatches=%0d",
             left_sent, right_sent, compared, mismatches);
    left_valid = 0;
    right_valid = 0;
    out_ready = 1;
    repeat (8) begin
      @(posedge clk);
      if (out_valid !== 1'b0) $fatal(1, "extra output after %0d transactions", COUNT);
      @(negedge clk);
    end
    if (stalled == 0) $fatal(1, "test did not exercise output backpressure");
`else
    while (compared < COUNT) begin
      {left, right} = rows[compared][2*WIDTH+31:32];
      // Allow the fixed three-stage pipeline to settle for each input pair.
      repeat (8) @(negedge clk);
      check_output();
    end
`endif
    $display("COMPARE compared=%0d mismatches=%0d", compared, mismatches);
    $finish;
  end
endmodule
