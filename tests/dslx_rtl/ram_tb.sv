// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

module ram_tb;
  timeunit 1ns;
  timeprecision 1ps;

  reg clk = 0, clear = 1;
  always #5 clk = ~clk;

  reg [19:0] command = 0;
  reg command_valid = 0;
  reg answer_ready = 0;
  wire command_ready, answer_valid;
  wire [15:0] answer;
  reg bridge_held = 0;
  reg [15:0] held_word;
  int bridge_stalls = 0;

  ram_check dut(.clk(clk), .clear(clear),
                ._command(command), ._command_vld(command_valid), ._command_rdy(command_ready),
                ._answer(answer), ._answer_vld(answer_valid), ._answer_rdy(answer_ready));

  always @(posedge clk) begin
    if (clear) begin
      bridge_held <= 0;
    end else begin
      if (bridge_held && (dut._first_resp_vld !== 1'b1 || dut._first_resp !== held_word))
        $fatal(1, "RAM response changed under backpressure");
      bridge_held <= dut._first_resp_vld && !dut._first_resp_rdy;
      if (dut._first_resp_vld && !dut._first_resp_rdy) begin
        held_word <= dut._first_resp;
        bridge_stalls <= bridge_stalls + 1;
      end
    end
  end

  task submit(input bank, input write, input [1:0] addr, input [15:0] value);
    @(negedge clk);
    command = {bank, write, addr, value};
    command_valid = 1;
    @(posedge clk);
    while (command_ready !== 1'b1) @(posedge clk);
    @(negedge clk);
    command_valid = 0;
  endtask

  task receive(input [15:0] expected);
    while (answer_valid !== 1'b1) @(negedge clk);
    repeat (4) begin
      if (answer_valid !== 1'b1 || answer !== expected)
        $fatal(1, "stalled read expected=%04h got=%04h valid=%b", expected, answer, answer_valid);
      @(negedge clk);
    end
    answer_ready = 1;
    @(negedge clk);
    answer_ready = 0;
  endtask

  task check_read(input bank, input [1:0] addr, input [15:0] expected);
    submit(bank, 1'b0, addr, 16'd0);
    receive(expected);
  endtask

  task check_idle;
    repeat (10) begin
      @(negedge clk);
      if (answer_valid !== 1'b0) $fatal(1, "unsolicited response");
    end
  endtask

  initial begin
    repeat (3) @(negedge clk);
    clear = 0;
    check_read(1'b0, 2'd0, 16'h0123);
    check_read(1'b0, 2'd1, 16'hfffe);
    check_read(1'b0, 2'd2, 16'hbeef);
    check_read(1'b1, 2'd1, 16'hffff);
    submit(1'b0, 1'b1, 2'd1, 16'h12ca);
    submit(1'b1, 1'b1, 2'd1, 16'h0456);
    check_idle();
    check_read(1'b0, 2'd1, 16'h12ca);
    check_read(1'b1, 2'd1, 16'h0456);
    submit(1'b0, 1'b1, 2'd3, 16'hbad1);
    check_idle();
    check_read(1'b0, 2'd3, 16'h0000);
    check_read(1'b0, 2'd2, 16'hbeef);

    fork
      begin
        submit(1'b0, 1'b0, 2'd0, 16'd0);
        submit(1'b0, 1'b0, 2'd1, 16'd0);
        submit(1'b0, 1'b0, 2'd2, 16'd0);
        submit(1'b0, 1'b0, 2'd0, 16'd0);
      end
      begin
        repeat (30) @(negedge clk);
        receive(16'h0123);
        receive(16'h12ca);
        receive(16'hbeef);
        receive(16'h0123);
      end
    join
    if (bridge_stalls == 0) $fatal(1, "RAM response channel was never stalled");

    submit(1'b0, 1'b0, 2'd1, 16'd0);
    while (answer_valid !== 1'b1) @(negedge clk);
    clear = 1;
    repeat (3) @(negedge clk);
    clear = 0;
    check_idle();
    check_read(1'b0, 2'd1, 16'hfffe);
    check_read(1'b1, 2'd1, 16'hffff);
    check_read(1'b0, 2'd0, 16'h0123);
    $display("RAM handshake and reset passed");
    $finish;
  end

  initial begin
    repeat (3000) @(posedge clk);
    $fatal(1, "RAM test timed out");
  end
endmodule
