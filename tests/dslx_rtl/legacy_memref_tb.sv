// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

module legacy_memref_tb;
  reg clk = 0, rst = 1;
  always #5 clk = ~clk;

  wire [1:0] a_req, b_req;
  wire a_req_valid, b_req_valid, a_resp_ready, b_resp_ready;
  reg [31:0] a_resp = 0, b_resp = 0;
  reg a_resp_valid = 0, b_resp_valid = 0;
  int cycle = 0, sent_a = 0, sent_b = 0, compared = 0;
  wire a_req_ready = (!a_resp_valid || a_resp_ready) && (cycle % 3 != 0);
  wire b_req_ready = (!b_resp_valid || b_resp_ready) && (cycle % 5 != 1);
  wire result_ready = cycle % 7 > 1;
  wire result_valid;

`ifdef TEST_DOT
  wire [31:0] result;
  `DUT dut(.clk(clk), .rst(rst),
    ._mem0__read_req(a_req), ._mem0__read_req_vld(a_req_valid), ._mem0__read_req_rdy(a_req_ready),
    ._mem0__read_resp(a_resp), ._mem0__read_resp_vld(a_resp_valid), ._mem0__read_resp_rdy(a_resp_ready),
    ._mem1__read_req(b_req), ._mem1__read_req_vld(b_req_valid), ._mem1__read_req_rdy(b_req_ready),
    ._mem1__read_resp(b_resp), ._mem1__read_resp_vld(b_resp_valid), ._mem1__read_resp_rdy(b_resp_ready),
    ._out0(result), ._out0_vld(result_valid), ._out0_rdy(result_ready));
`else
  wire [33:0] result;
  wire write_resp_ready;
  reg write_resp_valid = 0;
  wire write_ready = result_ready && (!write_resp_valid || write_resp_ready);
  `DUT dut(.clk(clk), .rst(rst),
    ._mem0__read_req(a_req), ._mem0__read_req_vld(a_req_valid), ._mem0__read_req_rdy(a_req_ready),
    ._mem0__read_resp(a_resp), ._mem0__read_resp_vld(a_resp_valid), ._mem0__read_resp_rdy(a_resp_ready),
    ._mem1__read_req(b_req), ._mem1__read_req_vld(b_req_valid), ._mem1__read_req_rdy(b_req_ready),
    ._mem1__read_resp(b_resp), ._mem1__read_resp_vld(b_resp_valid), ._mem1__read_resp_rdy(b_resp_ready),
    ._mem2__write_req(result), ._mem2__write_req_vld(result_valid), ._mem2__write_req_rdy(write_ready),
    ._mem2__write_resp_vld(write_resp_valid), ._mem2__write_resp_rdy(write_resp_ready));
`endif

  function [31:0] a_value(input integer trial, input integer addr);
    a_value = 9 + trial * 13 - addr * 7;
  endfunction

  function [31:0] b_value(input integer trial, input integer addr);
    b_value = (addr ^ 3) * 11 - trial * 5;
  endfunction

`ifdef TEST_DOT
  function [31:0] dot(input integer trial);
    dot = 17;
    for (integer i = 0; i < 4; i++)
      dot += a_value(trial, i) * b_value(trial, i);
  endfunction
`endif

  always @(posedge clk) begin
    if (!rst) begin
      cycle <= cycle + 1;
      if (a_resp_valid && a_resp_ready) a_resp_valid <= 0;
      if (b_resp_valid && b_resp_ready) b_resp_valid <= 0;
      if (a_req_valid && a_req_ready) begin
        if (a_req !== sent_a[1:0]) $fatal(1, "left RAM request order: %0d", sent_a);
        a_resp <= a_value(sent_a / 4, int'(a_req));
        a_resp_valid <= 1;
        sent_a <= sent_a + 1;
      end
      if (b_req_valid && b_req_ready) begin
        if (b_req !== sent_b[1:0]) $fatal(1, "right RAM request order: %0d", sent_b);
        b_resp <= b_value(sent_b / 4, int'(b_req));
        b_resp_valid <= 1;
        sent_b <= sent_b + 1;
      end
`ifdef TEST_DOT
      if (result_valid && result_ready) begin
        if (result !== dot(compared))
          $fatal(1, "dot index=%0d expected=%h got=%h", compared, dot(compared), result);
        compared <= compared + 1;
        if (compared == 5) begin
          $display("COMPARE compared=6 mismatches=0");
          $finish;
        end
      end
`else
      if (write_resp_valid && write_resp_ready) write_resp_valid <= 0;
      if (result_valid && write_ready) begin
        if (result[33:32] !== compared[1:0] ||
            result[31:0] !== (a_value(compared / 4, compared % 4) + b_value(compared / 4, compared % 4)))
          $fatal(1, "vector index=%0d addr=%h got=%h", compared, result[33:32], result[31:0]);
        write_resp_valid <= 1;
        compared <= compared + 1;
        if (compared == 23) begin
          $display("COMPARE compared=24 mismatches=0");
          $finish;
        end
      end
`endif
    end
  end

  initial begin
    repeat (4) @(negedge clk);
    rst = 0;
    #20000;
    $fatal(1, "timeout: left=%0d right=%0d compared=%0d", sent_a, sent_b, compared);
  end
endmodule
