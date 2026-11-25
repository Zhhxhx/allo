// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

module oracle_tb;
  import otpu_fp::*;

  string input_path, operation;
  int vectors, input_file, fields;
  logic [31:0] left, right, result;

  initial begin
    if (!$value$plusargs("input=%s", input_path) ||
        !$value$plusargs("op=%s", operation) ||
        !$value$plusargs("count=%d", vectors))
      $fatal(1, "require +input=<words> +op=add|mul +count=<vectors>");
    if (operation != "add" && operation != "mul")
      $fatal(1, "invalid operation %s", operation);
    input_file = $fopen(input_path, "r");
    if (input_file == 0) $fatal(1, "cannot open %s", input_path);
    for (int index = 0; index < vectors; index++) begin
      fields = $fscanf(input_file, "%h %h", left, right);
      if (fields != 2) $fatal(1, "bad vector at index %0d", index);
      result = (operation == "add") ? fp_add(left, right) : fp_mul(left, right);
      $display("ORACLE %0d %08h %08h %08h", index, left, right, result);
    end
    $fclose(input_file);
    $finish;
  end
endmodule
