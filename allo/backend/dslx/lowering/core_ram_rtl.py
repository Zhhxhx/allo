# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind external RAM channels in an XLS datapath to synthesizable RTL."""

from collections import Counter
import re


def _literal(value, width):
    return f"{width}'h{value & ((1 << width) - 1):x}"


def _range(width):
    return "" if width == 1 else f" [{width - 1}:0]"


def _pin(channel):
    return f"_{channel}"


def _channel_signals(channel, width, direction):
    pin = _pin(channel)
    opposite = "output" if direction == "input" else "input"
    return (
        (pin, direction, width),
        (f"{pin}_vld", direction, 1),
        (f"{pin}_rdy", opposite, 1),
    )


def _initial_word(ram, index):
    if isinstance(ram.initial, int):
        return "", _literal(ram.initial, ram.width)
    if len(ram.initial) != ram.size:
        raise ValueError(f"RAM {ram.name} requires {ram.size} initial words")

    words = [_literal(value, ram.width) for value in ram.initial]
    default, count = Counter(words).most_common(1)[0]
    if count == ram.size:
        return "", default

    name = f"allo_initial_{index}"
    values = "\n".join(
        f"        {ram.address_width}'d{address}: {name} = {word};"
        for address, word in enumerate(words)
        if word != default
    )
    function = f"""  function automatic [{ram.width - 1}:0] {name};
    input [{ram.address_width - 1}:0] addr;
    begin
      case (addr)
{values}
        default: {name} = {default};
      endcase
    end
  endfunction"""
    request = _pin(ram.request)
    address_slice = f"{request}[{ram.width} +: {ram.address_width}]"
    return function, f"{name}({address_slice})"


def _instantiate_ram(ram, index, bridge, reset, preserve_static_memories):
    if ram.width < 1 or ram.address_width < 1:
        raise ValueError(f"RAM {ram.name} requires positive data and address widths")
    if ram.size < 1 or ram.size > (1 << ram.address_width):
        raise ValueError(f"RAM {ram.name} size does not fit its address width")

    function, initial = _initial_word(ram, index)
    request, response = _pin(ram.request), _pin(ram.response)
    retain = int(preserve_static_memories and ram.is_static)
    binding = f"""  {bridge} #(
    .DATA_WIDTH({ram.width}), .ADDR_WIDTH({ram.address_width}), .DEPTH({ram.size}),
    .RETAIN_WRITES({retain})
  ) allo_ram_{index} (
    .clk(clk), .reset({reset}),
    .request({request}), .request_valid({request}_vld), .request_ready({request}_rdy),
    .response({response}), .response_valid({response}_vld), .response_ready({response}_rdy),
    .initial_word({initial})
  );"""
    return "\n".join(part for part in (function, binding) if part)


def wrap_ram_rtl(
    rtl: str,
    design_name: str,
    external_ports,
    rams,
    reset="rst",
    *,
    preserve_static_memories=False,
) -> str:
    """Expose the datapath's external channels and bind each RAM request/response.

    Channel names in the input descriptors are DSLX names. The corresponding
    XLS RTL ports have a leading underscore and valid/ready suffixes. Each RAM
    accepts {is_write, address, value}; only reads produce responses.
    With static preservation, backing RAMs start from the declared initial
    image at FPGA power-up and retain written words across control resets.
    """
    datapath = f"{design_name}_datapath"
    if not re.search(rf"\bmodule\s+{re.escape(datapath)}\s*\(", rtl):
        raise ValueError(f"XLS RTL does not define {datapath}")

    port_decls = ["input wire clk", f"input wire {reset}"]
    connections = [".clk(clk)", f".{reset}({reset})"]
    for port in external_ports:
        direction = {"in": "input", "out": "output"}[port.direction]
        for pin, flow, width in _channel_signals(port.name, port.width, direction):
            port_decls.append(f"{flow} wire{_range(width)} {pin}")
            connections.append(f".{pin}({pin})")

    bridge = f"{design_name}_allo_ram_bridge"
    declarations = []
    bindings = []
    for index, ram in enumerate(rams):
        request_width = 1 + ram.address_width + ram.width
        request = _channel_signals(ram.request, request_width, "output")
        response = _channel_signals(ram.response, ram.width, "input")
        for pin, _, width in (*request, *response):
            declarations.append(f"  wire{_range(width)} {pin};")
            connections.append(f".{pin}({pin})")
        bindings.append(
            _instantiate_ram(ram, index, bridge, reset, preserve_static_memories)
        )

    port_list = ",\n  ".join(port_decls)
    wires = "\n".join(declarations)
    pin_list = ",\n    ".join(connections)
    instances = "\n\n".join(bindings)
    wrapper = f"""module {design_name}(
  {port_list}
);
{wires}
  {datapath} datapath (
    {pin_list}
  );
{instances}
endmodule
"""
    helper = _RAM_BRIDGE.replace("ALLO_RAM_BRIDGE", bridge) if bindings else ""
    return f"{rtl}\n{wrapper}{helper}"


_RAM_BRIDGE = """
module ALLO_RAM_BRIDGE #(
  parameter integer DATA_WIDTH = 32,
  parameter integer ADDR_WIDTH = 1,
  parameter integer DEPTH = 1,
  parameter RETAIN_WRITES = 1'b0
)(
  input wire clk,
  input wire reset,
  input wire [ADDR_WIDTH+DATA_WIDTH:0] request,
  input wire request_valid,
  output wire request_ready,
  output wire [DATA_WIDTH-1:0] response,
  output reg response_valid,
  input wire response_ready,
  input wire [DATA_WIDTH-1:0] initial_word
);
  reg [DATA_WIDTH-1:0] memory [0:DEPTH-1];
  reg [DEPTH-1:0] written;
  reg [DATA_WIDTH-1:0] read_data, read_initial;
  reg read_written;

  wire write_cmd = request[ADDR_WIDTH+DATA_WIDTH];
  wire [ADDR_WIDTH-1:0] address = request[DATA_WIDTH +: ADDR_WIDTH];
  localparam [ADDR_WIDTH:0] LIMIT = DEPTH;
  wire in_bounds = {1'b0, address} < LIMIT;
  wire accepted = request_valid && request_ready;
  wire read_enable = accepted && !write_cmd;
  wire write_enable = accepted && write_cmd && in_bounds;

  assign request_ready = !reset && !response_valid;
  assign response = read_written ? read_data : read_initial;

  // The write array and its synchronous output register have no reset path.
  // Static RAMs retain written-word tags across a control reset when enabled.
  initial begin
    if (RETAIN_WRITES)
      written = {DEPTH{1'b0}};
  end
  always @(posedge clk) begin
    if (write_enable)
      memory[address] <= request[DATA_WIDTH-1:0];
    if (read_enable && in_bounds)
      read_data <= memory[address];
  end

  always @(posedge clk) begin
    if (reset) begin
      response_valid <= 1'b0;
      if (!RETAIN_WRITES)
        written <= {DEPTH{1'b0}};
    end else begin
      if (write_enable)
        written[address] <= 1'b1;
      if (response_valid && response_ready)
        response_valid <= 1'b0;
      if (read_enable) begin
        response_valid <= 1'b1;
        read_written <= in_bounds && written[address];
        read_initial <= in_bounds ? initial_word : {DATA_WIDTH{1'b0}};
      end
    end
  end
endmodule
"""
