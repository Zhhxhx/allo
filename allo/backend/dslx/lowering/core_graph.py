# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Connect independently scheduled actors using the dataflow top's actual buffers."""

from dataclasses import dataclass, field
from math import prod
import re

from allo._mlir import ir

from .core_actor import ActorLowerer, ActorPort, prepare_actor_module, width
from .float32 import FP32_DSLX
from .mlir_to_dslx_fn import function_name
from .straight_line_actor import StraightLineActorLowerer


@dataclass(frozen=True)
class Channel:
    name: str
    width: int
    direction: str

    @property
    def declaration(self):
        return f"{self.name}: chan<uN[{self.width}]> {self.direction}"


@dataclass
class Network:
    source: str
    ports: list
    rams: list
    actors: list
    fifos: list
    top: str
    topology: str
    definitions: dict


@dataclass
class Wire:
    names: tuple
    width: int
    depth: int
    external: bool
    uses: list = field(default_factory=list)


def _wire(typ, stem, external=False):
    if not ir.MemRefType.isinstance(typ):
        raise ValueError(f"dataflow channel {stem} must have a memref type, got {typ}")
    ref = ir.MemRefType(typ)
    if any(n <= 0 for n in ref.shape):
        raise ValueError(f"dynamic channel shape on {stem}: {ref.shape}")
    space = ref.memory_space
    spec = space.value if space is not None and ir.StringAttr.isinstance(space) else ""
    if spec:
        match = re.fullmatch(r"stream:(\d+)(?:;([ST]+))?", spec)
        if not match:
            raise ValueError(f"unsupported dataflow channel layout on {stem}: {spec}")
        depth = int(match[1])
        axes = match[2] or ("T" * len(ref.shape))
        if len(axes) != len(ref.shape) or any(
            mark == "T" and size != 1 for mark, size in zip(axes, ref.shape)
        ):
            raise ValueError(f"unsupported temporal channel dimension on {stem}: {typ}")
    elif external and prod(ref.shape) == 1:
        depth = 0
    else:
        raise ValueError(
            f"expected an internal FIFO or an external scalar sample: {stem} {typ}"
        )
    count = prod(ref.shape)
    names = (stem,) if count == 1 else tuple(f"{stem}_lane{i}" for i in range(count))
    return Wire(names, width(ref.element_type), depth, external)


def _external_wires(top):
    external_names = (
        top.attributes["inputs"].value.split(",")
        if "inputs" in top.attributes
        else [f"arg{i}" for i in range(len(top.arguments))]
    )
    if len(external_names) != len(top.arguments):
        raise ValueError(
            f"channel name count disagrees with the signature of @{top.name.value}"
        )
    return {
        arg: _wire(arg.type, function_name(name), True)
        for arg, name in zip(top.arguments, external_names)
    }


def _emit_actor(module, function, symbol, ports):
    arguments = module, function, symbol, ports
    try:
        sampling_actor = StraightLineActorLowerer(*arguments)
        return sampling_actor, sampling_actor.lower(include_helpers=False)
    except NotImplementedError:
        lowerer = ActorLowerer(*arguments)
        return lowerer, lowerer.lower()


def build_actor(module, top_name):
    """Compile a standalone actor with the same channel ABI as the dataflow graph."""
    normalized = prepare_actor_module(module)
    top = next(
        op
        for op in normalized.body.operations
        if op.operation.name == "func.func" and op.name.value == top_name
    )
    wires = list(_external_wires(top).values())
    bindings = {i: ActorPort(wire.names) for i, wire in enumerate(wires)}
    symbol = function_name(top_name)
    lowerer, source = _emit_actor(normalized, top, symbol, bindings)
    ports = [
        Channel(n, wires[i].width, direction)
        for i, direction in sorted(lowerer.directions.items())
        for n in wires[i].names
    ]
    if lowerer.uses_float:
        source = FP32_DSLX + "\n" + source
    return Network(
        source,
        ports,
        lowerer.rams,
        [(top_name, symbol)],
        [],
        top_name,
        "",
        {symbol: source},
    )


def build_network(module, top_name):
    normalized = prepare_actor_module(module)
    functions = {
        op.attributes["sym_name"].value: op
        for op in normalized.body.operations
        if op.operation.name == "func.func"
    }
    top = functions[top_name]
    if len(top.body.blocks) != 1 or top.type.results:
        raise ValueError(
            "a dataflow network must have a single wiring block with no return value"
        )
    nets = _external_wires(top)
    actors, rams, spawns, sources = [], [], [], {}
    ram_ports = []
    globals_owner = {}
    float_needed = False
    for op in top.entry_block.operations:
        kind = op.operation.name
        if kind in {"memref.alloc", "memref.alloca"}:
            nets[op.result] = _wire(op.result.type, f"fifo{len(nets)}")
        elif kind == "func.call":
            callee = op.callee.value
            if callee not in functions or len(op.results):
                raise ValueError(
                    f"dataflow call must name a defined actor without results: {op}"
                )
            symbol = f"Actor{len(actors)}_{function_name(callee)}"
            if any(v not in nets for v in op.operands):
                raise ValueError(f"unbound channel in call to @{callee}")
            edges = [nets[v] for v in op.operands]
            bindings = {
                i: ActorPort(tuple(f"p{i}_{j}" for j in range(len(w.names))))
                for i, w in enumerate(edges)
            }
            lowerer, body = _emit_actor(normalized, functions[callee], symbol, bindings)
            endpoints = []
            for i, direction in sorted(lowerer.directions.items()):
                wire = edges[i]
                wire.uses.append(direction)
                endpoints.extend(
                    n if wire.external else f"{n}_{direction}" for n in wire.names
                )
            for ram in lowerer.rams:
                endpoints.extend((ram.request, ram.response))
                ram_ports.extend(
                    (
                        Channel(ram.request, 1 + ram.address_width + ram.width, "out"),
                        Channel(ram.response, ram.width, "in"),
                    )
                )
            for name in lowerer.global_memories:
                if name in globals_owner:
                    raise ValueError(
                        f"global @{name} is shared between {globals_owner[name]} and {callee}"
                    )
                globals_owner[name] = callee
            actors.append((callee, symbol))
            sources[symbol] = body
            float_needed |= lowerer.uses_float
            rams.extend(lowerer.rams)
            spawns.append(f"    spawn {symbol}({', '.join(endpoints)});")
        elif kind != "func.return":
            raise NotImplementedError(
                f"unsupported @{top_name} dataflow wiring operation: {kind}"
            )
    wires, ports, fifos = [], [], []
    for wire in nets.values():
        if wire.external:
            if len(wire.uses) != 1:
                raise ValueError(
                    f"expected one actor on external {wire.names[0]}: {wire.uses}"
                )
            ports.extend(Channel(n, wire.width, wire.uses[0]) for n in wire.names)
        elif wire.uses:
            if sorted(wire.uses) != ["in", "out"]:
                raise ValueError(
                    f"expected a reader and a writer on {wire.names[0]}: {wire.uses}"
                )
            # A stored FIFO breaks combinational ready/valid cycles between actors.
            depth = wire.depth
            policy = (
                f"depth={depth}, bypass={str(depth == 0).lower()}, "
                f"register_push_outputs={str(depth > 0).lower()}, "
                f"register_pop_outputs={str(depth >= 2).lower()}"
            )
            for n in wire.names:
                wires.append(
                    f'    let ({n}_out, {n}_in) = #[channel({policy})] chan<uN[{wire.width}]>("{n}");'
                )
                fifos.append((n, wire.width, wire.depth))
    params = ", ".join(p.declaration for p in ports + ram_ports)
    topology = "\n".join(
        [
            f"pub proc {function_name(top_name)} {{",
            f"  config({params}) {{",
            *wires,
            *spawns,
            "    ()",
            "  }",
            "  init { () }",
            "  next(state: ()) { () }",
            "}",
        ]
    )
    source = "\n\n".join(
        ([FP32_DSLX] if float_needed else []) + list(sources.values()) + [topology]
    )
    if fifos:
        source = "#![feature(channel_attributes)]\n" + source
    return Network(source, ports, rams, actors, fifos, top_name, topology, sources)
