# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Choose a process interface from the selected function and its callees."""

from math import prod

from allo._mlir import ir
from allo._mlir.dialects import func as func_d


def _nested(op):
    yield op
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _nested(child)


def _reachable(module, top):
    functions = {
        op.name.value: op
        for op in module.body.operations
        if isinstance(op, func_d.FuncOp)
    }
    pending = [top.name.value]
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if name not in functions:
            raise ValueError(f"cannot resolve call to @{name} from @{top.name.value}")
        fn = functions[name]
        yield fn
        pending.extend(
            op.callee.value for op in _nested(fn) if isinstance(op, func_d.CallOp)
        )


def _channel_function(module, top):
    from .mlir_to_dslx_fn import dslx_type, function_name, lower_functions

    used = {function_name(fn.name.value) for fn in _reachable(module, top)}
    body_name = "allo_function_body"
    while body_name in used:
        body_name += "_"
    source = lower_functions(module, top.name.value, emitted_top_name=body_name)
    inputs = [(f"in{i}", dslx_type(t)) for i, t in enumerate(top.type.inputs)]
    output_type = dslx_type(top.type.results[0])
    channels = [(name, f"chan<{typ}> in") for name, typ in inputs] + [
        ("out0", f"chan<{output_type}> out")
    ]
    fields = [f"{name}: {typ}" for name, typ in channels]
    names = [name for name, _ in channels]
    handles = ", ".join(names) + ("," if len(names) == 1 else "")
    lines = [source, f"pub proc {function_name(top.name.value)} {{"]
    lines.extend(f"  {field};" for field in fields)
    lines.extend(
        [
            f"  config({', '.join(fields)}) {{ ({handles}) }}",
            "  init { () }",
            "  next(state: ()) {",
        ]
    )
    tokens, values = [], []
    for i, (name, _) in enumerate(inputs):
        lines.append(f"    let (t{i}, a{i}) = recv(join(), {name});")
        tokens.append(f"t{i}")
        values.append(f"a{i}")
    lines.extend(
        [
            f"    send(join({', '.join(tokens)}), out0, {body_name}({', '.join(values)}));",
            "  }",
            "}",
        ]
    )
    return "\n".join(lines)


def emit_proc_module(module, top, *, legacy_lowerer):
    if (
        "dataflow" in top.attributes
        and not top.type.results
        and any(isinstance(op, func_d.CallOp) for op in top.entry_block.operations)
    ):
        from .core_graph import build_network

        graph = build_network(module, top.name.value)
        return graph.source, graph
    # Shaped legacy proc buffers expose address and data channels for each RAM.
    if (
        len(top.type.results) == 1
        and not any(
            ir.ShapedType.isinstance(typ)
            for typ in (*top.type.inputs, *top.type.results)
        )
        and not any(
            op.operation.name == "memref.get_global"
            for fn in _reachable(module, top)
            for op in _nested(fn)
        )
        and not any("stream" in str(t) for t in top.type.inputs)
    ):
        return _channel_function(module, top), None
    if (
        not top.type.results
        and top.type.inputs
        and all(
            ir.MemRefType.isinstance(t)
            and ("stream:" in str(t) or prod(ir.MemRefType(t).shape) == 1)
            for t in top.type.inputs
        )
    ):
        from .core_graph import build_actor

        graph = build_actor(module, top.name.value)
        return graph.source, graph
    return legacy_lowerer(module, top.name.value), None
