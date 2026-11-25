# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lower straight-line sampling actors directly to one XLS proc tick."""

from dataclasses import dataclass

from allo._mlir import ir
from allo._mlir.dialects import func as func_d

from .core_actor import bit_expression, constant, fail, integer, literal, tuple_expr
from .float32 import FP32_DSLX
from .mlir_to_dslx_fn import dslx_type, function_name, scalar_expression
from .scalar import scalar_type


@dataclass(eq=False)
class _Cell:
    typ: str
    value: str | None = None
    port_name: str | None = None


class StraightLineActorLowerer:
    """Inline a sampling actor's calls into one state transition per tick.

    The input is a normalized actor with singleton write-only channel arguments.
    Local singleton allocations live for one invocation. Global singleton
    allocations come from dense integer initializers and persist across ticks.
    External channels are blocking sends, so a free-running actor requires its
    output consumers to remain ready.
    """

    def __init__(self, module, function, symbol, ports):
        self.module = module
        self.function = function
        self.symbol = function_name(symbol)
        self.ports = ports
        self.rams = ()
        self._functions = {
            op.name.value: op
            for op in module.body.operations
            if isinstance(op, func_d.FuncOp)
        }
        self._globals = {
            op.attributes["sym_name"].value: op
            for op in module.body.operations
            if op.operation.name == "memref.global"
        }
        self._reset()

    def _reset(self):
        self.directions = {}
        self.global_memories = {}
        self.uses_float = False
        self._globals_initial = []
        self._active = set()
        self._lines = []
        self._sent = []

    @staticmethod
    def _cell_type(typ, op):
        if not ir.MemRefType.isinstance(typ):
            fail(op, f"Straight-line actor expected a singleton memref, got {typ}")
        ref = ir.MemRefType(typ)
        if list(ref.shape) not in ([], [1]):
            fail(op, f"Straight-line actor only supports singleton memrefs, got {typ}")
        try:
            dslx_type(ref)
        except ValueError as exc:
            fail(op, str(exc))
        return scalar_type(ref.element_type)

    def _global(self, op):
        symbol = op.attributes["name"].value
        memory = self.global_memories.get(symbol)
        if memory is not None:
            return memory
        decl = self._globals.get(symbol)
        if decl is None or "initial_value" not in decl.attributes:
            fail(op, f"Global '{symbol}' needs a local dense initializer")
        attr = decl.attributes["initial_value"]
        if not ir.DenseIntElementsAttr.isinstance(attr):
            fail(decl, "Straight-line actor globals require dense integer initializers")
        dense = ir.DenseIntElementsAttr(attr)
        if not dense.is_splat:
            fail(decl, "Straight-line actor globals require a singleton initializer")
        memref = ir.MemRefType(op.results[0].type)
        typ = self._cell_type(memref, op)
        width = ir.IntegerType(memref.element_type).width
        signless = integer(dense.get_splat_value()) % (1 << width)
        register = f"s{len(self._globals_initial)}"
        self._globals_initial.append((register, typ, literal(typ, signless)))
        memory = _Cell(typ, f"state.{register}")
        self.global_memories[symbol] = memory
        return memory

    def _let(self, expression, typ):
        name = f"_v{len(self._lines)}"
        self._lines.append(f"    let {name}: {typ} = {expression};")
        return name

    def _access(self, op, values, memories):
        load = op.operation.name == "memref.load"
        if load:
            base, *indices = op.operands
        else:
            value, base, *indices = op.operands
        cell = memories.get(base)
        if cell is None:
            fail(op, "Unknown straight-line actor memref")
        if any(constant(idx) != 0 for idx in indices):
            fail(op, "Straight-line actor memref index must be statically zero")
        if load:
            if cell.port_name is not None:
                fail(op, "Straight-line actor arguments are write-only outputs")
            if cell.value is None:
                fail(op, "Read from an uninitialized straight-line actor memref")
            values[op.results[0]] = cell.value
            return
        if cell.port_name is not None:
            if cell.value is not None:
                fail(
                    op, f"Output '{cell.port_name}' is written more than once per tick"
                )
            self._sent.append(cell)
        cell.value = values[value]

    def _inline(self, function, args):
        symbol = function.name.value
        if symbol in self._active:
            fail(function, f"Nested recursive call to '{symbol}' is unsupported")
        if len(function.body.blocks) != 1:
            fail(function, "Expected a single defined block in a straight-line actor")
        self._active.add(symbol)
        values = {}
        memories = {}
        for arg, binding in zip(function.arguments, args, strict=True):
            if ir.MemRefType.isinstance(arg.type):
                memories[arg] = binding
            else:
                values[arg] = binding

        for op in function.body.blocks[0].operations:
            name = op.operation.name
            if name in {"memref.alloc", "memref.alloca"}:
                val = op.results[0]
                memories[val] = _Cell(self._cell_type(val.type, op))
            elif name == "memref.get_global":
                memories[op.results[0]] = self._global(op)
            elif name == "memref.dealloc":
                del memories[op.operands[0]]
            elif name in {"memref.load", "memref.store"}:
                self._access(op, values, memories)
            elif name == "func.call":
                callee = op.attributes["callee"].value
                if callee not in self._functions:
                    fail(op, f"Undefined straight-line actor helper '{callee}'")
                if any(ir.MemRefType.isinstance(res.type) for res in op.results):
                    fail(
                        op,
                        "Returning a memref from a straight-line actor is unsupported",
                    )
                bindings = [
                    memories[arg] if ir.MemRefType.isinstance(arg.type) else values[arg]
                    for arg in op.operands
                ]
                returned = self._inline(self._functions[callee], bindings)
                values.update(zip(op.results, returned, strict=True))
            elif name == "func.return":
                self._active.remove(symbol)
                return [values[arg] for arg in op.operands]
            elif len(op.results) == 1:
                expression = bit_expression(op, values.__getitem__)
                if expression is None:
                    expression = scalar_expression(op, values.__getitem__)
                if expression is None:
                    fail(op, "Unsupported straight-line actor scalar operation")
                result = op.results[0]
                values[result] = self._let(expression, scalar_type(result.type))
                self.uses_float |= "__allo_fp32_" in expression
            else:
                fail(op, "Unsupported straight-line actor operation")
        fail(function, "Missing straight-line actor return")

    def lower(self):
        self._reset()
        if len(self.function.body.blocks) != 1:
            fail(
                self.function,
                "Expected a single defined block in a straight-line actor",
            )
        if self.function.type.results:
            fail(self.function, "Straight-line actor results must be channel writes")
        outputs = []
        names = set()
        for index, arg in enumerate(self.function.arguments):
            port = self.ports.get(index)
            if port is None or len(port.names) != 1:
                fail(self.function, f"Expected one channel lane for argument {index}")
            typ = self._cell_type(arg.type, self.function)
            name = port.names[0]
            if name != function_name(name):
                fail(self.function, f"Invalid straight-line actor output name '{name}'")
            if name in names:
                fail(self.function, f"Output '{name}' is bound to several arguments")
            names.add(name)
            outputs.append(_Cell(typ, port_name=name))
            self.directions[index] = "out"
        if not outputs:
            fail(self.function, "A straight-line sampling actor requires an output")
        self._inline(self.function, outputs)
        for cell in outputs:
            if cell.value is None:
                fail(self.function, f"Output '{cell.port_name}' is never written")
        return self._render(outputs)

    def _render(self, outputs):
        state_type = f"{self.symbol}State" if self._globals_initial else "()"
        ports = [f"{cell.port_name}: chan<{cell.typ}> out" for cell in outputs]
        lines = []
        if self.uses_float:
            lines.append(FP32_DSLX)
        if self._globals_initial:
            lines.append(f"struct {state_type} {{")
            lines.extend(f"  {reg}: {typ}," for reg, typ, _ in self._globals_initial)
            lines += ["}", ""]
        lines.append(f"proc {self.symbol} {{")
        lines.extend(f"  {port};" for port in ports)
        lines += [
            f"  config({', '.join(ports)}) {{",
            "    " + tuple_expr([cell.port_name for cell in outputs]),
            "  }",
        ]
        if self._globals_initial:
            lines += [f"  init {{ {state_type} {{"]
            lines.extend(f"    {reg}: {val}," for reg, _, val in self._globals_initial)
            lines += ["  } }", f"  next(state: {state_type}) {{"]
        else:
            lines += ["  init { () }", "  next(_: ()) {"]
        lines.extend(self._lines)
        token = "join()"
        for index, cell in enumerate(self._sent):
            lines.append(
                f"    let _token{index} = send({token}, {cell.port_name}, {cell.value});"
            )
            token = f"_token{index}"
        if self._globals_initial:
            lines.append(f"    {state_type} {{")
            for (reg, _, _), cell in zip(
                self._globals_initial, self.global_memories.values(), strict=True
            ):
                lines.append(f"      {reg}: {cell.value},")
            lines.append("    }")
        else:
            lines.append("    ()")
        lines += ["  }", "}"]
        return "\n".join(lines) + "\n"
