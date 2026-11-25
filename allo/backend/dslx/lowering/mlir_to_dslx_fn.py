"""Lower pure MLIR functions to DSLX with explicit SSA and local buffer state."""

import re

from allo._mlir import ir
from allo._mlir.dialects import func as func_d

from .float32 import FP32_DSLX
from .scalar import scalar_expression, scalar_type


_DSLX_RESERVED = {
    "as",
    "const",
    "else",
    "enum",
    "false",
    "fn",
    "for",
    "if",
    "impl",
    "import",
    "in",
    "let",
    "match",
    "proc",
    "pub",
    "struct",
    "true",
    "type",
    "unroll_for",
    "use",
    "uN",
    "sN",
    "bits",
    "bool",
    "token",
    "chan",
    "config",
    "init",
    "next",
    "spawn",
    "clz",
    "range",
    "update",
}
_STORE = {"affine.store", "memref.store"}
_LOAD = {"affine.load", "memref.load"}
_YIELD = {"affine.yield", "scf.yield"}


class DslxFunctionError(ValueError):
    """A function or operation cannot be represented by the pure DSLX backend."""


def function_name(name):
    """Keep ordinary function names and encode reserved or non-DSLX symbols."""
    if (
        re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*", name)
        and name not in _DSLX_RESERVED
        and not re.fullmatch(r"[us][0-9]+", name)
        and not name.startswith("__allo_")
    ):
        return name
    return f"__allo_fn_{name.encode().hex()}"


def dslx_type(typ):
    """Map a scalar or a statically shaped, ordinary memref to a DSLX value."""
    if not ir.MemRefType.isinstance(typ):
        return scalar_type(typ)
    memref = ir.MemRefType(typ)
    if memref.memory_space is not None:
        raise ValueError(f"unsupported memory space in {memref}")
    layout = memref.layout
    identity = ir.AffineMap.get_identity(memref.rank, context=typ.context)
    if (
        not ir.AffineMapAttr.isinstance(layout)
        or ir.AffineMapAttr(layout).value != identity
    ):
        raise ValueError(f"unsupported memref layout in {memref}")
    if any(dim < 1 for dim in memref.shape):
        raise ValueError(f"expected a nonempty, static memref shape in {memref}")
    return scalar_type(memref.element_type) + "".join(
        f"[{size}]" for size in reversed(memref.shape)
    )


def _walk(ops):
    for op in ops:
        yield op
        for region in op.regions:
            for block in region.blocks:
                yield from _walk(block.operations)


def _tuple(values):
    values = list(values)
    if len(values) == 1:
        return values[0]
    return f"({', '.join(values)})"


def _floor_div(lhs, rhs):
    return f"(({lhs} / {rhs}) - (({lhs} % {rhs} < s64:0) as s64))"


def _affine_expr(expr, dimensions, symbols):
    if ir.AffineDimExpr.isinstance(expr):
        return dimensions[ir.AffineDimExpr(expr).position]
    if ir.AffineSymbolExpr.isinstance(expr):
        return symbols[ir.AffineSymbolExpr(expr).position]
    if ir.AffineConstantExpr.isinstance(expr):
        return f"s64:{ir.AffineConstantExpr(expr).value}"
    if not ir.AffineBinaryExpr.isinstance(expr):
        raise ValueError(f"unsupported affine expression {expr}")
    binary = ir.AffineBinaryExpr(expr)
    lhs = _affine_expr(binary.lhs, dimensions, symbols)
    rhs = _affine_expr(binary.rhs, dimensions, symbols)
    if ir.AffineAddExpr.isinstance(expr):
        return f"({lhs} + {rhs})"
    if ir.AffineMulExpr.isinstance(expr):
        return f"({lhs} * {rhs})"

    # DSLX division truncates toward zero, whereas affine divisors are positive.
    if ir.AffineFloorDivExpr.isinstance(expr):
        return _floor_div(lhs, rhs)
    if ir.AffineCeilDivExpr.isinstance(expr):
        return f"(({lhs} / {rhs}) + (({lhs} % {rhs} > s64:0) as s64))"
    if ir.AffineModExpr.isinstance(expr):
        return f"({lhs} - ({_floor_div(lhs, rhs)} * {rhs}))"
    raise ValueError(f"unsupported affine expression {expr}")


def _constant_bound(value):
    if not isinstance(value, ir.OpResult) or value.owner.name != "arith.constant":
        raise ValueError("expected a statically bounded loop")
    return ir.IntegerAttr(value.owner.attributes["value"]).value


def _affine_bound(attr, combine):
    values = []
    for expr in ir.AffineMapAttr(attr).value.results:
        if not ir.AffineConstantExpr.isinstance(expr):
            raise ValueError("expected constant affine loop bounds")
        values.append(ir.AffineConstantExpr(expr).value)
    return combine(values)


class MlirToDslxLowerer:
    def __init__(self, func_op, *, symbols=None, emitted_name=None):
        self.func_op = func_op
        self.name = func_op.name.value
        self.emitted_name = emitted_name or function_name(self.name)
        self.symbols = symbols

    def error(self, op, message):
        return DslxFunctionError(
            f"DSLX function @{self.name}: {op.operation.name} at {op.location}: {message}"
        )

    def lower(self):
        """Emit this function and its reachable pure callees."""
        parent = self.func_op.operation.parent
        if parent is None:
            return _lower_functions({self.name: self.func_op}, self.name)
        return _lower_functions(
            _function_table(parent.regions[0].blocks[0].operations), self.name
        )

    def _emit_function(self):
        self.values = {}
        self.buffers = {}
        self.lines = []
        self.indent = 0
        self.serial = 0
        self.argument_buffers = set()

        if len(self.func_op.body.blocks) != 1:
            raise self.error(self.func_op, "expected one defined entry block")
        try:
            arguments = self.func_op.body.blocks[0].arguments
            params = []
            for i, arg in enumerate(arguments):
                name = f"arg{i}"
                params.append(f"{name}: {dslx_type(arg.type)}")
                if ir.MemRefType.isinstance(arg.type):
                    self.buffers[arg] = name
                    self.argument_buffers.add(arg)
                else:
                    self.values[arg] = name
            result_type = _tuple(dslx_type(typ) for typ in self.func_op.type.results)
        except ValueError as exc:
            raise self.error(self.func_op, str(exc)) from exc

        self.line(f"fn {self.emitted_name}({', '.join(params)}) -> {result_type} {{")
        self.indent = 1
        values = self.block(self.func_op.body.blocks[0], {"func.return"})
        self.line(_tuple(values))
        self.indent = 0
        self.line("}")
        return "\n".join(self.lines)

    def line(self, text):
        self.lines.append("  " * self.indent + text)

    def fresh(self):
        # A value can be live in MLIR yet unused after buffer state is lowered.
        self.serial += 1
        return f"_v{self.serial}"

    def assign(self, expression, typ=None):
        name = self.fresh()
        annotation = f": {typ}" if typ is not None else ""
        self.line(f"let {name}{annotation} = {expression};")
        return name

    def value(self, val):
        expr = (
            self.buffers.get(val)
            if ir.MemRefType.isinstance(val.type)
            else self.values.get(val)
        )
        if expr is None:
            raise ValueError(f"unbound or uninitialized value of type {val.type}")
        return expr

    def block(self, block, terminators):
        for op in block.operations:
            try:
                if op.operation.name in terminators:
                    return [self.value(val) for val in op.operands]
                self.lower_op(op)
            except DslxFunctionError:
                raise
            except ValueError as exc:
                raise self.error(op, str(exc)) from exc
        raise ValueError(
            f"missing expected block terminator {', '.join(sorted(terminators))}"
        )

    def capture(self, region, bindings=(), carried=()):
        old_lines, old_values, old_buffers = self.lines, self.values, self.buffers
        self.lines, self.values, self.buffers = [], dict(old_values), dict(old_buffers)
        self.values.update(bindings)
        self.buffers.update(carried)
        self.indent += 1
        try:
            if not region.blocks:
                return [], [], self.buffers
            if len(region.blocks) != 1:
                raise ValueError("expected a single-block structured region")
            yielded = self.block(region.blocks[0], _YIELD)
            return self.lines, yielded, self.buffers
        finally:
            self.indent -= 1
            self.lines, self.values, self.buffers = old_lines, old_values, old_buffers

    def rebind(self, results, buffers, packed):
        parts = (
            [packed]
            if len(results) + len(buffers) == 1
            else [f"{packed}.{i}" for i in range(len(results) + len(buffers))]
        )
        self.values.update(zip(results, parts))
        self.buffers.update(zip(buffers, parts[len(results) :]))

    def lower_op(self, op):
        name = op.operation.name
        if name in {"memref.alloc", "memref.alloca"}:
            val = op.results[0]
            typ = ir.MemRefType(val.type)
            annotation = dslx_type(typ)
            if typ.rank:
                # Fresh array elements are undefined in MLIR. Use zero bits as
                # their concrete values until a store writes them.
                self.buffers[val] = self.assign(
                    self.fill(typ, f"{scalar_type(typ.element_type)}:0"), annotation
                )
            else:
                self.buffers[val] = None
            return
        if name in _LOAD:
            base = op.operands[0]
            indexes = self.indexes(op, list(op.operands)[1:])
            expr = self.value(base) + "".join(f"[{index}]" for index in indexes)
            self.bind_result(op, expr)
            return
        if name in _STORE:
            value, base, *indices = op.operands
            self.check_store(base)
            indexes = self.indexes(op, indices)
            expr = self.value(value)
            if indexes:
                expr = self.update(self.value(base), indexes, expr)
            self.buffers[base] = self.assign(expr)
            return
        if name == "linalg.fill":
            if len(op.operands) != 2 or len(op.results):
                raise ValueError("expected a buffer-form linalg.fill with one output")
            value, base = op.operands
            self.check_store(base)
            self.buffers[base] = self.assign(
                self.fill(ir.MemRefType(base.type), self.value(value)),
                dslx_type(base.type),
            )
            return
        if name == "affine.apply":
            expressions = self.affine_map(op.attributes["map"], list(op.operands))
            self.bind_result(op, expressions[0])
            return
        if name in {"affine.for", "scf.for"}:
            self.lower_for(op)
            return
        if name == "scf.if":
            self.lower_if(op)
            return
        if name == "func.call":
            self.lower_call(op)
            return
        expression = scalar_expression(op, self.value)
        if expression is None:
            raise ValueError("unsupported operation in a pure function")
        self.bind_result(op, expression)

    def bind_result(self, op, expression):
        result = op.results[0]
        self.values[result] = self.assign(expression, dslx_type(result.type))

    def check_store(self, base):
        if base in self.argument_buffers:
            raise ValueError(
                "writes to memref arguments require an effectful interface"
            )
        if base not in self.buffers:
            raise ValueError(f"unsupported store through an alias of type {base.type}")

    @staticmethod
    def fill(typ, value):
        for size in reversed(typ.shape):
            value = f"[{value}, ...]" if size > 1 else f"[{value}]"
        return value

    def update(self, array, indices, value):
        index, *remaining = indices
        if remaining:
            value = self.update(f"{array}[{index}]", remaining, value)
        return f"update({array}, {index}, {value})"

    def affine_map(self, attr, operands):
        amap = ir.AffineMapAttr(attr).value
        values = [self.value(val) for val in operands]
        return [
            _affine_expr(expr, values[: amap.n_dims], values[amap.n_dims :])
            for expr in amap.results
        ]

    def indexes(self, op, operands):
        if op.operation.name.startswith("affine."):
            expressions = self.affine_map(op.attributes["map"], operands)
        else:
            expressions = [self.value(val) for val in operands]
        return [f"({expr} as u64)" for expr in expressions]

    @staticmethod
    def check_region_results(results):
        if any(ir.MemRefType.isinstance(val.type) for val in results):
            raise ValueError("memref aliases as call or region results are unsupported")

    def lower_if(self, op):
        results = list(op.results)
        self.check_region_results(results)
        then_lines, then_yield, then_buffers = self.capture(op.regions[0])
        else_lines, else_yield, else_buffers = self.capture(op.regions[1])
        buffers = [
            val
            for val in self.buffers
            if then_buffers[val] != else_buffers[val]
            and then_buffers[val] is not None
            and else_buffers[val] is not None
        ]
        if not results and not buffers:
            return
        condition = self.value(op.operands[0])
        packed = self.fresh()
        self.line(f"let {packed} = if {condition} {{")
        self.lines.extend(then_lines)
        self.line("  " + _tuple(then_yield + [then_buffers[val] for val in buffers]))
        self.line("} else {")
        self.lines.extend(else_lines)
        self.line("  " + _tuple(else_yield + [else_buffers[val] for val in buffers]))
        self.line("};")
        self.rebind(results, buffers, packed)

    def lower_for(self, op):
        results = list(op.results)
        self.check_region_results(results)
        if op.operation.name == "affine.for":
            lower = _affine_bound(op.attributes["lowerBoundMap"], max)
            upper = _affine_bound(op.attributes["upperBoundMap"], min)
            step = ir.IntegerAttr(op.attributes["step"]).value
            inputs = list(op.operands)
        else:
            lower, upper, step = map(_constant_bound, list(op.operands)[:3])
            inputs = list(op.operands)[3:]
        if step <= 0:
            raise ValueError("expected a positive loop step")
        remaining = max(0, upper - lower)
        count = (remaining + step - 1) // step
        if count >= 2**32:
            raise ValueError("loop trip count is outside the supported u32 range")
        if len(inputs) != len(results):
            raise ValueError("unexpected loop operand segments")
        if count == 0:
            self.values.update(
                (val, self.value(arg)) for val, arg in zip(results, inputs)
            )
            return

        body = op.regions[0].blocks[0]
        stores = set()
        for nested in _walk(body.operations):
            if (
                nested.operation.name in _STORE
                or nested.operation.name == "linalg.fill"
            ):
                stores.add(nested.operands[1])
        buffers = [val for val in self.buffers if val in stores]
        # Every loop-carried buffer needs its value from before the first iteration.
        initial = [self.value(val) for val in inputs + buffers]
        inner = [self.fresh() for _ in initial]
        iteration = self.fresh()
        ivar = self.fresh()
        bindings = [(body.arguments[0], ivar)]
        bindings.extend(zip(list(body.arguments)[1:], inner[: len(inputs)]))
        body_lines, yielded, final = self.capture(
            op.regions[0], bindings, zip(buffers, inner[len(inputs) :])
        )
        if not initial:
            return
        packed = self.fresh()
        self.line(
            f"let {packed} = for ({iteration}, {_tuple(inner)}) in u32:0..u32:{count} {{"
        )
        self.line(f"  let {ivar} = s64:{lower} + ({iteration} as s64) * s64:{step};")
        self.lines.extend(body_lines)
        self.line("  " + _tuple(yielded + [final[val] for val in buffers]))
        self.line(f"}}({_tuple(initial)});")
        self.rebind(results, buffers, packed)

    def lower_call(self, op):
        callee = ir.FlatSymbolRefAttr(op.attributes["callee"]).value
        if callee not in self.symbols:
            raise ValueError(f"missing callee @{callee}")
        results = list(op.results)
        self.check_region_results(results)
        args = ", ".join(self.value(arg) for arg in op.operands)
        packed = self.assign(f"{self.symbols[callee]}({args})")
        self.rebind(results, [], packed)


def _function_table(ops):
    return {op.name.value: op for op in ops if isinstance(op, func_d.FuncOp)}


def _lower_functions(functions, top_name, emitted_top_name=None):
    if top_name not in functions:
        raise DslxFunctionError(f"DSLX function @{top_name}: MLIR top symbol not found")
    symbols = {name: function_name(name) for name in functions}
    if emitted_top_name is not None:
        symbols[top_name] = function_name(emitted_top_name)
        if any(
            name != top_name and symbol == symbols[top_name]
            for name, symbol in symbols.items()
        ):
            raise DslxFunctionError(
                f"DSLX function @{top_name}: emitted top symbol collides with a helper"
            )
    definitions = []
    active = set()
    emitted = set()

    def visit(name):
        if name in emitted:
            return
        active.add(name)
        lowerer = MlirToDslxLowerer(
            functions[name], symbols=symbols, emitted_name=symbols[name]
        )
        source = lowerer._emit_function()
        for op in _walk(functions[name].body.blocks[0].operations):
            if op.operation.name != "func.call":
                continue
            callee = ir.FlatSymbolRefAttr(op.attributes["callee"]).value
            if callee in active:
                raise lowerer.error(op, f"recursive call to @{callee} is unsupported")
            if callee not in symbols:
                raise lowerer.error(op, f"missing callee @{callee}")
            visit(callee)
        active.remove(name)
        emitted.add(name)
        definitions.append(source)

    visit(top_name)
    functions_source = "\n\n".join(definitions) + "\n"
    if "__allo_fp32_" in functions_source:
        return FP32_DSLX + "\n" + functions_source
    return functions_source


def lower_functions(module, top_name, *, emitted_top_name=None):
    """Emit the selected function and its transitively called pure helpers."""
    return _lower_functions(
        _function_table(module.body.operations), top_name, emitted_top_name
    )
