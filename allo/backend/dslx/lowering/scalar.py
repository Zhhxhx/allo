# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scalar operations shared by the pure-function and stateful DSLX lowerers."""

import struct

from allo._mlir import ir


_INTEGER_BINARY = {
    "addi": "+",
    "subi": "-",
    "muli": "*",
    "andi": "&",
    "ori": "|",
    "xori": "^",
    "divui": "/",
    "remui": "%",
}
_SIGNED_BINARY = {"divsi": "/", "remsi": "%"}
_SHIFT = {"shli": "<<", "shrui": ">>", "shrsi": ">>"}
_INTEGER_PREDICATES = ("==", "!=", "<", "<=", ">", ">=", "<", "<=", ">", ">=")


def _bits(width, signed=False):
    prefix = "s" if signed else "u"
    return f"{prefix}{width}" if width <= 64 else f"{prefix}N[{width}]"


def _integer_width(typ):
    if ir.IndexType.isinstance(typ):
        return 64
    return ir.IntegerType(typ).width


def scalar_type(typ):
    """Represent floats and integers by their raw bits, and index by signed i64."""
    if ir.IndexType.isinstance(typ):
        return "s64"
    if ir.IntegerType.isinstance(typ):
        return _bits(_integer_width(typ))
    if ir.F32Type.isinstance(typ):
        return "u32"
    raise ValueError(f"unsupported scalar type {typ}")


def _cast(expr, typ):
    return f"({expr} as {typ})"


def _int_cast(expr, typ, signed=False):
    return _cast(expr, _bits(_integer_width(typ), signed))


def _constant(op):
    typ = op.results[0].type
    attr = op.attributes["value"]
    name = scalar_type(typ)
    if ir.F32Type.isinstance(typ):
        literal = str(attr).split()[0]
        if literal.startswith("0x"):
            return f"u32:{literal}"
        bits = struct.unpack(">I", struct.pack(">f", ir.FloatAttr(attr).value))[0]
        return f"u32:0x{bits:08x}"
    if ir.IndexType.isinstance(typ):
        return f"s64:{ir.IntegerAttr(attr).value}"
    number = (
        int(str(attr).split()[0], 0)
        if not ir.BoolAttr.isinstance(attr)
        else int(attr.value)
    )
    number &= (1 << _integer_width(typ)) - 1
    return f"{name}:0x{number:x}"


def _float_cmp(op, lhs, rhs):
    predicate = ir.IntegerAttr(op.attributes["predicate"]).value
    equal = f"__allo_fp32_equal({lhs}, {rhs})"
    less = f"__allo_fp32_less({lhs}, {rhs})"
    greater = f"__allo_fp32_less({rhs}, {lhs})"
    unordered = f"(__allo_fp32_nan({lhs}) || __allo_fp32_nan({rhs}))"
    comparisons = (
        equal,
        greater,
        f"({greater} || {equal})",
        less,
        f"({less} || {equal})",
        f"!{equal}",
    )
    if predicate == 0:
        return "false"
    if predicate == 15:
        return "true"
    if predicate == 7:
        return f"!{unordered}"
    if predicate == 8:
        return unordered
    if predicate < 7:
        return f"(!{unordered} && {comparisons[predicate - 1]})"
    return f"({unordered} || {comparisons[predicate - 9]})"


def scalar_expression(op, value):
    """Lower a scalar operation using value(MLIR Value) to resolve its inputs.

    Return None when the operation is outside the scalar lowering's scope.
    """
    name = op.operation.name
    if not name.startswith("arith."):
        return None
    kind = name.removeprefix("arith.")
    if kind == "constant":
        return _constant(op)
    if kind == "select":
        if ir.MemRefType.isinstance(op.results[0].type):
            return None
        scalar_type(op.results[0].type)
        cond, lhs, rhs = (value(operand) for operand in op.operands)
        return f"if {cond} {{ {lhs} }} else {{ {rhs} }}"
    if kind in {"addf", "subf", "mulf", "negf", "cmpf"}:
        scalar_type(op.operands[0].type)
        lhs = value(op.operands[0])
        if kind == "negf":
            return f"({lhs} ^ u32:0x80000000)"
        rhs = value(op.operands[1])
        if kind == "cmpf":
            return _float_cmp(op, lhs, rhs)
        if kind == "subf":
            rhs = f"({rhs} ^ u32:0x80000000)"
        helper = "mul" if kind == "mulf" else "add"
        return f"__allo_fp32_{helper}({lhs}, {rhs})"
    if kind == "bitcast":
        source = value(op.operands[0])
        scalar_type(op.operands[0].type)
        return _cast(source, scalar_type(op.results[0].type))
    if kind in {"extsi", "extui", "trunci", "index_cast", "index_castui"}:
        source = op.operands[0]
        target = op.results[0].type
        expr = value(source)
        if kind in {"extsi", "index_cast"}:
            expr = _int_cast(
                _int_cast(expr, source.type, signed=True), target, signed=True
            )
        elif kind in {"extui", "index_castui"}:
            expr = _int_cast(expr, source.type)
        return _cast(expr, scalar_type(target))

    if (
        kind not in _INTEGER_BINARY
        and kind not in _SIGNED_BINARY
        and kind not in _SHIFT
    ):
        if kind not in {
            "cmpi",
            "floordivsi",
            "ceildivsi",
            "ceildivui",
            "minsi",
            "minui",
            "maxsi",
            "maxui",
        }:
            return None

    lhs, rhs = (value(operand) for operand in op.operands)
    operand_type = op.operands[0].type
    result_type = scalar_type(op.results[0].type)
    if kind == "cmpi":
        predicate = ir.IntegerAttr(op.attributes["predicate"]).value
        if predicate >= 2:
            signed = predicate < 6
            lhs = _int_cast(lhs, operand_type, signed)
            rhs = _int_cast(rhs, operand_type, signed)
        return f"({lhs} {_INTEGER_PREDICATES[predicate]} {rhs})"
    if kind in _SHIFT:
        lhs = _int_cast(lhs, operand_type, signed=kind == "shrsi")
        rhs = _int_cast(rhs, op.operands[1].type)
        return _cast(f"({lhs} {_SHIFT[kind]} {rhs})", result_type)
    if kind in _INTEGER_BINARY or kind in _SIGNED_BINARY:
        signed = kind in _SIGNED_BINARY
        lhs = _int_cast(lhs, operand_type, signed)
        rhs = _int_cast(rhs, operand_type, signed)
        operator = (_SIGNED_BINARY if signed else _INTEGER_BINARY)[kind]
        return _cast(f"({lhs} {operator} {rhs})", result_type)

    signed = kind.endswith("si")
    lhs = _int_cast(lhs, operand_type, signed)
    rhs = _int_cast(rhs, operand_type, signed)
    if kind.startswith(("min", "max")):
        operator = "<" if kind.startswith("min") else ">"
        return _cast(
            f"(if {lhs} {operator} {rhs} {{ {lhs} }} else {{ {rhs} }})", result_type
        )
    operand_dslx_type = _bits(_integer_width(operand_type), signed)
    quotient = f"({lhs} / {rhs})"
    has_remainder = f"({lhs} % {rhs} != {operand_dslx_type}:0)"
    if signed:
        operator = "-" if kind == "floordivsi" else "+"
        same_sign = "!=" if kind == "floordivsi" else "=="
        signs = f"(({lhs} < {operand_dslx_type}:0) {same_sign} ({rhs} < {operand_dslx_type}:0))"
        adjust = f"({has_remainder} && {signs})"
    else:
        operator, adjust = "+", has_remainder
    return _cast(
        f"({quotient} {operator} {_cast(adjust, operand_dslx_type)})", result_type
    )
