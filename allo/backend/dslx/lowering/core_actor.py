"""Lower scheduled Allo dataflow actors to blocking XLS procs."""

from collections import defaultdict
from dataclasses import dataclass, field
from math import prod

from allo._mlir import ir
from allo._mlir.passmanager import PassManager

from .mlir_to_dslx_fn import dslx_type, scalar_expression


def prepare_actor_module(module):
    """Normalize an independent copy to MLIR's scalar control-flow dialect."""
    with module.operation.context:
        result = ir.Module.parse(str(module))
        passes = PassManager.parse(
            "builtin.module(convert-linalg-to-loops,lower-affine,"
            "convert-scf-to-cf,canonicalize,mem2reg)"
        )
        passes.run(result.operation)
        return result


@dataclass(frozen=True)
class ActorPort:
    names: tuple[str, ...]


@dataclass(frozen=True)
class ActorRAM:
    name: str
    width: int
    size: int
    initial: int | tuple[int, ...]

    @property
    def address_width(self):
        return max(1, (self.size - 1).bit_length())

    @property
    def request(self):
        return f"{self.name}_req"

    @property
    def response(self):
        return f"{self.name}_resp"


@dataclass(eq=False)
class StateField:
    name: str
    type: str
    initial: str
    persistent: bool = False


@dataclass(eq=False)
class Storage:
    shape: tuple[int, ...]
    element_type: str
    state: StateField | None = None
    port: int | None = None
    ram: ActorRAM | None = None


@dataclass(eq=False)
class Step:
    number: int
    lines: list[str] = field(default_factory=list)
    values: dict[StateField, str] = field(default_factory=dict)
    target: str = ""
    channel_access: bool = False

    @property
    def active(self):
        return f"active_{self.number}"


@dataclass(eq=False)
class FunctionInstance:
    function: ir.OpView
    entry: Step
    return_pc: StateField
    returned: tuple[StateField, ...]
    blocks: dict[ir.Block, Step]
    values: dict[ir.Value, StateField] = field(default_factory=dict)
    memories: dict[ir.Value, Storage] = field(default_factory=dict)


def fail(op, message):
    raise NotImplementedError(f"{op.operation.name} at {op.location}: {message}")


def width(ty):
    if ir.IntegerType.isinstance(ty):
        return ir.IntegerType(ty).width
    if ir.F32Type.isinstance(ty):
        return 32
    raise NotImplementedError(f"Unsupported actor memory element type {ty}")


def literal(typ, value):
    return f"({typ}:{value})" if value < 0 else f"{typ}:{value}"


def integer(attr):
    # IntegerAttr.value is limited to 64 bits in some MLIR builds.
    spelling = str(attr).split(" : ", 1)[0]
    if spelling in ("true", "false"):
        return int(spelling == "true")
    return int(spelling, 0)


def constant(value):
    owner = value.owner
    if isinstance(owner, ir.Block) or owner.operation.name != "arith.constant":
        return None
    attr = owner.attributes["value"]
    return integer(attr) if ir.IntegerAttr.isinstance(attr) else None


def tuple_expr(values):
    return "(" + "".join(f"{value}, " for value in values) + ")"


def bit_expression(op, value):
    name = op.operation.name
    if name not in ("allo.get_bit", "allo.get_slice", "allo.set_bit", "allo.set_slice"):
        return None
    operands = list(op.operands)
    data = value(operands[0])
    low_operand = operands[2] if name.endswith("slice") else operands[1]
    low = f"({value(low_operand)} as u64)"
    result_type = dslx_type(op.results[0].type)
    if name.endswith("slice"):
        high = f"({value(operands[1])} as u64)"
        count = f"({high} - {low} + u64:1)"
    else:
        count = "u64:1"
    mask = f"(({result_type}:1 << {count}) - {result_type}:1)"
    if name.startswith("allo.get"):
        return f"((({data} >> {low}) as {result_type}) & {mask})"
    data_value = value(operands[-1])
    return f"(({data} & !({mask} << {low})) | ((({data_value} as {result_type}) & {mask}) << {low}))"


class ActorLowerer:
    """Compile an isolated actor, specializing calls by their memref bindings.

    A step runs a straight-line sequence and transfers control at a branch,
    call, or blocking channel access. Only values used by a later step enter
    proc state. Each function specialization owns its own return address.
    """

    def __init__(self, module, function, symbol, ports):
        self.module = module
        self.function = function
        self.symbol = symbol
        self._pc_type = f"{symbol}Pc"
        self.ports = ports
        self.directions = {}
        self.rams = []
        self.global_memories = {}
        self.uses_float = False
        self._functions = {}
        self._globals = {}
        for op in module.body.operations:
            name = op.operation.name
            if name == "func.func":
                self._functions[op.attributes["sym_name"].value] = op
            elif name == "memref.global":
                self._globals[op.attributes["sym_name"].value] = op

        self._locals = 0
        self._fields = []
        self._steps = []
        self._instances = {}
        self._receives = defaultdict(list)
        self._sends = defaultdict(list)

    def _new_field(self, typ, initial=None):
        name = f"v{len(self._fields)}"
        value = StateField(name, typ, initial or f"zero!<{typ}>()")
        self._fields.append(value)
        return value

    def _new_step(self):
        step = Step(len(self._steps))
        self._steps.append(step)
        return step

    def _temporary(self, step, expression, typ=None):
        name = f"t{self._locals}"
        self._locals += 1
        annotation = f": {typ}" if typ else ""
        step.lines.append(f"let {name}{annotation} = {expression};")
        return name

    def _reg(self, instance, value):
        if value not in instance.values:
            instance.values[value] = self._new_field(dslx_type(value.type))
        return instance.values[value]

    def _read(self, step, reg):
        if reg in step.values:
            return step.values[reg]
        reg.persistent = True
        return f"state.{reg.name}"

    def _write(self, step, reg, expression):
        step.values[reg] = self._temporary(step, expression, reg.type)

    def _value(self, instance, step, value):
        frozen = constant(value)
        if frozen is not None:
            typ = dslx_type(value.type)
            if ir.IntegerType.isinstance(value.type):
                frozen %= 1 << ir.IntegerType(value.type).width
            return literal(typ, frozen)
        return self._read(step, self._reg(instance, value))

    def _storage(self, typ, initial=0, *, port=None):
        ref = ir.MemRefType(typ)
        shape = tuple(ref.shape)
        if any(dim <= 0 for dim in shape):
            raise NotImplementedError(f"Actor storage requires positive static dimensions: {typ}")
        element_width = width(ref.element_type)
        element_type = dslx_type(ref.element_type)
        storage = Storage(shape, element_type, port=port)
        size = prod(shape)
        if port is not None:
            if len(self.ports[port].names) != size:
                raise ValueError(f"Argument {port} of {self.symbol} needs {size} channel lanes")
            return storage
        space = ref.memory_space
        if space is not None and ir.StringAttr.isinstance(space) and space.value.startswith("stream:"):
            raise NotImplementedError("Actor-local stream allocation needs a dataflow producer and consumer")
        if size >= 256 or space is not None:
            ram = ActorRAM(f"{self.symbol}_ram{len(self.rams)}", element_width, size, initial)
            self.rams.append(ram)
            storage.ram = ram
            return storage
        if size == 1:
            val = initial if isinstance(initial, int) else initial[0]
            storage.state = self._new_field(element_type, literal(element_type, val))
        else:
            array_type = f"{element_type}[{size}]"
            if isinstance(initial, int):
                val = f"[{literal(element_type, initial)}, ...]"
            else:
                val = "[" + ", ".join(literal(element_type, n) for n in initial) + "]"
            storage.state = self._new_field(array_type, f"{array_type}:{val}")
        return storage

    def _global(self, op):
        name = op.attributes["name"].value
        if name in self.global_memories:
            return self.global_memories[name]
        decl = self._globals.get(name)
        if decl is None or "initial_value" not in decl.attributes:
            fail(op, f"Global '{name}' needs a local dense initializer")
        attr = decl.attributes["initial_value"]
        if not ir.DenseIntElementsAttr.isinstance(attr):
            fail(decl, "Only dense integer global initializers are supported in actors")
        dense = ir.DenseIntElementsAttr(attr)
        typ = op.results[0].type
        modulus = 1 << width(ir.MemRefType(typ).element_type)
        if dense.is_splat:
            initial = integer(dense.get_splat_value()) % modulus
        else:
            initial = tuple(int(x) % modulus for x in dense)
        storage = self._storage(typ, initial)
        self.global_memories[name] = storage
        return storage

    def _new_instance(self, func, memories):
        key = (func.attributes["sym_name"].value, memories)
        if key in self._instances:
            return self._instances[key], False
        blocks = {block: self._new_step() for block in func.body.blocks}
        if not blocks:
            fail(func, "External declaration cannot be called from an actor")
        return_pc = self._new_field(self._pc_type)
        returned = []
        for typ in func.type.results:
            if ir.MemRefType.isinstance(typ):
                fail(func, "Returning memrefs from actor helpers is unsupported")
            returned.append(self._new_field(dslx_type(typ)))
        instance = FunctionInstance(func, next(iter(blocks.values())), return_pc, tuple(returned), blocks)
        self._instances[key] = instance

        for arg, storage in zip(func.arguments, memories, strict=True):
            if storage is not None:
                instance.memories[arg] = storage
        for block in blocks:
            for op in block.operations:
                name = op.operation.name
                if name in ("memref.alloc", "memref.alloca"):
                    if len(op.operands):
                        fail(op, "Actor allocation requires a static identity binding")
                    instance.memories[op.results[0]] = self._storage(op.results[0].type)
                elif name == "memref.get_global":
                    instance.memories[op.results[0]] = self._global(op)
        return instance, True

    def _address(self, instance, step, storage, indices, bits):
        if len(indices) != len(storage.shape):
            raise ValueError(f"Expected {len(storage.shape)} memref indices")
        fixed = 0
        terms = []
        typ = f"uN[{bits}]"
        for i, idx in enumerate(indices):
            value = constant(idx)
            if value is not None and not 0 <= value < storage.shape[i]:
                raise ValueError(f"Static memref index {value} is out of bounds for {storage.shape}")
            if storage.shape[i] == 1:
                continue
            stride = prod(storage.shape[i + 1 :])
            if value is None:
                index = f"({self._value(instance, step, idx)} as {typ})"
                terms.append(f"({index} * {typ}:{stride})" if stride != 1 else index)
            else:
                fixed += value * stride
        if not terms:
            return literal(typ, fixed), fixed
        if fixed:
            terms.append(literal(typ, fixed % (1 << bits)))
        return "(" + " + ".join(terms) + ")", None

    def _port_lanes(self, instance, step, storage, indices, direction, op):
        arg = storage.port
        prior = self.directions.setdefault(arg, direction)
        if prior != direction:
            fail(op, f"Argument {arg} is both a producer and consumer")
        names = self.ports[arg].names
        bits = max(1, (len(names) - 1).bit_length())
        addr, fixed = self._address(instance, step, storage, indices, bits)
        if fixed is not None:
            return [(names[fixed], step.active)]
        return [
            (name, f"({step.active} && ({addr} == uN[{bits}]:{i}))")
            for i, name in enumerate(names)
        ]

    def _receive(self, step, sources, ty):
        step.channel_access = True
        values = []
        for name, pred in sources:
            self._receives[name].append((pred, ty))
            values.append(f"(if {pred} {{ recv_{name} }} else {{ {ty}:0 }})")
        return "(" + " | ".join(values) + ")"

    def _send(self, step, destinations, ty, value):
        step.channel_access = True
        for name, pred in destinations:
            self._sends[name].append((pred, ty, value))

    def _access(self, instance, step, op):
        load = op.operation.name == "memref.load"
        operands = list(op.operands)
        memref = operands[0 if load else 1]
        indices = operands[1 if load else 2 :]
        if memref not in instance.memories:
            fail(op, "Unknown or dynamically aliased memref")
        storage = instance.memories[memref]
        ty = storage.element_type
        if storage.port is not None:
            lanes = self._port_lanes(instance, step, storage, indices, "in" if load else "out", op)
            if load:
                self._write(step, self._reg(instance, op.results[0]), self._receive(step, lanes, ty))
            else:
                self._send(step, lanes, ty, self._value(instance, step, operands[0]))
        elif storage.ram:
            ram = storage.ram
            addr, _ = self._address(instance, step, storage, indices, ram.address_width)
            request = f"u1:{int(not load)} ++ {addr} ++ "
            request += f"{ty}:0" if load else self._value(instance, step, operands[0])
            request_ty = f"uN[{1 + ram.address_width + ram.width}]"
            self._send(step, [(ram.request, step.active)], request_ty, request)
            if load:
                response = self._new_step()
                step.target = self._pc(response)
                data = self._receive(response, [(ram.response, response.active)], ty)
                self._write(response, self._reg(instance, op.results[0]), data)
                return response
        else:
            state = storage.state
            size = prod(storage.shape)
            addr, _ = self._address(instance, step, storage, indices, max(1, (size - 1).bit_length()))
            if load:
                data = self._read(step, state)
                if size != 1:
                    data += f"[{addr}]"
                self._write(step, self._reg(instance, op.results[0]), data)
            else:
                value = self._value(instance, step, operands[0])
                if size != 1:
                    value = f"update({self._read(step, state)}, {addr}, {value})"
                self._write(step, state, value)
        return step

    def _pc(self, step):
        return f"{self._pc_type}:{step.number}"

    def _assignments(self, instance, step, op, successor, operands):
        assignments = {}
        for dest, source in zip(successor.arguments, operands, strict=True):
            if ir.MemRefType.isinstance(dest.type):
                fail(op, "Control-flow merges of memref bindings are unsupported")
            assignments[self._reg(instance, dest)] = self._value(instance, step, source)
        return assignments

    def _edge_arguments(self, instance, step, op, cond=None):
        successors = list(op.successors)
        operands = list(op.operands)
        if cond is None:
            assignments = self._assignments(instance, step, op, successors[0], operands)
            for reg, val in assignments.items():
                self._write(step, reg, val)
            step.target = self._pc(instance.blocks[successors[0]])
            return

        true_size = len(successors[0].arguments)
        yes = self._assignments(instance, step, op, successors[0], operands[1 : 1 + true_size])
        no = self._assignments(instance, step, op, successors[1], operands[1 + true_size :])
        for reg in dict.fromkeys([*yes, *no]):
            true_val = yes[reg] if reg in yes else self._read(step, reg)
            false_val = no[reg] if reg in no else self._read(step, reg)
            self._write(step, reg, f"if {cond} {{ {true_val} }} else {{ {false_val} }}")
        step.target = f"if {cond} {{ {self._pc(instance.blocks[successors[0]])} }} else {{ {self._pc(instance.blocks[successors[1]])} }}"

    def _call(self, instance, step, op):
        symbol = op.attributes["callee"].value
        if symbol not in self._functions:
            fail(op, f"Undefined function '{symbol}'")
        func = self._functions[symbol]
        args = list(op.operands)
        bindings = []
        for value in args:
            if ir.MemRefType.isinstance(value.type):
                if value not in instance.memories:
                    fail(op, "Unknown memref argument binding")
                bindings.append(instance.memories[value])
            else:
                bindings.append(None)
        callee, fresh = self._new_instance(func, tuple(bindings))
        if not fresh and callee in self._compiling:
            fail(op, "Recursive calls cannot maintain distinct actor memory frames")
        continuation = self._new_step()
        self._write(step, callee.return_pc, self._pc(continuation))
        for arg, actual, storage in zip(func.arguments, args, bindings, strict=True):
            if storage is None:
                self._write(step, self._reg(callee, arg), self._value(instance, step, actual))
        step.target = self._pc(callee.entry)
        if fresh:
            self._compile(callee)
        for actual, result in zip(op.results, callee.returned, strict=True):
            self._write(continuation, self._reg(instance, actual), self._read(continuation, result))
        return continuation

    def _compile(self, instance, root=False):
        self._compiling.append(instance)
        for block, step in instance.blocks.items():
            operations = list(block.operations)
            for op in operations:
                name = op.operation.name
                if name in ("memref.alloc", "memref.alloca", "memref.get_global", "memref.dealloc"):
                    continue
                if name in ("memref.load", "memref.store"):
                    memref = op.operands[0 if name == "memref.load" else 1]
                    storage = instance.memories.get(memref)
                    if storage and (storage.port is not None or storage.ram):
                        # Compute the channel address from established state before issuing the access.
                        if step.lines or step.channel_access:
                            access = self._new_step()
                            step.target = self._pc(access)
                            step = access
                    step = self._access(instance, step, op)
                elif name == "func.call":
                    step = self._call(instance, step, op)
                elif name == "func.return":
                    if root:
                        if len(op.operands):
                            fail(op, "A dataflow actor returns data through its channel arguments")
                        step.target = self._pc(instance.entry)
                    else:
                        for reg, value in zip(instance.returned, op.operands, strict=True):
                            self._write(step, reg, self._value(instance, step, value))
                        step.target = self._read(step, instance.return_pc)
                elif name in ("cf.br", "cf.cond_br"):
                    cond = self._value(instance, step, op.operands[0]) if name == "cf.cond_br" else None
                    self._edge_arguments(instance, step, op, cond)
                else:
                    if len(op.results) != 1:
                        fail(op, "Expected a supported scalar operation")
                    if name == "arith.constant" and constant(op.results[0]) is not None:
                        continue

                    def value(operand):
                        return self._value(instance, step, operand)

                    expression = bit_expression(op, value)
                    if expression is None:
                        expression = scalar_expression(op, value)
                    if expression is None:
                        fail(op, "Unsupported scalar actor operation")
                    self.uses_float |= name in ("arith.addf", "arith.mulf")
                    self._write(step, self._reg(instance, op.results[0]), expression)

            if not step.target:
                fail(operations[-1], "Unterminated actor control-flow block")
        self._compiling.pop()

    def _render(self, entry):
        typ = f"{self.symbol}State"
        pc = self._pc_type
        state = [reg for reg in self._fields if reg.persistent]
        pc_bits = f"uN[{max(1, (len(self._steps) - 1).bit_length())}]"
        lines = [f"type {pc} = {pc_bits};",
                 f"struct {typ} {{", f"  pc: {pc},"]
        lines.extend(f"  {reg.name}: {reg.type}," for reg in state)
        lines += ["}", f"proc {self.symbol} {{"]

        ports = []
        for index, direction in sorted(self.directions.items()):
            arg = self.function.arguments[index]
            elem = ir.MemRefType(arg.type).element_type
            ports.extend(f"{name}: chan<{dslx_type(elem)}> {direction}" for name in self.ports[index].names)
        for ram in self.rams:
            ports.append(f"{ram.request}: chan<uN[{1 + ram.address_width + ram.width}]> out")
            ports.append(f"{ram.response}: chan<uN[{ram.width}]> in")
        port_names = [port.split(":", 1)[0] for port in ports]
        lines.extend(f"  {port};" for port in ports)
        lines += [
            "  config(" + ", ".join(ports) + ") {",
            f"    {tuple_expr(port_names)}",
            "  }",
            f"  init {{ {typ} {{",
            f"    pc: {self._pc(entry)},",
        ]
        lines.extend(f"    {reg.name}: {reg.initial}," for reg in state)
        lines += ["  } }", f"  next(state: {typ}) {{"]
        for step in self._steps:
            lines.append(f"    let {step.active} = state.pc == {self._pc(step)};")
        for name, receives in self._receives.items():
            predicate = " || ".join(pred for pred, _ in receives)
            ty = receives[0][1]
            lines.append(f"    let (_, recv_{name}) = recv_if(join(), {name}, {predicate}, {ty}:0);")
        for step in self._steps:
            lines.extend(f"    {line}" for line in step.lines)
        for name, sends in self._sends.items():
            predicate = " || ".join(pred for pred, _, _ in sends)
            data = " | ".join(f"(if {pred} {{ {value} }} else {{ {ty}:0 }})" for pred, ty, value in sends)
            lines.append(f"    let _ = send_if(join(), {name}, {predicate}, {data});")
        lines += [f"    {typ} {{", "      pc: match state.pc {"]
        for step in self._steps:
            lines.append(f"        {pc_bits}:{step.number} => {step.target},")
        lines += ["        _ => state.pc,", "      },"]
        writes = defaultdict(list)
        for step in self._steps:
            for reg, value in step.values.items():
                if reg.persistent:
                    writes[reg].append((step, value))
        for reg in state:
            lines.append(f"      {reg.name}: match state.pc {{")
            for step, value in writes[reg]:
                lines.append(f"        {pc_bits}:{step.number} => {value},")
            lines += [f"        _ => state.{reg.name},", "      },"]
        lines += ["    }", "  }", "}"]
        return "\n".join(lines) + "\n"

    def lower(self):
        args = []
        for index, arg in enumerate(self.function.arguments):
            if index not in self.ports or not ir.MemRefType.isinstance(arg.type):
                fail(self.function, "Dataflow actor arguments must bind to channel lanes")
            args.append(self._storage(arg.type, port=index))
        self._compiling = []
        root, _ = self._new_instance(self.function, tuple(args))
        self._compile(root, root=True)
        return self._render(root.entry)

