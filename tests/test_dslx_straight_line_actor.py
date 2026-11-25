# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from allo._mlir import ir
from allo.backend.dslx.lowering.core_actor import ActorPort
from allo.backend.dslx.lowering.straight_line_actor import StraightLineActorLowerer
from allo.utils import register_dialect


_CLOCK = """
module {
  memref.global "private" @counter : memref<i64> = dense<19>
  func.func @tick(%increment: i64) -> i64 {
    %state = memref.get_global @counter : memref<i64>
    %old = memref.load %state[] : memref<i64>
    %next = arith.addi %old, %increment : i64
    memref.store %next, %state[] : memref<i64>
    return %old : i64
  }
  func.func @sampling(%output: memref<1xi64>) {
    %zero = arith.constant 0 : index
    %inc = arith.constant 3 : i64
    %sample = func.call @tick(%inc) : (i64) -> i64
    memref.store %sample, %output[%zero] : memref<1xi64>
    return
  }
}
"""


@pytest.fixture
def lower():
    with ir.Context() as ctx:
        register_dialect(ctx)

        def create(source):
            module = ir.Module.parse(source)
            module.operation.verify()
            actor = next(
                op
                for op in module.body.operations
                if op.operation.name == "func.func" and op.name.value == "sampling"
            )
            return StraightLineActorLowerer(
                module, actor, "sampling", {0: ActorPort(("sample",))}
            )

        yield create


def test_straight_line_sampling_uses_old_persistent_value_as_output(lower):
    actor = lower(_CLOCK)
    source = actor.lower()
    assert "proc sampling {" in source
    assert "s0: u64:19," in source
    assert "send(join(), sample, state.s0)" in source
    assert "state.s0 as u64) +" in source
    assert "u64:0x3" in source
    assert actor.directions == {0: "out"}
    assert list(actor.global_memories) == ["counter"]
    assert actor.lower() == source


@pytest.mark.parametrize(
    ("addition", "message"),
    [
        (
            "memref.store %inc, %output[%zero] : memref<1xi64>",
            "written more than once per tick",
        ),
        (
            "%read = memref.load %output[%zero] : memref<1xi64>\n"
            "memref.store %read, %output[%zero] : memref<1xi64>",
            "write-only outputs",
        ),
    ],
)
def test_rejects_non_sampling_port_effects(lower, addition, message):
    actor = lower(_CLOCK.replace("    return\n", f"    {addition}\n    return\n"))
    with pytest.raises(NotImplementedError, match=message):
        actor.lower()
