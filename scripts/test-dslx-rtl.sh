#!/usr/bin/env bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -n "${XLS_BIN:-}" ]]; then
    export PATH="$XLS_BIN:$PATH"
fi
export ALLO_DSLX_RTL_REQUIRED=1
exec "${PYTHON:-python}" -m pytest tests/test_dslx*.py tests/test_xls.py "$@"

