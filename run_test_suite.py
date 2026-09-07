#!/usr/bin/env python3
"""Minimal zero-dependency collector for this repo's test modules.

The box has no pytest, and installing one into a shared production venv while
a run is live is not worth the risk. This runs both styles used here: bare
`test_*` functions and unittest.TestCase classes.
"""
import glob
import importlib
import inspect
import sys
import traceback
import unittest

only = sys.argv[1:]
modules = sorted(only or glob.glob("test_*.py"))

passed = failed = errors_importing = skipped = 0
subtests = 0
failures = []

for path in modules:
    name = path[:-3] if path.endswith(".py") else path
    try:
        module = importlib.import_module(name)
    except Exception:
        errors_importing += 1
        failures.append((name, "<import>", traceback.format_exc()))
        print(f"ERROR import {name}")
        continue

    # unittest.TestCase classes
    suite = unittest.TestLoader().loadTestsFromModule(module)
    if suite.countTestCases():
        result = unittest.TextTestRunner(verbosity=0, stream=open("/dev/null", "w")).run(suite)
        subtests += result.testsRun
        passed += result.testsRun - len(result.failures) - len(result.errors)
        for case, tb in list(result.failures) + list(result.errors):
            failed += 1
            failures.append((name, str(case), tb))

    # bare pytest-style functions
    for attr in sorted(vars(module)):
        if not attr.startswith("test_"):
            continue
        func = getattr(module, attr)
        if not callable(func):
            continue
        if isinstance(func, type):
            continue
        try:
            needs = [
                p for p in inspect.signature(func).parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)
            ]
        except (TypeError, ValueError):
            needs = []
        if needs:
            skipped += 1          # pytest fixture (tmp_path/monkeypatch)
            continue
        try:
            func()
            passed += 1
        except Exception:
            failed += 1
            failures.append((name, attr, traceback.format_exc()))

print()
for name, case, tb in failures:
    print("=" * 70)
    print(f"FAIL {name}::{case}")
    print(tb)
print("=" * 70)
print(f"passed={passed} failed={failed} import_errors={errors_importing} "
      f"skipped_fixture={skipped} unittest_cases={subtests} modules={len(modules)}")
sys.exit(1 if (failed or errors_importing) else 0)
