"""Check that the standalone runtime stays complete and within its release scope."""

import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "retro_guided_generation"


def delivered_modules():
    modules = {}
    for path in (ROOT / "src").rglob("*.py"):
        parts = list(path.relative_to(ROOT / "src").with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        modules[".".join(parts)] = path
    return modules


class ReleaseScopeTest(unittest.TestCase):
    def test_source_inventory_excludes_unreleased_workflows(self):
        modules = delivered_modules()
        manifest = json.loads((ROOT / "docs/verification/source_manifest.json").read_text())
        self.assertEqual(sorted(modules), manifest["modules"])
        self.assertEqual(len(modules), manifest["module_count"])
        for name in modules:
            with self.subTest(module=name):
                self.assertNotIn("stage_c", name)
                self.assertNotIn("llm", name)
                self.assertNotIn(name, {
                    f"{PREFIX}.guidance.builder",
                    f"{PREFIX}.guidance.structure",
                    f"{PREFIX}.guidance.panel_structure_review",
                    f"{PREFIX}.data.mosyn",
                    f"{PREFIX}.data.mosyn_gap",
                })

    def test_entrypoints_and_deferred_imports_resolve_from_release(self):
        modules = delivered_modules()
        imports = set()
        sources = {**modules, **{
            path.stem: path for path in (
                ROOT / "reproduce.py", ROOT / "stage_b_flow.py",
                ROOT / "score_results.py",
            )
        }}
        for owner, path in sources.items():
            package = owner if path.name == "__init__.py" else owner.rpartition(".")[0]
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    target = node.module or ""
                    if node.level:
                        target = importlib.util.resolve_name("." * node.level + target, package)
                    if target.startswith(PREFIX):
                        imports.update((target, alias.name) for alias in node.names)
                elif isinstance(node, ast.Import):
                    imports.update((alias.name, "") for alias in node.names
                                   if alias.name.startswith(PREFIX))

        # A fresh interpreter cannot satisfy missing release imports from another
        # test's sys.modules. No model is constructed or executed here.
        code = """
import importlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(root / 'src'), str(root)]
modules, imports = json.loads(sys.argv[2])
for name in modules + ['reproduce', 'stage_b_flow', 'score_results']:
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(root), name
for target, symbol in imports:
    module = importlib.import_module(target)
    if symbol and not hasattr(module, symbol):
        importlib.import_module(target + '.' + symbol)
"""
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
                           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", code, str(ROOT),
             json.dumps([sorted(modules), sorted(imports)])],
            env=environment, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
