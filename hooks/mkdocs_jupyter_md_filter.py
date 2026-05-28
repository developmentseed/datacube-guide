"""Prevent the ``mkdocs-jupyter`` plugin from probing ``.md`` files.

The plugin's ``should_include`` runs ``jupytext.read`` on every ``.md`` file
to look for a Python kernel spec. ``jupytext`` detects any markdown file
containing a line that starts with ``:::`` as pandoc-format and shells out to
pandoc — and pandoc then emits "unclosed Div" warnings for the
``mkdocstrings`` autodoc directives on our API reference pages.

We have no markdown-format notebooks, so stripping ``.md`` from the plugin's
supported extension list short-circuits the probe before jupytext is ever
called.
"""

from __future__ import annotations


def on_startup(*, command, dirty):  # noqa: ARG001 - mkdocs hook signature
    try:
        from mkdocs_jupyter.plugin import Plugin
    except ImportError:  # pragma: no cover - plugin not installed
        return
    Plugin._supported_extensions = [
        ext for ext in Plugin._supported_extensions if ext != ".md"
    ]
