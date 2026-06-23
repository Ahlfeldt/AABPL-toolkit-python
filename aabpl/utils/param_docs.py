"""
Lightweight parameter-description accessor for public API functions.

Usage
-----
Decorate a function with ``@attach_params``:

    @attach_params
    def radius_search(pts, r, c, ...):
        \"\"\"
        Args:
        -------
        r (float):
            Search radius in metres.
        c (str or list):
            Column(s) to aggregate.
        \"\"\"

Then at runtime:

    aabpl.radius_search.params.r      # → "Search radius in metres."
    aabpl.radius_search.params        # → full listing of all params

Descriptions are parsed lazily from the existing docstring on first access —
no duplication, no extra maintenance.
"""
import re as _re


class _ParamDocs:
    """Lazy accessor for parameter descriptions parsed from a Google/Sphinx docstring."""

    def __init__(self, func):
        self._func = func
        self._cache = None

    def _build(self):
        doc = self._func.__doc__ or ''
        # Find the Args block: everything after "Args:" up to the next
        # section header (a word followed by optional whitespace and "---..."),
        # or end of docstring.
        args_m = _re.search(
            r'Args\s*:\s*\n\s*-{2,}(.*?)(?=\n\s*\w[\w\s]*:\s*\n\s*-{2,}|\Z)',
            doc, _re.DOTALL,
        )
        self._cache = {}
        if not args_m:
            return
        block = args_m.group(1)
        # Each entry: "    param_name (type):\n        description...\n"
        # The description is one or more lines indented more than the param line.
        for m in _re.finditer(
            r'^(\w+)\s*\([^)]*\)\s*:\s*\n((?:[ \t]+[^\n]*\n?)*)',
            block, _re.MULTILINE,
        ):
            name = m.group(1)
            raw = m.group(2)
            # strip the minimum indentation present across description lines
            lines = [l for l in raw.splitlines() if l.strip()]
            min_indent = min((len(l) - len(l.lstrip())) for l in lines) if lines else 0
            desc = _re.sub(r'^[ \t]{' + str(min_indent) + r'}', '', raw,
                           flags=_re.MULTILINE).strip()
            self._cache[name] = desc

    def __getattr__(self, name):
        if self._cache is None:
            self._build()
        try:
            return self._cache[name]
        except KeyError:
            raise AttributeError(
                f"No parameter '{name}' documented in "
                f"{self._func.__qualname__}. "
                f"Available: {sorted(self._cache)}"
            ) from None

    def __repr__(self):
        if self._cache is None:
            self._build()
        if not self._cache:
            return f"<no documented parameters for {self._func.__qualname__}>"
        lines = [f"Parameters of {self._func.__qualname__}:"]
        for k, v in self._cache.items():
            first_line = v.split('\n')[0]
            lines.append(f"  {k}: {first_line}")
        return '\n'.join(lines)

    def __dir__(self):
        if self._cache is None:
            self._build()
        return sorted(self._cache)


def attach_params(func):
    """Attach a ``.params`` accessor to *func* that exposes its docstring params."""
    func.params = _ParamDocs(func)
    return func
