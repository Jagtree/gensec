import os
from pathlib import Path

EXCLUDE = {".venv", "__pycache__", "node_modules", ".git", "results",
           "examples", "claude", "build", "dist", ".doctrees",
           "_sources", "_static", "multiversion", "htmlcov"}

def generate_tree(root_dir=".", output_file="tree_output.txt"):
    root = Path(root_dir)
    lines = []

    def _walk(path, prefix=""):
        items = sorted(
            [i for i in path.iterdir() if not (i.is_dir() and i.name in EXCLUDE)],
            key=lambda x: (not x.is_dir(), x.name.lower())
        )
        for i, item in enumerate(items):
            connector = "└── " if i == len(items) - 1 else "├── "
            icon = "📁" if item.is_dir() else "📄"
            lines.append(f"{prefix}{connector}{icon} {item.name}")
            if item.is_dir():
                extension = "    " if i == len(items) - 1 else "│   "
                _walk(item, prefix + extension)

    _walk(root)
    Path(output_file).write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    generate_tree()
    print("Tree exported to tree_output.txt")