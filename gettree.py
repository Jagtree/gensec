import os
from pathlib import Path

def generate_tree(
    root_dir: str = ".",
    exclude_dirs: list = [
        ".venv", "__pycache__", "node_modules", ".git", "results",
        "examples", "claude", "build", "dist", ".doctrees",
        "_sources", "_static", "multiversion", "{source,user_guide,theory,api,examples}"
    ],
    output_file: str = "tree_output.txt",
):
    """Generate a perfectly structured and indented tree, excluding specified directories."""
    root = Path(root_dir)
    stack = [(root, "", True)]  # (path, prefix, is_last)

    with open(output_file, "w", encoding="utf-8") as f:
        while stack:
            current_path, prefix, is_last = stack.pop()
            items = sorted(
                [item for item in current_path.iterdir() if not (item.is_dir() and item.name in exclude_dirs)],
                key=lambda x: (not x.is_dir(), x.name.lower())
            )

            for i, item in enumerate(items):
                is_last_item = (i == len(items) - 1)
                connector = "└── " if is_last_item else "├── "
                new_prefix = prefix + ("    " if is_last_item else "│   ")

                if item.is_dir():
                    f.write(f"{prefix}{connector}📁 {item.name}\n")
                    stack.append((item, new_prefix, is_last_item))
                else:
                    f.write(f"{prefix}{connector}📄 {item.name}\n")

if __name__ == "__main__":
    generate_tree()
    print("Tree structure exported to 'tree_output.txt'!")