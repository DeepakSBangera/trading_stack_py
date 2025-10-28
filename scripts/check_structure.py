import os
import sys

EXPECT = [
    "app",
    "config",
    "data",
    "db",
    "econo",
    "pricing",
    "reports",
    "src",
    "docs",
    "tests",
    "scripts",
]
missing = [d for d in EXPECT if not os.path.isdir(d)]
print("EXPECTED DIRS:", ", ".join(EXPECT))
if missing:
    print("MISSING:", ", ".join(missing))
    sys.exit(1)
for root, dirs, _files in os.walk(".", topdown=True):
    depth = root.count(os.sep)
    if depth > 2:
        continue
    print(root, "->", ", ".join(sorted(dirs)))
