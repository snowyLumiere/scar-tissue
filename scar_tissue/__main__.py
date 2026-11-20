"""Module entrypoint for `python -m scar_tissue`."""

import sys

from scar_tissue.cli import main

if __name__ == "__main__":
    sys.exit(main())
