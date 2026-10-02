"""Module execution entry point: python -m ffa_karad."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
