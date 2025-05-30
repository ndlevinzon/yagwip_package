# YAGWIP: Yet Another GROMACS Wrapper In Python

Because YAGWIP is written entirely in Python3 with minimal dependencies, we have structured our code to be as developer-friendly as possible. The following outlines our directory structures, as well as how our code is written and utilized. The idea here is to be as extendable as possible, so that others can make whatever additions they need to perform their GROMACS molecular simulations

---
## Dependencies
- Python 3.x
- GROMACS installed and available on PATH
- SLURM scheduler (for job scripts)
- Bash

## Project Directory Structure
```
yagwip_package/
    ├── src/
        ├── yagwip/
            ├── templates/
            ├── assets/
            ├── yagwip.py
            ├── tremd_calc.py
            ├── parser.py
        ├── examples/
    ├── docs/

```

## Modules Overview

### `yagwip.py`

### `tremd_calc.py`

### `parser.py`




