# Vendored third-party libraries (research only)

`00_extract_pru.py` in the parent directory adds this folder to `sys.path` to read the EUROCONTROL PRU taxi-out
workbook (`txo.xlsx`). Nothing in `src/`, `scripts/` or the final submission chain imports these packages.

| package | version | licence | licence file here | upstream |
|---|---|---|---|---|
| openpyxl | 3.1.5 | MIT | `openpyxl/LICENCE.rst` | https://openpyxl.readthedocs.io, https://foss.heptapod.net/openpyxl/openpyxl |
| et_xmlfile | 2.0.0 | MIT; includes code from the Python standard library (PSF License) | `et_xmlfile/LICENCE.rst`, `et_xmlfile/LICENCE.python`, `et_xmlfile/AUTHORS.txt` | https://foss.heptapod.net/openpyxl/et_xmlfile |

The package files are byte-identical to the PyPI wheels of these versions (checked with `diff -r` on 2026-10-14).
The licence files were copied from the same wheels' `.dist-info` directories; they were missing when the
packages were first committed (7ecd55e).

Both licences are GPLv3-compatible.
