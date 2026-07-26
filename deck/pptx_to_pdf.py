"""Convert the deck to PDF using the PowerPoint that is installed on this machine.

Run:  python deck/pptx_to_pdf.py [in.pptx] [out.pdf]

LibreOffice and pandoc are not present here, so PowerPoint COM automation is the
conversion path. The presentation is opened without a window and closed in a
``finally`` block so a failure cannot leave an orphan POWERPNT.EXE holding the
file open. If COM is unavailable the script says so plainly and exits non-zero
rather than raising a bare traceback.
"""

from __future__ import annotations

import sys
from pathlib import Path

DECK = Path(__file__).resolve().parent
DEFAULT_IN = DECK / "codebase_engineering.pptx"
DEFAULT_OUT = DECK / "codebase_engineering.pdf"

PP_SAVE_AS_PDF = 32  # ppSaveAsPDF
MSO_FALSE, MSO_TRUE = 0, -1


def _dispatch():
    """Return a PowerPoint.Application COM object, or None with a reason printed."""
    try:
        import win32com.client as com  # type: ignore

        return com.Dispatch("PowerPoint.Application")
    except ImportError:
        pass
    try:
        import comtypes.client as com  # type: ignore

        return com.CreateObject("PowerPoint.Application")
    except ImportError:
        pass
    print(
        "No Python COM bridge found. Install one with:\n"
        "    pip install pywin32\n"
        "(LibreOffice and pandoc are not installed on this machine, so COM is "
        "the available conversion path.)",
        file=sys.stderr,
    )
    return None


def convert(src: Path, dst: Path) -> int:
    if not src.exists():
        print(f"Input not found: {src}\nRun deck/build_deck.py first.", file=sys.stderr)
        return 1

    app = _dispatch()
    if app is None:
        return 1

    presentation = None
    try:
        try:
            app.Visible = MSO_TRUE  # PowerPoint refuses some operations while hidden
        except Exception:  # noqa: BLE001 - not fatal, keep going
            pass
        # Late-bound COM: pass arguments positionally. Keyword arguments need the
        # type library and fail with "cannot be converted to a COM object".
        # Presentations.Open(FileName, ReadOnly, Untitled, WithWindow)
        presentation = app.Presentations.Open(
            str(src), MSO_FALSE, MSO_FALSE, MSO_FALSE
        )
        # SaveAs(FileName, FileFormat) rather than ExportAsFixedFormat: the latter
        # has trailing optional parameters that late-bound pywin32 cannot marshal.
        presentation.SaveAs(str(dst), PP_SAVE_AS_PDF)
    except Exception as exc:  # noqa: BLE001 - surface whatever COM reports
        print(f"PowerPoint conversion failed: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            if presentation is not None:
                presentation.Close()
        finally:
            app.Quit()

    if not dst.exists():
        print("PowerPoint reported success but produced no file.", file=sys.stderr)
        return 1

    print(f"Wrote {dst}  ({dst.stat().st_size / 1024:.0f} KB)")
    return 0


def main() -> int:
    src = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_IN
    dst = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else DEFAULT_OUT
    return convert(src, dst)


if __name__ == "__main__":
    sys.exit(main())
