#!/bin/zsh
# Converts the thesis PDF to PDF/A-2b for submission.
#
# Usage:
#   pdfa_tool/make_pdfa.sh [input.pdf] [--signature scan.pdf] [--signature-page N]
#
#   input.pdf          defaults to main_signed.pdf in the Diplom folder
#   --signature        PDF with the signed Statement of authorship; its page
#                      replaces page N of input.pdf (default N = 3)
#
# Output: Diplom/pdfa/Okaianchenko_Bachelor_Thesis.pdf, checked with veraPDF.
# Images keep their full resolution; PNG figures stay lossless.
# Needs: brew install ghostscript verapdf  (pypdf and Pillow go into pdfa_tool/.venv)
set -euo pipefail

TOOL_DIR=${0:A:h}
ROOT=${TOOL_DIR:h}
IN=$ROOT/main_signed.pdf
if [[ $# -gt 0 && $1 != --* ]]; then IN=$1; shift; fi
IN=${IN:A}
OUT_DIR=$ROOT/pdfa
OUT=$OUT_DIR/Okaianchenko_Bachelor_Thesis.pdf
ICC=/opt/homebrew/share/ghostscript/iccprofiles/srgb.icc
VENV=$TOOL_DIR/.venv

[[ -f $IN ]] || { echo "Input not found: $IN" >&2; exit 1; }
[[ -f $ICC ]] || { echo "sRGB ICC profile not found (brew install ghostscript)" >&2; exit 1; }
if ! "$VENV/bin/python" -c "import pypdf, PIL" 2>/dev/null; then
  python3 -m venv "$VENV" && "$VENV/bin/pip" -q install pypdf pillow
fi
mkdir -p "$OUT_DIR"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "Input:  $IN"
"$VENV/bin/python" -I "$TOOL_DIR/prepare.py" "$IN" "$TMP/prepared.pdf" "$@"

gs -q -dBATCH -dNOPAUSE -dNOOUTERSAVE \
  -sDEVICE=pdfwrite -dPDFA=2 -dPDFACompatibilityPolicy=1 \
  -sColorConversionStrategy=RGB -sProcessColorModel=DeviceRGB \
  -dAutoRotatePages=/None \
  -dDownsampleColorImages=false -dDownsampleGrayImages=false -dDownsampleMonoImages=false \
  -dAutoFilterColorImages=false -dColorImageFilter=/FlateEncode \
  -dAutoFilterGrayImages=false -dGrayImageFilter=/FlateEncode \
  -dPassThroughJPEGImages=true \
  --permit-file-read="$ICC" -sPDFA_ICC="$ICC" \
  -sOutputFile="$OUT" "$TOOL_DIR/PDFA_def.ps" "$TMP/prepared.pdf" 2>&1 \
  | grep -v -E 'Overprint Mode|overprint mode not set|Interpolate true|^$' || true
[[ ${pipestatus[1]} -eq 0 ]] || { echo "Ghostscript failed" >&2; exit 1; }

echo "Output: $OUT ($(du -h "$OUT" | cut -f1))"

if command -v verapdf >/dev/null; then
  REPORT=$(verapdf --flavour 2b --format text "$OUT")
  echo "$REPORT"
  [[ $REPORT == PASS* ]] || { echo "veraPDF: NOT valid PDF/A-2b" >&2; exit 2; }
  echo "veraPDF: valid PDF/A-2b"
else
  echo "veraPDF not installed; check the file with the SLUB validator."
fi
