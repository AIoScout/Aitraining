#!/bin/bash
# Build the macOS .app + .dmg for distribution.
#
# Apple Silicon rules for a DMG handed to M1/M2/M3 users:
#   - The .app INSIDE the dmg must be arm64 (or the user needs Rosetta).
#     The bundle arch = THIS machine's arch — build the M1 DMG on an
#     Apple Silicon Mac.
#   - arm64 bundles MUST be signed (ad-hoc at minimum) or macOS refuses
#     to launch them.  This script always ad-hoc signs and verifies.
#   - For smoother distribution (no right-click->Open on first launch)
#     sign with a Developer ID and notarize:
#       CODESIGN_IDENTITY="Developer ID Application: ..." \
#       APPLE_ID=you@example.com APP_PASSWORD=xxxx \
#       ./build_macos.sh
#   - Intel Mac distribution: the bundle arch = the BUILDING Python's arch.
#     On Apple Silicon, build x86_64 with a Rosetta venv:
#       softwareupdate --install-rosetta
#       arch -x86_64 /usr/bin/python3 -m venv .venv-x86
#       arch -x86_64 .venv-x86/bin/pip install -r requirements.txt -r requirements-dev.txt
#       arch -x86_64 .venv-x86/bin/python -m PyInstaller TFLiteTraining.spec \
#           --noconfirm --target-arch x86_64
#     or on any machine: TARGET_ARCH=x86_64 ./build_macos.sh (the active
#     python must be an x86_64 interpreter running under Rosetta).
set -e
cd "$(dirname "$0")"

echo "== host arch: $(uname -m)  (TARGET_ARCH=${TARGET_ARCH:-native}) =="
[ "$(uname -m)" = "arm64" ] && [ -z "${TARGET_ARCH:-}" ] && echo "   -> this DMG will run natively on Apple Silicon"
[ "$(uname -m)" = "x86_64" ] && echo "   -> WARNING: x86_64 DMG — M1/M2 users will need Rosetta (macOS auto-prompts)"

PYINSTALLER_ARGS=(--noconfirm)
if [ -n "${TARGET_ARCH:-}" ]; then
  PYINSTALLER_ARGS+=(--target-arch "$TARGET_ARCH")
fi
python -m PyInstaller TFLiteTraining.spec "${PYINSTALLER_ARGS[@]}"

APP="dist/TFLiteTraining.app"
[ -d "$APP" ] || { echo "ERROR: bundle not found at $APP"; exit 1; }

echo "== bundle arch =="
file "$APP/Contents/MacOS/TFLiteTraining"

echo "== signing =="
if [ -n "${CODESIGN_IDENTITY:-}" ]; then
  codesign --force --deep --options runtime --sign "$CODESIGN_IDENTITY" "$APP"
else
  codesign --force --deep --sign - "$APP"
fi
codesign --verify --deep --strict "$APP" && echo "signature OK"

if [ -n "${APPLE_ID:-}" ] && [ -n "${APP_PASSWORD:-}" ]; then
  echo "== notarizing =="
  ditto -c -k --keepParent "$APP" /tmp/TFLiteTraining_notarize.zip
  xcrun notarytool submit /tmp/TFLiteTraining_notarize.zip \
    --apple-id "$APPLE_ID" --password "$APP_PASSWORD" --team-id "${APPLE_TEAM_ID:-}" --wait
  xcrun stapler staple "$APP" && echo "notarization OK"
  rm -f /tmp/TFLiteTraining_notarize.zip
else
  echo "== notarization skipped (set APPLE_ID/APP_PASSWORD to notarize) =="
fi

echo "== building dmg =="
DMG_OUT="dist/TFLiteTraining.dmg"
if [ -n "${TARGET_ARCH:-}" ]; then
  DMG_OUT="dist/TFLiteTraining-${TARGET_ARCH}.dmg"
fi
python -m dmgbuild -s dmg_settings.py "TFLiteTraining" "$DMG_OUT"
echo "== done: $DMG_OUT =="
