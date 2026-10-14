#!/usr/bin/env bash
# Add "AI Buddy" to the GNOME app launcher (Activities search). Remove with --uninstall.
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
APPS="$HOME/.local/share/applications"
ICONS="$HOME/.local/share/icons/hicolor/scalable/apps"

if [[ "${1:-}" == "--uninstall" ]]; then
  rm -f "$APPS/ai-buddy.desktop" "$ICONS/ai-buddy.svg"
  echo "Removed the AI Buddy launcher."
  exit 0
fi

UV="$(command -v uv || true)"
[[ -n "$UV" ]] || { echo "uv not found on PATH" >&2; exit 1; }
mkdir -p "$APPS" "$ICONS"
cp "$DIR/ai_buddy/static/icon.svg" "$ICONS/ai-buddy.svg"
cat > "$APPS/ai-buddy.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=AI Buddy
Comment=Local voice companion and speaking coach on the Ryzen AI NPU
Exec=$UV run --project "$DIR" ai-buddy
Icon=ai-buddy
Terminal=false
Categories=Education;AudioVideo;
StartupWMClass=ai-buddy
DESKTOP
update-desktop-database "$APPS" 2>/dev/null || true
echo "Installed. Search for \"AI Buddy\" in Activities."
