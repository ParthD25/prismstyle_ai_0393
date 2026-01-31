#!/bin/bash

# Fix for app_links Swift-only plugin import issue
# This script removes the problematic @import app_links; line from GeneratedPluginRegistrant.m

# Patch any GeneratedPluginRegistrant.m under ios/ (covers regenerated copies)
PATCHED=0
echo "Searching for GeneratedPluginRegistrant.m under ios/..."
while IFS= read -r file; do
    if [ -f "$file" ]; then
        echo "Patching $file"
        # Remove any @import <plugin>; fallback lines (keeps #if/#endif intact)
        # Compatible with macOS sed (-i '')
        sed -i '' '/^[[:space:]]*@import[[:space:]]\+[a-zA-Z0-9_]+;[[:space:]]*$/d' "$file" || true
        PATCHED=1
    fi
done < <(find ios -type f -name GeneratedPluginRegistrant.m 2>/dev/null)

if [ $PATCHED -eq 1 ]; then
    echo "Patched GeneratedPluginRegistrant.m files successfully."
else
    echo "No GeneratedPluginRegistrant.m files found under ios/. Ensure Flutter has generated them (run 'flutter pub get' or a build)."
fi