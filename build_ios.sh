#!/bin/bash

# Wrapper script for Flutter iOS builds that automatically fixes app_links import issue
# Usage: ./build_ios.sh [build-args...]

echo "Starting Flutter iOS build with app_links fix..."

# Run the Flutter build
flutter build ios "$@"

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Flutter build completed successfully. Applying app_links fix..."
    ./fix_app_links.sh
    echo "✅ iOS build completed with app_links fix applied!"
else
    echo "❌ Flutter build failed. Fix not applied."
    exit 1
fi