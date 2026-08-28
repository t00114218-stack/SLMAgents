#!/bin/sh

# Attempt to locate Java
if [ -n "$JAVA_HOME" ] ; then
    JAVACMD="$JAVA_HOME/bin/java"
else
    JAVACMD="java"
fi

if ! command -v "$JAVACMD" >/dev/null 2>&1; then
    echo "ERROR: JAVA_HOME is not set and no 'java' command could be found in your PATH." >&2
    exit 1
fi

APP_HOME=$(cd "$(dirname "$0")" && pwd)

echo "Building SLMAgents Offline Android App..."

# Check if Android SDK is present
if [ -z "$ANDROID_HOME" ] && [ -d "$HOME/Library/Android/sdk" ]; then
    export ANDROID_HOME="$HOME/Library/Android/sdk"
fi

if command -v gradle >/dev/null 2>&1; then
    gradle assembleDebug
else
    echo "=========================================================================="
    echo "To build and install the APK to your mobile phone:"
    echo "1. Open Android Studio (https://developer.android.com/studio)"
    echo "2. Open Project: $APP_HOME"
    echo "3. Connect your Android phone with USB Cable (enable USB Debugging)"
    echo "4. Click the green 'Run' button (Shift + F10)"
    echo "=========================================================================="
fi
