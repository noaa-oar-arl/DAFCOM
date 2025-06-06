# Generate Sphinx Documentation for DAFCOM

# This script builds the Sphinx documentation for the DAFCOM package

# Ensure we're in the docs directory
cd "$(dirname "$0")"

# Create required directories if they don't exist
mkdir -p _static _templates _build

# Clean previous build if it exists
if [ -d "_build" ]; then
    echo "Cleaning previous build..."
    rm -rf _build/*
fi

# Generate API documentation from source
echo "Generating API documentation..."
sphinx-apidoc -f -o api ../src/dafcom

# Build HTML documentation
echo "Building HTML documentation..."
export PYTHONPATH="../src:$PYTHONPATH"
sphinx-build -b html -d _build/doctrees . _build/html
echo "Building HTML documentation..."
sphinx-build -b html . _build/html

# Output success message
echo "Documentation built successfully!"
echo "Open the documentation with your browser: _build/html/index.html"

# Optional: Open the documentation in the default browser
if [ "$(uname)" == "Darwin" ]; then
    open _build/html/index.html
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    if [ -n "$DISPLAY" ]; then
        xdg-open _build/html/index.html
    else
        echo "No display server available. Please open the documentation manually."
    fi
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW32_NT" ] || [ "$(expr substr $(uname -s) 1 10)" == "MINGW64_NT" ]; then
    start _build/html/index.html
else
    echo "Unknown OS. Please open the documentation manually."
fi
