#!/usr/bin/env bash
set -euo pipefail

# Build script for AWS Lambda deployment package
# Usage: ./build_lambda_package.sh [--clean]

CLEAN=false
if [[ "${1:-}" == "--clean" ]]; then
    CLEAN=true
fi

DIST_DIR="dist"
PACKAGE_NAME="tg_summarizer_lambda.zip"
REQUIREMENTS_FILE="requirements.txt"

echo "Building Lambda deployment package..."

# Clean previous build if requested
if [[ "$CLEAN" == "true" ]]; then
    echo "Cleaning previous build artifacts..."
    rm -rf "$DIST_DIR"
    rm -rf build
    rm -rf *.egg-info
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
fi

# Create dist directory
mkdir -p "$DIST_DIR"

# Install dependencies into dist directory
echo "Installing dependencies..."
pip install --platform manylinux2014_x86_64 \
    --target "$DIST_DIR" \
    --implementation cp \
    --python-version 3.12 \
    --only-binary=:all: \
    -r "$REQUIREMENTS_FILE"

# Copy application code
echo "Copying application code..."
cp -r *.py "$DIST_DIR/"

# Create zip archive
echo "Creating deployment package..."
cd "$DIST_DIR"
zip -r ../"$PACKAGE_NAME" . -x "*.pyc" "*.pyo" "__pycache__/*"
cd ..

# Clean up dist directory
rm -rf "$DIST_DIR"

echo "✅ Deployment package created: $PACKAGE_NAME"
echo ""
echo "Next steps:"
echo "1. Upload to Lambda: aws lambda update-function-code --function-name tg-summarizer --zip-file fileb://$PACKAGE_NAME"
echo "2. Or use SAM: sam build && sam package && sam deploy"
echo ""
echo "Package size: $(du -h "$PACKAGE_NAME" | cut -f1)"
