#!/bin/bash
# Run all checks before pushing to catch CI failures early

set -e

echo "🔍 Running pre-push checks..."
echo ""

# Format check
echo "1️⃣  Checking code formatting..."
black --check shipml/ tests/ || {
    echo "❌ Code formatting failed!"
    echo "   Run: black shipml/ tests/"
    exit 1
}
echo "✅ Formatting OK"
echo ""

# Lint check
echo "2️⃣  Linting code..."
ruff check shipml/ tests/ || {
    echo "❌ Linting failed!"
    echo "   Run: ruff check --fix shipml/ tests/"
    exit 1
}
echo "✅ Linting OK"
echo ""

# Type check (allow to fail)
echo "3️⃣  Type checking..."
mypy shipml/ || echo "⚠️  Type check warnings (non-blocking)"
echo ""

# Run tests
echo "4️⃣  Running tests..."
pytest tests/ -v || {
    echo "❌ Tests failed!"
    exit 1
}
echo "✅ Tests OK"
echo ""

echo "✅ All pre-push checks passed! Safe to push."
