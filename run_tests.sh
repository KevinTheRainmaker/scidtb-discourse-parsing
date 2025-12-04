#!/bin/bash
# Test runner script for SciDTB discourse parsing project

set -e  # Exit on error

echo "=================================="
echo "SciDTB Discourse Parsing Test Suite"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest is not installed${NC}"
    echo "Install dependencies with: pip install -r requirements-dev.txt"
    exit 1
fi

# Parse command line arguments
TEST_TYPE="${1:-all}"
VERBOSE=""
COVERAGE="--cov=src --cov-report=term-missing --cov-report=html"

# Function to run tests
run_tests() {
    local test_path="$1"
    local description="$2"

    echo -e "${BLUE}Running $description...${NC}"
    pytest $test_path $VERBOSE $COVERAGE
    echo ""
}

# Main test execution
case "$TEST_TYPE" in
    all)
        echo -e "${GREEN}Running all tests with coverage${NC}"
        pytest tests/ $COVERAGE
        ;;

    models)
        run_tests "tests/test_models.py" "model tests"
        ;;

    data)
        echo -e "${GREEN}Running data tests${NC}"
        run_tests "tests/test_data_loader.py" "data loader tests"
        run_tests "tests/test_data_preprocessor.py" "data preprocessor tests"
        ;;

    parsers)
        run_tests "tests/test_parsers.py" "parser tests"
        ;;

    evaluation)
        run_tests "tests/test_evaluation.py" "evaluation tests"
        ;;

    training)
        run_tests "tests/test_training.py" "training tests"
        ;;

    fast)
        echo -e "${GREEN}Running fast tests only (skipping slow tests)${NC}"
        pytest tests/ -m "not slow" $COVERAGE
        ;;

    unit)
        echo -e "${GREEN}Running unit tests only${NC}"
        pytest tests/ -m "unit" $COVERAGE
        ;;

    integration)
        echo -e "${GREEN}Running integration tests only${NC}"
        pytest tests/ -m "integration" $COVERAGE
        ;;

    no-api)
        echo -e "${GREEN}Running tests without API dependencies${NC}"
        pytest tests/ -m "not requires_api" $COVERAGE
        ;;

    parallel)
        echo -e "${GREEN}Running tests in parallel${NC}"
        if ! command -v pytest-xdist &> /dev/null; then
            echo -e "${YELLOW}Warning: pytest-xdist not installed, running sequentially${NC}"
            pytest tests/ $COVERAGE
        else
            pytest tests/ -n auto $COVERAGE
        fi
        ;;

    verbose)
        echo -e "${GREEN}Running all tests with verbose output${NC}"
        pytest tests/ -v $COVERAGE
        ;;

    coverage)
        echo -e "${GREEN}Generating detailed coverage report${NC}"
        pytest tests/ --cov=src --cov-report=term-missing --cov-report=html --cov-report=xml
        echo ""
        echo -e "${GREEN}Coverage report generated:${NC}"
        echo "  - Terminal: above"
        echo "  - HTML: htmlcov/index.html"
        echo "  - XML: coverage.xml"
        ;;

    clean)
        echo -e "${YELLOW}Cleaning test artifacts...${NC}"
        rm -rf .pytest_cache
        rm -rf htmlcov
        rm -f .coverage
        rm -f coverage.xml
        find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
        echo -e "${GREEN}Test artifacts cleaned${NC}"
        ;;

    help|--help|-h)
        echo "Usage: ./run_tests.sh [option]"
        echo ""
        echo "Options:"
        echo "  all          Run all tests with coverage (default)"
        echo "  models       Run model tests only"
        echo "  data         Run data loader and preprocessor tests"
        echo "  parsers      Run parser tests only"
        echo "  evaluation   Run evaluation tests only"
        echo "  training     Run training tests only"
        echo "  fast         Run fast tests only (skip slow tests)"
        echo "  unit         Run unit tests only"
        echo "  integration  Run integration tests only"
        echo "  no-api       Run tests without API dependencies"
        echo "  parallel     Run tests in parallel (requires pytest-xdist)"
        echo "  verbose      Run with verbose output"
        echo "  coverage     Generate detailed coverage report"
        echo "  clean        Clean test artifacts"
        echo "  help         Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./run_tests.sh               # Run all tests"
        echo "  ./run_tests.sh models        # Run model tests only"
        echo "  ./run_tests.sh parallel      # Run tests in parallel"
        echo "  ./run_tests.sh coverage      # Generate coverage report"
        exit 0
        ;;

    *)
        echo -e "${RED}Unknown option: $TEST_TYPE${NC}"
        echo "Run './run_tests.sh help' for usage information"
        exit 1
        ;;
esac

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ All tests passed!${NC}"
    if [ "$TEST_TYPE" != "clean" ] && [ "$TEST_TYPE" != "help" ]; then
        echo -e "${BLUE}View coverage report: htmlcov/index.html${NC}"
    fi
else
    echo ""
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
