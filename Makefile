CXX      := c++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2

SRC   := src/tensor.cpp
TESTS := tests/test_tensor.cpp

build/test_tensor: $(SRC) $(TESTS) src/tensor.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(SRC) $(TESTS) -o $@

.PHONY: test clean pymodule pytest

test: build/test_tensor
	./build/test_tensor

# Build the Python extension module into build/ using the project venv.
# -undefined dynamic_lookup is the macOS way to leave Python symbols
# unresolved until import time.
PY := .venv/bin/python
pymodule: $(SRC) src/bindings.cpp src/tensor.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -shared -fPIC -undefined dynamic_lookup \
		$$($(PY) -m pybind11 --includes) \
		$(SRC) src/bindings.cpp \
		-o build/inferno_core$$($(PY)-config --extension-suffix 2>/dev/null || $(PY) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# Run the Python-side correctness harness (needs `make pymodule` first).
pytest: pymodule
	$(PY) tests/test_vs_torch.py

clean:
	rm -rf build
