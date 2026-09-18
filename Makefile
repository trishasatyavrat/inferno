CXX      := c++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2

SRC   := src/tensor.cpp src/ops.cpp src/model.cpp src/checkpoint.cpp src/generate.cpp
HDRS  := src/tensor.h src/ops.h src/model.h src/checkpoint.h src/generate.h
TESTS := tests/test_tensor.cpp

build/test_tensor: $(SRC) $(TESTS) $(HDRS)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(SRC) $(TESTS) -o $@

.PHONY: test clean pymodule pytest bench inferno weights

test: build/test_tensor
	./build/test_tensor

# Build the Python extension module into build/ using the project venv.
# -undefined dynamic_lookup is the macOS way to leave Python symbols
# unresolved until import time.
PY := .venv/bin/python
pymodule: $(SRC) src/bindings.cpp $(HDRS)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -shared -fPIC -undefined dynamic_lookup \
		$$($(PY) -m pybind11 --includes) \
		$(SRC) src/bindings.cpp \
		-o build/inferno_core$$($(PY)-config --extension-suffix 2>/dev/null || $(PY) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# Run the Python-side correctness harness (needs `make pymodule` first).
pytest: pymodule
	$(PY) tests/test_vs_torch.py

build/bench_matmul: $(SRC) bench/bench_matmul.cpp $(HDRS)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(SRC) bench/bench_matmul.cpp -o $@

bench: build/bench_matmul
	./build/bench_matmul

# The command-line tool: build/inferno <weights.bin> <token ids...>
build/inferno: $(SRC) src/main.cpp $(HDRS)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(SRC) src/main.cpp -o $@

inferno: build/inferno

# Download + convert the released GPT-2 small weights (~548 MB) into
# weights/gpt2.bin. One-time; needs huggingface_hub + safetensors in .venv.
weights:
	$(PY) tools/export_gpt2.py

clean:
	rm -rf build
