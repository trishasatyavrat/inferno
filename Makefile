CXX      := c++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2

SRC   := src/tensor.cpp
TESTS := tests/test_tensor.cpp

build/test_tensor: $(SRC) $(TESTS) src/tensor.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(SRC) $(TESTS) -o $@

.PHONY: test clean
test: build/test_tensor
	./build/test_tensor

clean:
	rm -rf build
