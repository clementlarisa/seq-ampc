CXX = g++

# Common flags
COMMON_CXXFLAGS = -std=c++17 -Wall -O3 \
                  -Iinclude \
                  -Iext/highfive/include \
                  -Iext/eigen

# Target-specific flags
BENCHMARK_CXXFLAGS = $(COMMON_CXXFLAGS) \
	-mavx2 -msse2 -mfma -march=znver3

HARDWARE_CXXFLAGS = $(COMMON_CXXFLAGS) \
	-march=native \
	-I/usr/include/hdf5/serial 

BENCHMARK_LDFLAGS = $(COMMON_LDFLAGS) -lhdf5
HARDWARE_LDFLAGS = $(COMMON_LDFLAGS) \
	-L/opt/openrobots/lib \
	-Wl,-rpath,/usr/local/lib \
	-Wl,-rpath,/opt/openrobots/lib \
	-lpinocchio_parsers \
	-lpinocchio_default \
	-lfranka -lhdf5_serial

# Sources
BENCHMARK_SRC = src/main_benchmark.cpp
HARDWARE_SRC = src/main_hardware.cpp

# Outputs
BENCHMARK_OUT = main_benchmark
HARDWARE_OUT = main_hardware

all: $(BENCHMARK_OUT) $(HARDWARE_OUT)

benchmark: $(BENCHMARK_SRC)
	$(CXX) $(BENCHMARK_CXXFLAGS) $^ -o $@ $(BENCHMARK_LDFLAGS)

hardware: $(BENCHMARK_SRC) $(HARDWARE_SRC)
	$(CXX) $(HARDWARE_CXXFLAGS) $(BENCHMARK_SRC) -o $(BENCHMARK_OUT) $(HARDWARE_LDFLAGS)
	$(CXX) $(HARDWARE_CXXFLAGS) $(HARDWARE_SRC) -o $(HARDWARE_OUT) $(HARDWARE_LDFLAGS)

clean:
	rm -f $(BENCHMARK_OUT) $(HARDWARE_OUT)