CXX := clang++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -Iinclude -fobjc-arc
LDFLAGS := -framework Foundation -framework Metal -framework Accelerate

SRCS := \
	src/hardware_m5.mm \
	src/memory_arena.mm \
	src/metal_backend.mm \
	src/kv_cache.cpp \
	src/tokenizer.cpp \
	src/model_loader.cpp \
	src/engine.cpp \
	src/openai_server.cpp \
	src/main.cpp

TARGET := m5engine

all: $(TARGET)

$(TARGET): $(SRCS) src/kernels.metal
	$(CXX) $(CXXFLAGS) $(SRCS) $(LDFLAGS) -o $(TARGET)

bench: $(TARGET)
	./$(TARGET) bench

test: $(TARGET)
	./$(TARGET) test-api

clean:
	rm -f $(TARGET)

.PHONY: all bench test clean
