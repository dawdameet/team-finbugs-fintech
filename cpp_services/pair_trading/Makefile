# Pair Trading Backtest Makefile

CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -march=native
LDFLAGS =

# Target
TARGET = backtest

# Source files
SOURCES = backtest.cpp
HEADERS = backtest.h
OBJECTS = $(SOURCES:.cpp=.o)

.PHONY: all clean test run

all: $(TARGET)

# Build executable
$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(SOURCES) $(LDFLAGS)

# Compile object files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Run backtest
run: $(TARGET)
	./$(TARGET)

# Run tests
test: $(TARGET)
	./$(TARGET) --test

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Debug build
debug: CXXFLAGS = -std=c++17 -g -Wall -Wextra -DDEBUG
debug: clean $(TARGET)
