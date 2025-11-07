# Makefile for Monte Carlo simulation module

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

TARGET = monte_carlo
SRCS = monte_carlo.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
