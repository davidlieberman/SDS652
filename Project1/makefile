CXX      := c++ -std=c++11
CXXFLAGS := -pedantic-errors -Wall -Wextra

all: test

test: test.cpp smc.cpp smc.h
	$(CXX) $(CXXFLAGS) -o test test.cpp smc.cpp

clean:
	rm -rf test
