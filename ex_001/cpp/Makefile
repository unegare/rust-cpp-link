TARGET=libmylib.so

CXX=g++-10

SOURCES=$(shell find . -name *.cpp)
OBJECTS=${SOURCES:%.cpp=%.o}

CXXFLAGS=--std=c++17 -Wall -pedantic -O2 -g -fPIC

.PHONY: all
all: ${TARGET}


${TARGET}: ${OBJECTS}
	# ls -la $?
	# $< - for prerequisite
	$(CXX) --shared $^ -o $@

.PHONY: clean
clean:
	find . -name \*.o -delete

#%.o: %.cpp %.h
