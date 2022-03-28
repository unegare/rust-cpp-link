TARGET=main

CXX=g++-10
CXXFLAGS=-std=c++17 -Wall -pedantic -O2
SOURCES=$(shell find src -name \*.cpp)
OBJECTS=${SOURCES:%.cpp=%.o}
CUSOURCES=$(shell find src -name \*.cu)
CUOBJECTS=${CUSOURCES:%.cu=%.o}

TARGET_SHARED_LIB_NAME=cuda_wrapper

.PHONY: all
all: ${TARGET}

${TARGET}: ${OBJECTS} lib${TARGET_SHARED_LIB_NAME}.so
	$(CXX) ${OBJECTS} -L./ -l${TARGET_SHARED_LIB_NAME} -o ${TARGET} -Wl,-rpath,.

lib${TARGET_SHARED_LIB_NAME}.so: ${CUOBJECTS}
	nvcc -shared $^ -o lib${TARGET_SHARED_LIB_NAME}.so

$(CUOBJECTS): % : ${CUSOURCES}
	@echo '-rdc=true serves the role of the separate compilation of the .cu files' > /dev/null
	nvcc --verbose -rdc=true -std=c++17 -O2 -c $(addsuffix .cu,$(basename $@)) -o $(addsuffix .o,$(basename $@)) --compiler-options -fPIC

.PHONY: clean
clean:
	find . -name \*.o -delete
