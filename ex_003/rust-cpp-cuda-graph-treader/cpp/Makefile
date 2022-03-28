TARGET=main

CXX=g++-10
CXXFLAGS=-std=c++17 -Wall -pedantic -O2

SOURCES=$(shell find src -name \*.cpp)
HEADERS=$(shell find src -name \*.h)
OBJECTS=${SOURCES:%.cpp=%.o}

CUSOURCES=$(shell find src -name \*.cu)
CUHEADERS=$(shell find src -name \*.cuh)
CUOBJECTS=${CUSOURCES:%.cu=%.o}

TARGET_SHARED_LIB_NAME=cuda_wrapper

.PHONY: all
all: ${TARGET}

${TARGET}: ${OBJECTS} lib${TARGET_SHARED_LIB_NAME}.so
	$(CXX) ${OBJECTS} -L./ -l${TARGET_SHARED_LIB_NAME} -o ${TARGET} -Wl,-rpath,.

lib${TARGET_SHARED_LIB_NAME}.so: ${CUOBJECTS} ${CUHEADERS}
	nvcc -shared $(filter %.o, $^) -o lib${TARGET_SHARED_LIB_NAME}.so

$(CUOBJECTS): %.o : %.cu
	@echo '-rdc=true serves the role of the separate compilation of the .cu files' > /dev/null
	@#nvcc --verbose -rdc=true -std=c++17 -O2 -c $(addsuffix .cu,$(basename $@)) -o $(addsuffix .o,$(basename $@)) --compiler-options -fPIC
	nvcc -rdc=true -std=c++17 -O2 -c $(addsuffix .cu,$(basename $@)) -o $(addsuffix .o,$(basename $@)) --compiler-options -fPIC

${CUOBJECTS}: ${CUHEADERS}

${OBJECTS}: ${HEADERS}

.PHONY: clean
clean:
	find . -name \*.o -delete
