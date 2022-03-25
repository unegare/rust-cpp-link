TARGET=main
CXXFLAGS=--std=c++17 -Wall -pedantic -O2

CXX=g++-10
CUSOURCES=$(shell find . -name \*.cu)
CUOBJECTS=${CUSOURCES:%.cu=%.o}
SOURCES=$(shell find . -name \*.cpp)
OBJECTS=${SOURCES:%.cpp=%.o}
CUSHAREDLIB=cuda_wrapper

.PHONY: all
all: ${TARGET}

${TARGET}: ${OBJECTS} lib${CUSHAREDLIB}.so
	$(CXX) ${OBJECTS} -L./ -l${CUSHAREDLIB} -o ${TARGET} -Wl,-rpath,. 

lib${CUSHAREDLIB}.so: ${CUOBJECTS}
	nvcc -shared $^ -o lib${CUSHAREDLIB}.so

$(CUOBJECTS): %: ${CUSOURCES}
	nvcc -std=c++17 -O2 -c $? --compiler-options -fPIC
  

.PHONY: clean
clean:
	find . -name \*.o -delete
