#ifndef VERTEX_CUH_
#define VERTEX_CUH_

#include "ErrorType.h"
#include "EdgeState.cuh"

//class VertexBaseClass {
//public:
//	__device__ virtual ~VertexBaseClass();
//};
//__device__ VertexBaseClass::~VertexBaseClass() { }

template<typename T>
class Vertex {//: virtual public VertexBaseClass {
  EdgeState<T> *in;
  EdgeState<T> *out;
  size_t hasher_len;
  enum CUDAErrorType err;
protected:
	__device__ EdgeStateBuilder<T>& _getBuilderIn(size_t ind);
	__device__ EdgeStateBuilder<T>& _getBuilderOut(size_t ind);
public:
  __device__ Vertex() = delete;
  __device__ Vertex(size_t);
  __device__ ~Vertex();
  __device__ CUDAErrorType getErrorType() const;
  __device__ bool isOk() const;
  __device__ EdgeState<T>& getIn(size_t ind);
  __device__ EdgeState<T>& getOut(size_t ind);
  __device__ size_t getHasherLen() const;
};

template<typename T>
__device__ Vertex<T>::Vertex(size_t _hasher_len): hasher_len(_hasher_len), in(nullptr), out(nullptr) {
  printf("%s\n", __PRETTY_FUNCTION__);
  in = ::new EdgeStateBuilder<T>[hasher_len];
  out = ::new EdgeStateBuilder<T>[hasher_len];
  if (!in || !out) {
    delete[] in;
    delete[] out;
    in = nullptr;
    out = nullptr;
    err = CUDAErrorType::MallocError;
  } else {
    err = CUDAErrorType::NoError;
  }
// useless since !!! cudaFree !!! is NOT FOUND for __device__ code
//  if (cudaMalloc(&in, sizeof(T)*hasher_len) != cudaSuccess) {
//    err = CUDAErrorType::MallocError; 
//    return;
//  }
//  if (cudaMalloc(&out, sizeof(T)*hasher_len) != cudaSuccess) {
//    cudaFree(in);
//    err = CUDAErrorType::MallocError;
//  }
//  err = CUDAErrorType::NoError;
}

template<typename T>
__device__ Vertex<T>::~Vertex() {
  printf("%s\n", __PRETTY_FUNCTION__);
  delete[] in;
  delete[] out;
// nvcc cannot find cudaFree in __device__ code
//  cudaFree(in);
//  cudaFree(out);
}

template<typename T>
__device__ bool Vertex<T>::isOk() const {
  return err == CUDAErrorType::NoError;
}

template<typename T>
__device__ CUDAErrorType Vertex<T>::getErrorType() const {
  return err;
}

template<typename T>
__device__ EdgeState<T>& Vertex<T>::getIn(size_t ind) {
  return in[ind];
}

template<typename T>
__device__ EdgeState<T>& Vertex<T>::getOut(size_t ind) {
  return out[ind];
}

template<typename T>
__device__ size_t Vertex<T>::getHasherLen() const {
  return hasher_len;
}

template<typename T>
__device__ EdgeStateBuilder<T>& Vertex<T>::_getBuilderIn(size_t ind) {
	return static_cast<EdgeStateBuilder<T>&>(in[ind]);
}

template<typename T>
__device__ EdgeStateBuilder<T>& Vertex<T>::_getBuilderOut(size_t ind) {
	return static_cast<EdgeStateBuilder<T>&>(out[ind]);
}

template<typename T>
class VertexBuilder: public Vertex<T> {
public:
	using Vertex<T>::Vertex;
	__device__ ~VertexBuilder();
	__device__ EdgeStateBuilder<T>& getBuilderIn(size_t ind);
	__device__ EdgeStateBuilder<T>& getBuilderOut(size_t ind);
};

template<typename T>
__device__ VertexBuilder<T>::~VertexBuilder() {
	
}

template<typename T>
__device__ EdgeStateBuilder<T>& VertexBuilder<T>::getBuilderIn(size_t ind) {
	return this->_getBuilderIn(ind);
}

template<typename T>
__device__ EdgeStateBuilder<T>& VertexBuilder<T>::getBuilderOut(size_t ind) {
	return this->_getBuilderOut(ind);
}

#endif
