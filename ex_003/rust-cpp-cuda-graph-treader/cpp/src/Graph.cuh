#ifndef GRAPH_CUH_
#define GRAPH_CUH_

#include "Vertex.cuh"

template<typename T>
class Graph {
  Vertex<T> *arr;
  size_t hasher_len;
  CUDAErrorType err;
protected:
  __device__ VertexBuilder<T>& _getVertexBuilderAt(size_t);
public:
  __device__ Graph(size_t);
  __device__ virtual ~Graph();
  __device__ bool isOk() const;
  __device__ CUDAErrorType getErrorType() const;
  __device__ void reset();
};

template<typename T>
__device__ Graph<T>::Graph(size_t _hasher_len): arr(nullptr), hasher_len(0), err(CUDAErrorType::NoError) {
  VertexBuilder<T> *_arr = new VertexBuilder<T>[_hasher_len];
  if (!_arr) {
    err = CUDAErrorType::MallocError;
    return;
  }
  for (size_t i = 0; i < _hasher_len; i++) {
    if (!_arr[i].resize(hasher_len)) {
//    if (!arr[i].isOk()) {
      delete[] _arr;
      err = CUDAErrorType::MallocError;
      return;
    }
  }
  arr = _arr;
  hasher_len = _hasher_len;
}

template<typename T>
__device__ Graph<T>::~Graph() {}

template<typename T>
__device__ bool Graph<T>::isOk() const { return err == CUDAErrorType::NoError; }

template<typename T>
__device__ CUDAErrorType Graph<T>::getErrorType() const { return err; }

template<typename T>
__device__ VertexBuilder<T>& Graph<T>::_getVertexBuilderAt(size_t ind) {
  return static_cast<VertexBuilder<T>&>(arr[ind]);
}

template<typename T>
__device__ void Graph<T>::reset() {
  for (size_t i = 0; i < hasher_len; i++) {
    arr[i].reset();
  }
}


template<typename T>
class GraphBuilder: public Graph<T> {
public:
  using Graph<T>::Graph;
  __device__ VertexBuilder<T>& getVertexBuilderAt(size_t);
};

template<typename T>
__device__ VertexBuilder<T>& GraphBuilder<T>::getVertexBuilderAt(size_t ind) {
  return Graph<T>::_getVertexBuilderAt(ind);
}

#endif
