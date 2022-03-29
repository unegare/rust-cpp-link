#include <iostream>
#include <cinttypes>

#include "EdgeState.cuh"
#include "Vertex.cuh"
#include "Graph.cuh"

//template<typename T>
//class Wrapper {
//  EdgeState<T> *arr;
//public:
//  __device__ Wrapper(size_t);
//  __device__ virtual ~Wrapper();
//};
//
//template<typename T>
//__device__ Wrapper<T>::Wrapper(size_t len): arr(nullptr) {
//  printf("%s\n", __PRETTY_FUNCTION__);
//  arr = new EdgeStateBuilder<T>[len];
//}
//
//template<typename T>
//__device__ Wrapper<T>::~Wrapper() {
//  delete[] arr;
//}

__global__ void cuda_kernel(size_t hasher_len) {
//  Wrapper<double> *wp = new Wrapper<double>(10);
//  delete wp;

//  EdgeState<double> *es = new EdgeStateBuilder<double>[100];
//  delete[] es;

//  Vertex<double> *vtx = new VertexBuilder<double>(hasher_len);
//  if (vtx->isOk()) {
//    printf("Ok\n");
//    printf("isReady(): %d\n", vtx->getIn(0).getReadiness().isReady());
//  } else {
//    printf("Not Ok\n");
//  }
//  delete vtx;

  GraphBuilder<double> *gph = new GraphBuilder<double>(hasher_len);
  if (gph->isOk()) {
    printf("Ok\n");
  } else {
    printf("Not Ok\n");
  }
  delete gph;
}

extern "C" {
  void cuda_wrapper() {
    std::cout << __PRETTY_FUNCTION__ << std::endl;
    cuda_kernel<<<1,1>>>(200);
    cudaDeviceSynchronize();
  }
}
