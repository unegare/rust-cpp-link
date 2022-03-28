#include <iostream>
#include <cinttypes>

#include "EdgeState.cuh"
#include "Vertex.cuh"

__global__ void cuda_kernel(size_t hasher_len) {
  VertexBuilder<double> *vtx = new VertexBuilder<double>(hasher_len);
  if (vtx->isOk()) {
    printf("Ok\n");
    printf("isReady(): %d\n", vtx->getIn(0).getReadiness().isReady());
  } else {
    printf("Not Ok\n");
  }
  delete vtx;
}

extern "C" {
  void cuda_wrapper() {
    std::cout << __PRETTY_FUNCTION__ << std::endl;
    cuda_kernel<<<1,1>>>(10);
    cudaDeviceSynchronize();
  }
}
