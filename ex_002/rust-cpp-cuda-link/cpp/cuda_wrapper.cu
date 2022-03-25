#include <iostream>
#include <iterator>
#include <cstdio>
#include <cstdlib>

__global__ void cuda_func(double *dev_arr, size_t len) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  size_t step = blockDim.x * gridDim.x;
  printf(
    "blockDim.x = %d | blockDim.y = %d | blockDim.z = %d\n"
    "gridDim.x = %d | gridDim.y = %d | gridDim.z = %d\n",
    blockDim.x, blockDim.y, blockDim.z,
    gridDim.x, gridDim.y, gridDim.z);
  while (i < len) {
    double tmp = dev_arr[i];
    dev_arr[i] = tmp*tmp*tmp;
    i += step;
  }
  printf("Hello from cuda! (%d)\n", threadIdx.x);
}

extern "C" {
  void cuda_wrapper(double * const host_arr, size_t len) {
    if (!host_arr) {
      std::cout << __PRETTY_FUNCTION__ << ':' << __LINE__ << ": _host_arr == NULL" << std::endl;
      return;
    }
//    size_t len;
//    double *host_arr;
//    if (_host_arr) {
//      host_arr = _host_arr;
//      len = _len;
//    } else {
//      len = 100;
//      host_arr = (double*)malloc(len*sizeof(double));
//      if (!host_arr) {
//        std::cout << __PRETTY_FUNCTION__ << ':' << __LINE__ << ": malloc failed" << std::endl;
//        return;
//      }
//      for (size_t i = 0; i < len; i++) {
//        host_arr[i] = i;
//      }
//    }

    double *dev_arr = NULL;
    if (cudaMalloc((void**)&dev_arr, len*sizeof(double)) != cudaSuccess) {
      std::cout << __PRETTY_FUNCTION__ << ':' << __LINE__ << ": cudaMalloc failed" << std::endl; 
      return;
    }
    if (cudaMemcpy(dev_arr, host_arr, len*sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
      std::cout << __PRETTY_FUNCTION__ << ':' << __LINE__ << ": cudaMemcpy failed" << std::endl;
      return;
    }
    cuda_func<<<3,3>>>(dev_arr, len);
    if (cudaMemcpy(host_arr, dev_arr, len*sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
      std::cout << __PRETTY_FUNCTION__ << ':' << __LINE__ << ": cudaMemcpy failed" << std::endl;
      return;
    }
    cudaDeviceSynchronize();

//    for (size_t i = 0; i < len; i++) {
//      if (host_arr[i] != i*i) {
//        std::cout << "i: " << i << " | " << host_arr[i] << " != " << i*i << std::endl;
//      }
//    }

    std::copy(host_arr, host_arr + len, std::ostream_iterator<double>(std::cout, ", "));
    std::cout << std::endl;

    cudaFree(dev_arr);
//    free(host_arr);
  }
}
