#include <iostream>
#include "cuda_wrapper.h"

int main() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  cuda_wrapper();
  return 0;
}
