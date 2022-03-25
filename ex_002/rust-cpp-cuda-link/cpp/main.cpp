#include <iostream>
#include "cuda_wrapper.h"

int main() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  cuda_wrapper(NULL, 0);
  return 0;
}
