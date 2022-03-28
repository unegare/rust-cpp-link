#include <iostream>
#include "cuda_wrapper_2.h"

extern "C" {
	void cuda_wrapper_2() {
		std::cout << __PRETTY_FUNCTION__ << std::endl;
	}
}
