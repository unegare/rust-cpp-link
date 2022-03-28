#include "NumOutOf.cuh"

__device__ NumOutOf::NumOutOf(size_t _num, size_t _out_of): num(_num), out_of(_out_of) {}
__device__ bool NumOutOf::isReady() const {
  return num == out_of;
}
__device__ size_t NumOutOf::getNum() const {
  return num;
}
__device__ size_t NumOutOf::getOutOf() const {
  return out_of;
}
__device__ void NumOutOf::inc() {
  num++;
}
__device__ void NumOutOf::reset() {
  num = 0;
}
__device__ void NumOutOf::_incOutOf() {
  out_of++;
}

__device__ NumOutOfBuilder::NumOutOfBuilder(): NumOutOf(0, 0) { }
__device__ void NumOutOfBuilder::incOutOf() {
  _incOutOf();
}
