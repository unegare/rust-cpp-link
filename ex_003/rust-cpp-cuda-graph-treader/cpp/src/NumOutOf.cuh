#ifndef NUMOUTOF_H_
#define NUMOUTOF_H_

class NumOutOf {
  size_t num;
  size_t out_of;
protected:
  __device__ void _incOutOf();
public:
  __device__ NumOutOf() = delete;
  __device__ NumOutOf(size_t _num, size_t _out_of);
  __device__ bool isReady() const;
  __device__ size_t getNum() const;
  __device__ size_t getOutOf() const;
  __device__ void inc();
  __device__ void reset();
};

class NumOutOfBuilder: public NumOutOf {
public:
  using NumOutOf::NumOutOf;

	__device__ NumOutOfBuilder();
  __device__ void incOutOf();
};

#endif
