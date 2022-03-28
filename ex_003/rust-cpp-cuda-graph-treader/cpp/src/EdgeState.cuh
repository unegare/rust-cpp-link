#ifndef EDGESTATE_H_
#define EDGESTATE_H_

#include "NumOutOf.cuh"

template<typename T>
class EdgeState: protected NumOutOfBuilder {
  T balance;
protected:
  __device__ NumOutOfBuilder& _getNumOutOfBuilder();
public:
  __device__ EdgeState() = delete;
  __device__ virtual ~EdgeState();
  __device__ EdgeState(size_t _num, size_t _out_of);
  __device__ void topUpBalance(T);
  __device__ T getBalance() const;
  __device__ const NumOutOf& getReadiness() const;
  __device__ NumOutOf& getReadiness();
//  __device__ static void* operator new(size_t count);
//  __device__ static void* operator new[](size_t count, std::true_type&);
};

template<typename T>
class EdgeStateBuilder: public EdgeState<T> {
public:
  using EdgeState<T>::EdgeState;
  __device__ EdgeStateBuilder();
  __device__ NumOutOfBuilder& getNumOutOfBuilder();
};

template<typename T>
__device__ EdgeState<T>::~EdgeState() { }
template<typename T>
__device__ EdgeState<T>::EdgeState(size_t _num, size_t _out_of): NumOutOfBuilder(_num, _out_of), balance(0) {} //, readiness(_num, _out_of) { } 
template<typename T>
__device__ void EdgeState<T>::topUpBalance(T amount) {
  balance += amount;
}
template<typename T>
__device__ T EdgeState<T>::getBalance() const {
  return balance;
}
template<typename T>
__device__ NumOutOf& EdgeState<T>::getReadiness() {
  return static_cast<NumOutOf&>(*this);
}
template<typename T>
__device__ NumOutOfBuilder& EdgeState<T>::_getNumOutOfBuilder() {
  return static_cast<NumOutOfBuilder&>(*this);
}


template<typename T>
__device__ EdgeStateBuilder<T>::EdgeStateBuilder(): EdgeState<T>(0, 0) { }
template<typename T>
__device__ NumOutOfBuilder& EdgeStateBuilder<T>::getNumOutOfBuilder() {
  return this->_getNumOutOfBuilder();
//  return static_cast<NumOutOfBuilder&>(*this);
}

#endif
