#include <cstdio>
#include <cuda_runtime_api.h>

constexpr int func(int i) {
  if consteval {
    @meta printf("%d - Generating if-consteval branch\n", __LINE__);
    return 1 * i;

  } else if target(__is_host_target) {
    @meta printf("%d - Generating host branch\n", __LINE__);
    return 2 * i;

  } else if target(__is_spirv_target) {
    @meta printf("%d - Generting SPIR-V branch\n", __LINE__);
    return 3 * i;

  } else if target(__is_dxil_target) {
    @meta printf("%d - Generating DXIL branch\n", __LINE__);
    return 4 * i;

  } else if target(__is_nvvm_target) {
    @meta printf("%d - Generating NVVM branch\n", __LINE__);
    return 5 * i; 

  } else if target(__is_amdgpu_target) {
    @meta printf("%d - Generating AMDGPU branch\n", __LINE__);
    return 6 * i;

  } else {
    @meta printf("%d - Generating generic branch\n", __LINE__);
    return 7 * i;
  }
}

// It's 1, because it's consteval'd
constexpr int x = func(1);
@meta printf("consteval: %d\n", x);

__global__ void kernel(int i) {
  int x = func(i);
  printf("GPU: %d\n", x);
}

int main() {
  int x = func(1);
  printf("host: %d\n", x);

  kernel<<<1, 1>>>(1);
  cudaDeviceSynchronize();
}


