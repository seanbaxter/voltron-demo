#include <cstdio>
#include <cuda_runtime_api.h>

template<int X>
int constant_func() {
  return 10 * X;
}

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
    @meta for enum(nvvm_arch_t arch : nvvm_arch_t) {
      if target(__nvvm_arch == arch) {
        // arch is a constant, so we can do things like specialize 
        // templates!
        @meta printf("%d - Generating " + arch.string + " branch\n", __LINE__);
        return constant_func<(int)arch>();
      }
    }

  } else if target(__is_amdgpu_target) {
    @meta for enum(amdgpu_arch_t arch : amdgpu_arch_t) {
      if target(__amdgpu_arch == arch) {
        // arch is a constant, so we can do things like specialize 
        // templates!
        @meta printf("%d - Generating " + arch.string + " branch\n", __LINE__);
        return constant_func<(int)arch>();
      }
    }

  } else {
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


