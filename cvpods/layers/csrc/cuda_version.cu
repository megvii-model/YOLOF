// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

#include <cuda_runtime_api.h>

namespace cvpods {
int get_cudart_version() {
  return CUDART_VERSION;
}
} // namespace cvpods
