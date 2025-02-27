#pragma once

#include <optional>
#include <torch/library.h>

#include <vector>

void rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
  torch::Tensor& key, int64_t head_size,
  torch::Tensor& cos_sin_cache, bool is_neox);