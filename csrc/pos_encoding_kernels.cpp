#include <torch/all.h>


void rotary_embedding(
  torch::Tensor& positions,  // [batch_size, seq_len] or [num_tokens]
  torch::Tensor& query,  // [batch_size, seq_len, num_heads * head_size] or
                         // [num_tokens, num_heads * head_size]
  torch::Tensor& key,    // [batch_size, seq_len, num_kv_heads * head_size] or
                         // [num_tokens, num_kv_heads * head_size]
  int64_t head_size,
  torch::Tensor& cos_sin_cache,  // [max_position, rot_dim]
  bool is_neox) {
int64_t num_tokens = query.numel() / query.size(-1);
int rot_dim = cos_sin_cache.size(1);
int num_heads = query.size(-1) / head_size;
int num_kv_heads = key.size(-1) / head_size;
int64_t query_stride = query.stride(-2);
int64_t key_stride = key.stride(-2);

dim3 grid(num_tokens);
dim3 block(std::min<int64_t>(num_heads * rot_dim / 2, 512));
const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
VLLM_DISPATCH_FLOATING_TYPES(query.scalar_type(), "rotary_embedding", [&] {
  if (is_neox) {
    vllm::rotary_embedding_kernel<scalar_t, true><<<grid, block, 0, stream>>>(
        positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
        key.data_ptr<scalar_t>(), cos_sin_cache.data_ptr<scalar_t>(), rot_dim,
        query_stride, key_stride, num_heads, num_kv_heads, head_size);
  } else {
    vllm::rotary_embedding_kernel<scalar_t, false>
        <<<grid, block, 0, stream>>>(
            positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(), cos_sin_cache.data_ptr<scalar_t>(),
            rot_dim, query_stride, key_stride, num_heads, num_kv_heads,
            head_size);
  }
});
}