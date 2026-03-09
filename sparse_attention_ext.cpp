#include<torch/extensions.h>

torch::Tensor sparse_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor mask
);

PYBIND11_MODULE (TORCH_EXTENSION_NAME, m){
    m.def ("sparse_attention", &sparse_attention_forward, "Sparse Flash Attention (CUDA)");
}