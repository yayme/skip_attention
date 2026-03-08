__global__ void naive_attention(
    float* Q,
    float* K,
    float* V,
    float* O,
    int seq_len,
    int head_dim
)
{
    // each thread should handle one output element O[i, j]
    int i = blockIdx.x*blockDim.x + threadIdx.x; // each row handles the full sequence for a single head. so we have to move along the row
    int j = blockIdx.y*blockDim.y + threadIdx.y; // each columns handles the full head dimension for a single token. so we have to move along the column
    if( i>= seq_len || j>=head_dim) return;

    S[i,:]= Q[i, :] @ K.T;
    S= S/sqrt(head_dim);
    A= softmax(S, dim=-1);
    O = A @ V;
}