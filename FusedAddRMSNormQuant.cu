#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdlib>

#define WARP_SIZE 32
#define BLOCK_WARP_NUM 16		// warps per block -> one block processes 16 rows
#define STRIDE 4

constexpr float epsilon = 1e-6f;
// FP8 E4M3 representable maximum; values are clamped to this range so a
// downstream bf16->fp8 cast can never overflow to inf/nan
constexpr float FP8_E4M3_MAX = 448.0f;

// ---------------------------------------------------------------------------
//  a row is processed in chunks of 4 elements,
// and each chunk is loaded/stored with a SINGLE wide instruction: 
// - inputs (bf16): 4 * 2B = 8B -> one float2 load (widest legal for bf16) 
// - output (f32) : 4 * 4B = 16B -> one float4 load/store  
// The union lets us do both through the same storage: 
// - '.transfer' : the packed wide type, used ONLY for the memory ops 
// - '.data[j]' : the unpacked array, used for per-element compute
// No explicit __ldg / manual unrolling of element loads is needed;  
// the union guarantees the memory op and the element view alias the same registers, 
// so the compiler keeps everything in registers between load and store. 
// Note: this requires the base pointers to be 8B/16B aligned respectively,
// ---------------------------------------------------------------------------
union BF16_STRIDE {
    float2 transfer;			// 8B packed view  (4 bf16)
    __nv_bfloat16 data[STRIDE];		// 4-element compute view
};

union FLOAT_STRIDE {
    float4 transfer;			// 16B packed view (4 f32)
    float data[STRIDE];			// 4-element compute view
};

// VPT: "values per thread" in units of STRIDE-chunks. One warp owns one row;
// the row is split into WARP_SIZE * VPT chunks, each lane handling VPT of them.
template <int VPT>
__global__ void FusedAddRMSNormQuant(
    const __nv_bfloat16* X,
__nv_bfloat16* residual,
const __nv_bfloat16* W,
float* Y,
int64_t M,
int64_t N,
float scale)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane  = threadIdx.x % WARP_SIZE;

    // One warp per row; a block of 16 warps covers 16 consecutive rows.
    const int64_t row = (int64_t)blockIdx.x * BLOCK_WARP_NUM + warp_id;
    if (row >= M) 
	// tail guard when M is not a multiple of BLOCK_WARP_NUM
        return;

    const __nv_bfloat16* x_row  = X + row * N;
    __nv_bfloat16* res_row = residual + row * N;
    float* y_row = Y + row * N;

    // Lane l owns the element range [l * VPT * STRIDE, (l+1) * VPT * STRIDE)
    int col_start = lane * VPT * STRIDE;
    
    BF16_STRIDE X_value[VPT];
    BF16_STRIDE R_value[VPT];
    FLOAT_STRIDE Y_value[VPT];

    // ---- Step 1: elementwise add, X + residual -> Y, write back residual ----
    #pragma unroll
    for(int i = 0;i < VPT;i++ )
    {
	// one 8B load per chunk instead of 4 separate bf16 loads
        X_value[i].transfer = *reinterpret_cast<const float2*> (x_row  + col_start + i*STRIDE);
        R_value[i].transfer = *reinterpret_cast<const float2*> (res_row + col_start + i*STRIDE);
        for(int j =0;j<STRIDE;j++)
        {
            Y_value[i].data[j] = __bfloat162float(R_value[i].data[j]) +__bfloat162float(X_value[i].data[j]);
            R_value[i].data[j] = __float2bfloat16(Y_value[i].data[j]);
        }
	// one 8B store for the updated residual
        *reinterpret_cast<float2*>(res_row + col_start + i * STRIDE)  = R_value[i].transfer;
    }

    // ---- Step 2: row sum of squares (local accumulation) ---- 
    float sum_sq = 0.f;
    
    #pragma unroll
    for(int i =0;i<VPT;i++)
    {    
        #pragma unroll
        for(int j =0;j<STRIDE;j++)    
        {    
            sum_sq += Y_value[i].data[j] * Y_value[i].data[j];
        }
    }

    // ---- Warp-wide butterfly (XOR) reduction ----
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, offset);

    // rho = 1 / RMS, the standard RMSNorm reciprocal mean
    float rho = rsqrtf(sum_sq / (float)N + epsilon);        

    // ---- Step 3: load weight row (N-wide, same chunk layout as X) ----
    BF16_STRIDE Wi_bf16[VPT];
    FLOAT_STRIDE Wi_float[VPT];
    #pragma unroll
    for(int i = 0;i < VPT;i++) 
    {  
        Wi_bf16[i].transfer = *reinterpret_cast<const float2*>(W + col_start + i*STRIDE);  
        #pragma unroll
        for(int j =0;j<STRIDE;j++) 
        {   
            Wi_float[i].data[j] = Wi_bf16[i].data[j];
        }
    }

    // ---- Step 4: normalize, apply weight, clamp to FP8 E4M3 range ----
    float temp = rho / scale;  
    
    #pragma unroll
    for(int i = 0;i < VPT;i++)   
    {   
        #pragma unroll
        for(int j = 0;j < STRIDE;j++) 
        {
            Y_value[i].data[j] *= temp;
            Y_value[i].data[j] *= Wi_float[i].data[j]; 
            Y_value[i].data[j] = fminf(fmaxf(Y_value[i].data[j], -FP8_E4M3_MAX), FP8_E4M3_MAX);
        }
    }

    // ---- Step 5: store output -- one 16B float4 per chunk ----
     #pragma unroll
    for(int i = 0;i < VPT;i++)
    {  
        *reinterpret_cast<float4*>(y_row + col_start + i*STRIDE) = Y_value[i].transfer;
    }

    
}

void run_kernel(
    const __nv_bfloat16* X,
__nv_bfloat16* residual,
const __nv_bfloat16* W,
float* Y,
int64_t M,
int64_t N,
float scale)
{
// The kernel's memory layout HARD-REQUIRES that one warp covers exactly one
// built for the fixed sizes of a specific model family:
//   N = 128  -> VPT = 1
//   N = 256  -> VPT = 2
//   N = 384  -> VPT = 3
    int blockdim = WARP_SIZE*BLOCK_WARP_NUM;
    int gridim = (M + BLOCK_WARP_NUM - 1) / BLOCK_WARP_NUM;
    int VPT = N / (WARP_SIZE * STRIDE);
    switch(VPT)  
    {  
        case 1:FusedAddRMSNormQuant<1><<<gridim,blockdim>>>(X,residual,W,Y,M,N,scale);break;
        case 2:FusedAddRMSNormQuant<2><<<gridim,blockdim>>>(X,residual,W,Y,M,N,scale);break;
        case 3:FusedAddRMSNormQuant<3><<<gridim,blockdim>>>(X,residual,W,Y,M,N,scale);break;
        default:
            //N exceeds the supported range
            std::abort();
    }
}