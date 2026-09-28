#pragma once
#include "gemm_kernel.cu"
#include "cuda_check.cuh"
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

int main(int argc, char** argv) {
    size_t M = argc > 1 ? atoll(argv[1]) : 128;
    size_t N = argc > 2 ? atoll(argv[2]) : 128;   // Only supports 128/256/384
    float scale = argc > 3 ? atof(argv[3]) : 1.0f;

    if (N % 128 != 0 || N >= 512) 
    {
        printf("Unsupported N=%zu (must be 128/256/384/512)\n", N);
        return 1;
    }
    printf("FusedAddRMSNormQuant: M=%zu N=%zu scale=%f\n", M, N, scale);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    // host data
    std::vector<__nv_bfloat16> hX(M * N), hRes(M * N), hW(N);
    for (auto& v : hX)   v = __float2bfloat16(dist(rng));
    for (auto& v : hRes) v = __float2bfloat16(dist(rng));
    for (auto& v : hW)   v = __float2bfloat16(dist(rng));

    // reference result
    std::vector<float> hRef(M * N);
    constexpr float eps = 1e-6f, FP8_MAX = 448.0f;
    for (size_t r = 0; r < M; ++r) {
        float sum_sq = 0.f;
        for (size_t c = 0; c < N; ++c) {
            float x = __bfloat162float(hX[r * N + c]);
            float res = __bfloat162float(hRes[r * N + c]);
            hRef[r * N + c] = x + res;
            sum_sq += hRef[r * N + c] * hRef[r * N + c];
        }
        float rho = 1.0f / sqrtf(sum_sq / (float)N + eps);
        for (size_t c = 0; c < N; ++c) {
            float w = __bfloat162float(hW[c]);
            float y = hRef[r * N + c] * rho / scale * w;
            hRef[r * N + c] = fminf(fmaxf(y, -FP8_MAX), FP8_MAX);
        }
    }

    // device data
    __nv_bfloat16 *dX, *dRes, *dW;
    float* dY;
    CHECK_CUDA(cudaMalloc(&dX,   hX.size()   * 2));
    CHECK_CUDA(cudaMalloc(&dRes, hRes.size() * 2));
    CHECK_CUDA(cudaMalloc(&dW,   hW.size()   * 2));
    CHECK_CUDA(cudaMalloc(&dY,   M * N * 4));
    CHECK_CUDA(cudaMemcpy(dX,   hX.data(),   hX.size()   * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dRes, hRes.data(), hRes.size() * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dW,   hW.data(),   hW.size()   * 2, cudaMemcpyHostToDevice));

    // 1.check if the calculation is correct
    run_kernel(dX, dRes, dW, dY, M, N, scale);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    std::vector<float> hY(M * N);
    CHECK_CUDA(cudaMemcpy(hY.data(), dY, M * N * 4, cudaMemcpyDeviceToHost));

    double max_err = 0;
    for (size_t i = 0; i < M * N; ++i) {
        double ref = hRef[i];
        double denom = fabs(ref) > 1e-3 ? fabs(ref) : 1e-3;
        max_err = fmax(max_err, fabs(hY[i] - ref) / denom);
    }
    printf("max relative error = %e -> %s\n", max_err, max_err < 2e-2 ? "PASS" : "FAIL");
    if (max_err >= 2e-2) return 1;

    std::vector<__nv_bfloat16> hResNew(M * N);
    CHECK_CUDA(cudaMemcpy(hResNew.data(), dRes, M * N * 2, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < M * N; ++i) {
        float expect = __bfloat162float(hX[i]) + __bfloat162float(hRes[i]);
        if (fabsf(__bfloat162float(hResNew[i]) - expect) > 1e-2f) {
            printf("residual mismatch at %zu\n", i);
            return 1;
        }
    }

    // allocate a buffer much larger than L2 for flushing
    const size_t FLUSH_BYTES = 256ull << 20;   // 256MB
    float* dFlush;
    CHECK_CUDA(cudaMalloc(&dFlush, FLUSH_BYTES));
    CHECK_CUDA(cudaMemset(dFlush, 0, FLUSH_BYTES));

    const int ITERS = 20;
    std::vector<float> times(ITERS);
    std::vector<cudaEvent_t> evBeg(ITERS), evEnd(ITERS);
    for (auto& e : evBeg) cudaEventCreate(&e);
    for (auto& e : evEnd) cudaEventCreate(&e);

    // 2.Preheating 
    for (int i = 0; i < 5; ++i) run_kernel(dX, dRes, dW, dY, M, N, scale);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 4. Timer
    for (int i = 0; i < ITERS; ++i) {
        // flush outside the timing interval
        //first replace all of L2 with the flush buffer data
        cudaMemsetAsync(dFlush, 0, FLUSH_BYTES); 
        CHECK_CUDA(cudaDeviceSynchronize());
        cudaEventRecord(evBeg[i]);
        run_kernel(dX, dRes, dW, dY, M, N, scale);
        cudaEventRecord(evEnd[i]);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    for (int i = 0; i < ITERS; ++i) {
        float ms;
        cudaEventElapsedTime(&ms, evBeg[i], evEnd[i]);
        times[i] = ms;
    }
    float avg = times[0];
    for (int i = 1; i < ITERS; ++i) avg += times[i];
    avg /= ITERS;

    double bytes = (double)M * N * 10.0;
    printf("avg time: %.4f ms, effective bandwidth: %.2f GB/s\n",
           avg, bytes / (avg * 1e6));

    cudaFree(dX); cudaFree(dRes); cudaFree(dW);
    cudaFree(dY); cudaFree(dFlush);
    return 0;
}