#include <stdio.h>

#define N 3 // ����Ĵ�С

// CUDA �˺�����ִ�о���ӷ�
__global__ void matrixAdd(int *a, int *b, int *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        int idx = row * n + col;
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int *h_a, *h_b, *h_c; // �����������������ˣ�
    int *d_a, *d_b, *d_c; // �������������豸�ˣ�

    // �����������ڴ�
    h_a = (int*)malloc(N * N * sizeof(int));
    h_b = (int*)malloc(N * N * sizeof(int));
    h_c = (int*)malloc(N * N * sizeof(int));

    // Ϊ�������ֵ
    for (int i = 0; i < N * N; ++i) {
        h_a[i] = i;
        h_b[i] = i * i;
    }

    // �����豸���ڴ�
    cudaMalloc((void**)&d_a, N * N * sizeof(int));
    cudaMalloc((void**)&d_b, N * N * sizeof(int));
    cudaMalloc((void**)&d_c, N * N * sizeof(int));

    // ���������ݴ������˿������豸��
    cudaMemcpy(d_a, h_a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // �����߳̿���߳���
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    // ���� CUDA �˺���
    matrixAdd<<<gridDim, blockDim>>>(d_a, d_b, d_c, N);

    // �����������豸�˿�����������
    cudaMemcpy(h_c, d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // ��ӡ������
    printf("Matrix C (result):\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%d ", h_c[i * N + j]);
        }
        printf("\n");
    }

    // �ͷ��������ڴ�
    free(h_a);
    free(h_b);
    free(h_c);

    // �ͷ��豸���ڴ�
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
