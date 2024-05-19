#include <stdio.h>

__global__ void subtractKernel(float *a, float *b, float *c, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        if (b[tid] != 0) {
            c[tid] = a[tid] / b[tid];
        } else {
            c[tid] = 0; // �������Ϊ�㣬������Ϊ��������ʵ���ֵ
        }
    }
}

int main() {
    int n = 10; // ���鳤��
    float a[n], b[n], c[n]; // ������������

    // Ϊ�������鸳ֵ
    for (int i = 0; i < n; ++i) {
        a[i] = i;
        b[i] = i * i;
    }

    float *dev_a, *dev_b, *dev_c; // �豸��ָ��

    // �����豸���ڴ�
    cudaMalloc((void**)&dev_a, n * sizeof(int));
    cudaMalloc((void**)&dev_b, n * sizeof(int));
    cudaMalloc((void**)&dev_c, n * sizeof(int));

    // ���������ݴ������˿������豸��
    cudaMemcpy(dev_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, n * sizeof(int), cudaMemcpyHostToDevice);

    // �����߳̿���߳���
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // ���� CUDA �˺���
    subtractKernel<<<numBlocks, blockSize>>>(dev_a, dev_b, dev_c, n);

    // �����������豸�˿�����������
    cudaMemcpy(c, dev_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    // ��ӡ������
    printf("Result: ");
    for (int i = 0; i < n; ++i) {
        printf("%f ", c[i]);
    }
    printf("\n");

    // �ͷ��豸���ڴ�
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
