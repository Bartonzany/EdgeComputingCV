#include <stdio.h>

__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int n = 1000; // ���鳤��
    int *a, *b, *c; // ������������

    // Ϊ�������鸳ֵ
    a = (int*)malloc(n * sizeof(int));
    b = (int*)malloc(n * sizeof(int));
    c = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; ++i) {
        a[i] = i;
        b[i] = i * i;
    }

    int *dev_a, *dev_b, *dev_c; // �豸��ָ��

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
    vectorAdd<<<numBlocks, blockSize>>>(dev_a, dev_b, dev_c, n);

    // �����������豸�˿�����������
    cudaMemcpy(c, dev_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    // ��ӡ�������Ĳ�������
    printf("Result: ");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", c[i]);
    }
    printf("\n");

    // �ͷ��������ڴ�
    free(a);
    free(b);
    free(c);

    // �ͷ��豸���ڴ�
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
