#include <stdio.h>

#define ROWS_A 3 // ���� A ������
#define COLS_A 2 // ���� A ������
#define ROWS_B 2 // ���� B ������
#define COLS_B 4 // ���� B ������

// CUDA �˺�����ִ�о���˷�
__global__ void matrixMultiply(int *a, int *b, int *c, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rowsA && col < colsB) {
        int sum = 0;
        for (int i = 0; i < colsA; ++i) {
            sum += a[row * colsA + i] * b[i * colsB + col];
        }
        c[row * colsB + col] = sum;
    }
}

int main() {
    int *h_a, *h_b, *h_c; // �����������������ˣ�
    int *d_a, *d_b, *d_c; // �������������豸�ˣ�

    // �����������ڴ�
    h_a = (int*)malloc(ROWS_A * COLS_A * sizeof(int));
    h_b = (int*)malloc(ROWS_B * COLS_B * sizeof(int));
    h_c = (int*)malloc(ROWS_A * COLS_B * sizeof(int));

    // Ϊ�������ֵ
    for (int i = 0; i < ROWS_A * COLS_A; ++i) {
        h_a[i] = i;
    }
    for (int i = 0; i < ROWS_B * COLS_B; ++i) {
        h_b[i] = i * i;
    }

    // �����豸���ڴ�
    cudaMalloc((void**)&d_a, ROWS_A * COLS_A * sizeof(int));
    cudaMalloc((void**)&d_b, ROWS_B * COLS_B * sizeof(int));
    cudaMalloc((void**)&d_c, ROWS_A * COLS_B * sizeof(int));

    // ���������ݴ������˿������豸��
    cudaMemcpy(d_a, h_a, ROWS_A * COLS_A * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, ROWS_B * COLS_B * sizeof(int), cudaMemcpyHostToDevice);

    // �����߳̿���߳���
    dim3 blockDim(16, 16);
    dim3 gridDim((COLS_B + blockDim.x - 1) / blockDim.x, (ROWS_A + blockDim.y - 1) / blockDim.y);

    // ���� CUDA �˺���
    matrixMultiply<<<gridDim, blockDim>>>(d_a, d_b, d_c, ROWS_A, COLS_A, COLS_B);

    // �����������豸�˿�����������
    cudaMemcpy(h_c, d_c, ROWS_A * COLS_B * sizeof(int), cudaMemcpyDeviceToHost);

    // ��ӡ������
    printf("Matrix C (result):\n");
    for (int i = 0; i < ROWS_A; ++i) {
        for (int j = 0; j < COLS_B; ++j) {
            printf("%d ", h_c[i * COLS_B + j]);
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
