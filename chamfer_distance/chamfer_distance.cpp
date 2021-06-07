#include <torch/extension.h>

// CUDA forward declarations
void ChamferDistanceKernelLauncher(
    const int b, const int n,
    const float* xyzw1,
    const int m,
    const float* xyzw2,
    float* result,
    int* result_i,
    float* result2,
    int* result2_i);

void ChamferDistanceGradKernelLauncher(
    const int b, const int n,
    const float* xyzw1,
    const int m,
    const float* xyzw2,
    const float* grad_dist1,
    const int* idx1,
    const float* grad_dist2,
    const int* idx2,
    float* grad_xyzw1,
    float* grad_xyzw2);


void chamfer_distance_forward_cuda(
    const at::Tensor xyzw1, 
    const at::Tensor xyzw2, 
    const at::Tensor dist1, 
    const at::Tensor dist2, 
    const at::Tensor idx1, 
    const at::Tensor idx2) 
{
    ChamferDistanceKernelLauncher(xyzw1.size(0), xyzw1.size(1), xyzw1.data<float>(),
                                            xyzw2.size(1), xyzw2.data<float>(),
                                            dist1.data<float>(), idx1.data<int>(),
                                            dist2.data<float>(), idx2.data<int>());
}

void chamfer_distance_backward_cuda(
    const at::Tensor xyzw1,
    const at::Tensor xyzw2, 
    at::Tensor gradxyzw1, 
    at::Tensor gradxyzw2, 
    at::Tensor graddist1, 
    at::Tensor graddist2, 
    at::Tensor idx1, 
    at::Tensor idx2)
{
    ChamferDistanceGradKernelLauncher(xyzw1.size(0), xyzw1.size(1), xyzw1.data<float>(),
                                           xyzw2.size(1), xyzw2.data<float>(),
                                           graddist1.data<float>(), idx1.data<int>(),
                                           graddist2.data<float>(), idx2.data<int>(),
                                           gradxyzw1.data<float>(), gradxyzw2.data<float>());
}


void nnsearch(
    const int b, const int n, const int m,
    const float* xyzw1,
    const float* xyzw2,
    float* dist,
    int* idx)
{
    for (int i = 0; i < b; i++) {
        for (int j = 0; j < n; j++) {
            const float x1 = xyzw1[(i*n+j)*4+0];
            const float y1 = xyzw1[(i*n+j)*4+1];
            const float z1 = xyzw1[(i*n+j)*4+2];
            const float w1 = xyzw1[(i*n+j)*4+3];
            double best = 0;
            int besti = 0;
            for (int k = 0; k < m; k++) {
                const float x2 = xyzw2[(i*m+k)*4+0] - x1;
                const float y2 = xyzw2[(i*m+k)*4+1] - y1;
                const float z2 = xyzw2[(i*m+k)*4+2] - z1;
                const float w2 = xyzw2[(i*m+k)*4+3] - w1;
                const double d=x2*x2+y2*y2+z2*z2+w2*w2;
                if (k==0 || d < best){
                    best = d;
                    besti = k;
                }
            }
            dist[i*n+j] = best;
            idx[i*n+j] = besti;
        }
    }
}


void chamfer_distance_forward(
    const at::Tensor xyzw1, 
    const at::Tensor xyzw2, 
    const at::Tensor dist1, 
    const at::Tensor dist2, 
    const at::Tensor idx1, 
    const at::Tensor idx2) 
{
    const int batchsize = xyzw1.size(0);
    const int n = xyzw1.size(1);
    const int m = xyzw2.size(1);

    const float* xyzw1_data = xyzw1.data<float>();
    const float* xyzw2_data = xyzw2.data<float>();
    float* dist1_data = dist1.data<float>();
    float* dist2_data = dist2.data<float>();
    int* idx1_data = idx1.data<int>();
    int* idx2_data = idx2.data<int>();

    nnsearch(batchsize, n, m, xyzw1_data, xyzw2_data, dist1_data, idx1_data);
    nnsearch(batchsize, m, n, xyzw2_data, xyzw1_data, dist2_data, idx2_data);
}


void chamfer_distance_backward(
    const at::Tensor xyzw1, 
    const at::Tensor xyzw2, 
    at::Tensor gradxyzw1, 
    at::Tensor gradxyzw2, 
    at::Tensor graddist1, 
    at::Tensor graddist2, 
    at::Tensor idx1, 
    at::Tensor idx2) 
{
    const int b = xyzw1.size(0);
    const int n = xyzw1.size(1);
    const int m = xyzw2.size(1);

    const float* xyzw1_data = xyzw1.data<float>();
    const float* xyzw2_data = xyzw2.data<float>();
    float* gradxyzw1_data = gradxyzw1.data<float>();
    float* gradxyzw2_data = gradxyzw2.data<float>();
    float* graddist1_data = graddist1.data<float>();
    float* graddist2_data = graddist2.data<float>();
    const int* idx1_data = idx1.data<int>();
    const int* idx2_data = idx2.data<int>();

    for (int i = 0; i < b*n*4; i++)
        gradxyzw1_data[i] = 0;
    for (int i = 0; i < b*m*4; i++)
        gradxyzw2_data[i] = 0;
    for (int i = 0;i < b; i++) {
        for (int j = 0; j < n; j++) {
            const float x1 = xyzw1_data[(i*n+j)*4+0];
            const float y1 = xyzw1_data[(i*n+j)*4+1];
            const float z1 = xyzw1_data[(i*n+j)*4+2];
            const float w1 = xyzw1_data[(i*n+j)*4+3];
            const int j2 = idx1_data[i*n+j];

            const float x2 = xyzw2_data[(i*m+j2)*4+0];
            const float y2 = xyzw2_data[(i*m+j2)*4+1];
            const float z2 = xyzw2_data[(i*m+j2)*4+2];
            const float w2 = xyzw2_data[(i*m+j2)*4+3];
            const float g = graddist1_data[i*n+j]*2;

            gradxyzw1_data[(i*n+j)*4+0] += g*(x1-x2);
            gradxyzw1_data[(i*n+j)*4+1] += g*(y1-y2);
            gradxyzw1_data[(i*n+j)*4+2] += g*(z1-z2);
            gradxyzw1_data[(i*n+j)*4+3] += g*(w1-w2);
            gradxyzw2_data[(i*m+j2)*4+0] -= (g*(x1-x2));
            gradxyzw2_data[(i*m+j2)*4+1] -= (g*(y1-y2));
            gradxyzw2_data[(i*m+j2)*4+2] -= (g*(z1-z2));
            gradxyzw2_data[(i*m+j2)*4+3] -= (g*(w1-w2));
        }
        for (int j = 0; j < m; j++) {
            const float x1 = xyzw2_data[(i*m+j)*4+0];
            const float y1 = xyzw2_data[(i*m+j)*4+1];
            const float z1 = xyzw2_data[(i*m+j)*4+2];
            const float w1 = xyzw2_data[(i*m+j)*4+3];
            const int j2 = idx2_data[i*m+j];
            const float x2 = xyzw1_data[(i*n+j2)*4+0];
            const float y2 = xyzw1_data[(i*n+j2)*4+1];
            const float z2 = xyzw1_data[(i*n+j2)*4+2];
            const float w2 = xyzw1_data[(i*n+j2)*4+3];
            const float g = graddist2_data[i*m+j]*2;
            gradxyzw2_data[(i*m+j)*4+0] += g*(x1-x2);
            gradxyzw2_data[(i*m+j)*4+1] += g*(y1-y2);
            gradxyzw2_data[(i*m+j)*4+2] += g*(z1-z2);
            gradxyzw2_data[(i*m+j)*4+3] += g*(w1-w2);
            gradxyzw1_data[(i*n+j2)*4+0] -= (g*(x1-x2));
            gradxyzw1_data[(i*n+j2)*4+1] -= (g*(y1-y2));
            gradxyzw1_data[(i*n+j2)*4+2] -= (g*(z1-z2));
            gradxyzw1_data[(i*n+j2)*4+3] -= (g*(w1-w2));
        }
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &chamfer_distance_forward, "ChamferDistance forward");
    m.def("forward_cuda", &chamfer_distance_forward_cuda, "ChamferDistance forward (CUDA)");
    m.def("backward", &chamfer_distance_backward, "ChamferDistance backward");
    m.def("backward_cuda", &chamfer_distance_backward_cuda, "ChamferDistance backward (CUDA)");
}
