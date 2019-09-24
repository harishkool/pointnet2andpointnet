#include <torch/torch.h>
#include <vector>
#include <torch/extension.h>
#include <iostream>

#define CHECK_CUDA(x)                                          \
  do {                                                         \
    AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor"); \
  } while (0)

#define CHECK_CONTIGUOUS(x)                                         \
  do {                                                              \
    AT_CHECK(x.is_contiguous(), #x " must be a contiguous tensor"); \
  } while (0)

#define CHECK_IS_INT(x)                              \
  do {                                               \
    AT_CHECK(x.scalar_type() == at::ScalarType::Int, \
             #x " must be an int tensor");           \
  } while (0)

#define CHECK_IS_FLOAT(x)                              \
  do {                                                 \
    AT_CHECK(x.scalar_type() == at::ScalarType::Float, \
             #x " must be a float tensor");            \
  } while (0)

int approxmatch_cuda_forward(int b,int n,int m,const float * xyz1,const float * xyz2,float * match, float * temp);

int matchcost_cuda_forward(int b,int n,int m,const float * xyz1,const float * xyz2,const float * match,float * out);

int matchcost_cuda_backward(int b,int n,int m,const float * xyz1,const float * xyz2,const float * match,float * grad1,float * grad2);



at::Tensor approxmatch_forward(at::Tensor xyz1, at::Tensor xyz2) {
    CHECK_CONTIGUOUS(xyz1);
    CHECK_CONTIGUOUS(xyz2);

    at::Tensor match =
      torch::zeros({xyz1.size(0), xyz2.size(1), xyz1.size(1)},
                   at::device(xyz1.device()).dtype(at::ScalarType::Float));

    at::Tensor temp =
      torch::full({xyz1.size(0), (xyz2.size(1)+xyz1.size(1))*2},1e10,
                    at::device(xyz1.device()).dtype(at::ScalarType::Float));
    approxmatch_cuda_forward(xyz1.size(0), xyz1.size(1), xyz2.size(1), xyz1.data<float>(), xyz2.data<float>(), match.data<float>(), temp.data<float>());
    return match;
}


at::Tensor matchcost_forward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor match) {
    CHECK_CONTIGUOUS(xyz1);
    CHECK_CONTIGUOUS(xyz2);
    CHECK_CONTIGUOUS(match);

    at::Tensor cost =
      torch::zeros({xyz1.size(0)},
                   at::device(xyz1.device()).dtype(at::ScalarType::Float));
    matchcost_cuda_forward(xyz1.size(0), xyz1.size(1), xyz2.size(1), xyz1.data<float>(), xyz2.data<float>(), match.data<float>(), cost.data<float>());
    return cost;
}

int matchcost_backward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor match, at::Tensor grad1, at::Tensor grad2) {
    CHECK_CONTIGUOUS(xyz1);
    CHECK_CONTIGUOUS(xyz2);
    CHECK_CONTIGUOUS(match);
    CHECK_CONTIGUOUS(grad1);
    CHECK_CONTIGUOUS(grad2);

    
    return matchcost_cuda_backward(xyz1.size(0), xyz1.size(1), xyz2.size(1), xyz1.data<float>(), xyz2.data<float>(), match.data<float>(), grad1.data<float>(), grad2.data<float>());
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("approxmatch_forward", &approxmatch_forward, "approxmatch forward (CUDA)");
  m.def("matchcost_forward", &matchcost_forward, "matchcost forward (CUDA)");
  m.def("matchcost_backward", &matchcost_backward, "matchcost backward (CUDA)");
}