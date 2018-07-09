// AUTOMATICALLY GENERATED FILE, DO NOT EDIT
#include <string>
#include "caffe/common.hpp"
#ifdef USE_GREENTEA
#ifndef GREENTEA_CL_KERNELS_HPP_
#define GREENTEA_CL_KERNELS_HPP_
#include "caffe/greentea/greentea.hpp"
#include "viennacl/backend/opencl.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/ocl/context.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
namespace caffe {
viennacl::ocl::program & RegisterMyCommonKernels(viennacl::ocl::context *ctx);
template <typename Dtype>
viennacl::ocl::program & RegisterMyKernels(viennacl::ocl::context *ctx);
template <typename Dtype>
std::string getMyKernelBundleName(int index);
int getMyKernelBundleCount();
template<typename Dtype>
std::string getMyKernelBundleSource(int index);
}  // namespace caffe
#endif
#endif
