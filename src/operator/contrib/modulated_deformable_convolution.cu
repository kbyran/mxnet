/*!
 * Copyright (c) 2018 Microsoft
 * Licensed under The MIT License [see LICENSE for details]
 * \file modulated_deformable_convolution.cu
 * \brief
 * \author Yuwen Xiong, Haozhi Qi, Jifeng Dai, Xizhou Zhu, Han Hu
*/

#include "./modulated_deformable_convolution-inl.h"
#include <vector>

namespace mxnet {
namespace op {

  template<>
  Operator* CreateOp<gpu>(ModulatedDeformableConvolutionParam param, int dtype,
    mxnet::ShapeVector *in_shape,
    mxnet::ShapeVector *out_shape,
    Context ctx) {
    Operator *op = NULL;
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      op = new ModulatedDeformableConvolutionOp<gpu, DType>(param);
    })
      return op;
  }

}  // namespace op
}  // namespace mxnet

