// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/constants.hpp"
#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/pool/average_pool.hpp"
#include "tt_numpy/functions.hpp"

using tt::tt_metal::DataType;
using tt::tt_metal::Device;
using tt::tt_metal::Host;
using tt::tt_metal::Layout;
using tt::tt_metal::Shape;
using tt::tt_metal::Tensor;

Tensor run_avg_pool_2d_resnet(Shape& tensor_shape, Device* device) {
    auto input_tensor = tt::numpy::random::random(tensor_shape, DataType::BFLOAT16);
    auto device_output = average_pool_2d(input_tensor);
    return device_output.cpu();
};

int main () {
    int device_id = 0;
    auto device = tt::tt_metal::CreateDevice(device_id);

    Shape resnet18_shape = {1, 1, 7 * 7, 2048};
    auto result = run_avg_pool_2d_resnet(resnet18_shape, device);

    TT_FATAL(result.get_legacy_shape() == Shape({1, 1, TILE_HEIGHT, 2048}));
    TT_FATAL(result.get_legacy_shape().without_padding() == Shape({1, 1, 1, 2048}));

    TT_FATAL(tt::tt_metal::CloseDevice(device));
    return 0;
}
