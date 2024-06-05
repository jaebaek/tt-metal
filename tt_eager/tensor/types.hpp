// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "common/bfloat16.hpp"
#include "tt_eager/tensor/host_buffer/types.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/tt_stl/concepts.hpp"
#include "tt_metal/tt_stl/reflection.hpp"
#include "tt_metal/common/core_coord.h"

namespace tt {

namespace tt_metal {

static constexpr std::uint8_t VERSION_ID = 3;

enum class Layout { ROW_MAJOR = 0, TILE = 1, INVALID = 2 };

enum class DataType {
    BFLOAT16 = 0,
    FLOAT32 = 1,
    UINT32 = 2,
    BFLOAT8_B = 3,
    BFLOAT4_B = 4,
    UINT16 = 6,
    INT32 = 7,
    INVALID = 8,
};

inline bool is_floating_point(DataType dtype) {
    switch (dtype) {
        case DataType::BFLOAT16:
        case DataType::FLOAT32:
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B:
            return true;
        default:
            return false;
    }
}

enum class StorageType {
    OWNED,
    DEVICE,
    BORROWED,           // for storing torch/numpy/etc tensors
    MULTI_DEVICE,       // on-device storage for multi-device context
    MULTI_DEVICE_HOST,  // host storage for multi-device context
};

struct AllGatherTensor{};
bool operator==(const AllGatherTensor&, const AllGatherTensor&);
struct ReplicateTensor {};
bool operator==(const ReplicateTensor&, const ReplicateTensor&);
struct ShardTensor {
    int shard_dimension;
    ShardTensor(int shard_dimension) : shard_dimension(shard_dimension) {}
};
bool operator==(const ShardTensor& lhs, const ShardTensor& rhs);

// DistributedTensorConfig is a variant of different ways in which a tensor can be distributed across devices.
using DistributedTensorConfig = std::variant<ReplicateTensor, ShardTensor, AllGatherTensor>;
DistributedTensorConfig get_distributed_tensor_config(const std::unordered_map<std::string, std::string>& metadata);


tt::DataFormat datatype_to_dataformat_converter(DataType datatype);

static constexpr std::size_t MAX_NUM_DIMENSIONS = 8;

struct Padding {
    enum class PadValue { Any, Zero, Infinity, NegativeInfinity };

    struct PadDimension {
        std::size_t front;
        std::size_t back;

        static constexpr auto attribute_names = std::make_tuple("front", "back");
        const auto attribute_values() const { return std::make_tuple(std::cref(this->front), std::cref(this->back)); }
    };

    std::size_t rank_;
    std::array<PadDimension, MAX_NUM_DIMENSIONS> pad_dimensions_;
    PadValue pad_value_;

    Padding(const Padding &) = default;
    Padding &operator=(const Padding &) = default;
    Padding(Padding &&) = default;
    Padding &operator=(Padding &&) = default;
    ~Padding() = default;

    Padding(const std::size_t rank);
    Padding(const std::initializer_list<PadDimension> pad_dimensions, PadValue pad_value);
    Padding(const std::vector<PadDimension> &pad_dimensions, PadValue pad_value);

    template <std::size_t Rank>
    Padding(const std::array<std::array<uint32_t, 2>, Rank> pad_dimensions, PadValue pad_value) :
        rank_(pad_dimensions.size()), pad_dimensions_{}, pad_value_(pad_value) {
        for (auto index = 0; index < Rank; index++) {
            this->pad_dimensions_[index] = {.front = pad_dimensions[index][0], .back = pad_dimensions[index][1]};
        }
    }

    PadDimension &operator[](const std::int64_t index);
    const PadDimension operator[](const std::int64_t index) const;

    PadValue pad_value() const;

    static constexpr auto attribute_names = std::make_tuple("rank", "pad_dimensions", "pad_value");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->rank_), std::cref(this->pad_dimensions_), std::cref(this->pad_value_));
    }
    friend std::ostream &operator<<(std::ostream &os, const Padding &padding);
};

inline std::ostream &operator<<(std::ostream &os, const Padding &padding) {
    os << "Padding(";
    os << "rank: " << padding.rank_;
    os << ", pad_dimensions: [";
    for (std::size_t i = 0; i < padding.rank_; ++i) {
        os << "{front: " << padding.pad_dimensions_[i].front << ", back: " << padding.pad_dimensions_[i].back << "}";
        if (i < padding.rank_ - 1)
            os << ", ";
    }
    os << "]";
    os << ", pad_value: ";
    switch (padding.pad_value_) {
        case Padding::PadValue::Any: os << "Any"; break;
        case Padding::PadValue::Zero: os << "Zero"; break;
        case Padding::PadValue::Infinity: os << "Infinity"; break;
        case Padding::PadValue::NegativeInfinity: os << "NegativeInfinity"; break;
        default: os << "Unknown";
    }
    os << ")";
    return os;
}

bool operator==(const Padding &, const Padding &);
bool operator!=(const Padding &, const Padding &);

class Shape {
    std::size_t rank_;
    std::array<uint32_t, MAX_NUM_DIMENSIONS> dimensions_;
    Padding padding_;

   public:
    Shape(const Shape &) = default;
    Shape &operator=(const Shape &) = default;
    Shape(Shape &&) = default;
    Shape &operator=(Shape &&) = default;
    ~Shape() = default;

    Shape(const std::initializer_list<uint32_t>);
    Shape(const std::vector<uint32_t> &);
    Shape(const std::initializer_list<uint32_t>, const Padding &);
    Shape(const std::vector<uint32_t> &, const Padding &);

    explicit Shape(const Shape &, const Padding &);

    template <std::size_t Rank>
    Shape(const std::array<uint32_t, Rank> &shape) : rank_(Rank), dimensions_{}, padding_{Rank} {
        for (auto index = 0; index < Rank; index++) {
            this->dimensions_[index] = shape[index];
        }
    }

    template <std::size_t Rank>
    explicit Shape(const std::array<uint32_t, Rank> &shape, const std::array<uint32_t, Rank> &shape_tile_padding) :
        rank_(Rank), dimensions_{}, padding_{Rank} {
        for (auto index = 0; index < Rank; index++) {
            auto padded_dimension = shape_tile_padding[index];
            this->dimensions_[index] = padded_dimension;
            this->padding_[index] = {.front = 0, .back = padded_dimension - shape[index]};
        }
    }
    explicit Shape(const std::vector<uint32_t> &shape, const std::vector<uint32_t> &shape_tile_padding) :
        rank_(shape.size()), dimensions_{}, padding_{shape.size()} {
        TT_ASSERT(shape.size() == shape_tile_padding.size(), "Shape and shape_tile_padding must have the same size");
        for (auto index = 0; index < shape.size(); index++) {
            auto padded_dimension = shape_tile_padding[index];
            this->dimensions_[index] = padded_dimension;
            this->padding_[index] = {.front = 0, .back = padded_dimension - shape[index]};
        }
    }

    std::size_t rank() const;
    std::size_t size() const;

    uint32_t &operator[](const std::int64_t index);
    const uint32_t operator[](const std::int64_t index) const;

    const uint32_t *begin() const;
    const uint32_t *end() const;

    const Padding &padding() const;
    const Shape without_padding() const;

    const uint32_t get_normalized_index(std::int64_t index) const;

    static constexpr auto attribute_names = std::make_tuple("rank", "dimensions", "padding");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->rank_), std::cref(this->dimensions_), std::cref(this->padding_));
    }
    friend std::ostream &operator<<(std::ostream &os, const Shape &shape);
};

inline std::ostream &operator<<(std::ostream &os, const Shape &shape) {
    const auto shape_without_padding = shape.without_padding();
    const auto &padding = shape.padding();
    os << "Shape([";
    for (auto i = 0; i < shape_without_padding.rank(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << shape_without_padding[i];
        if (padding[i].back > 0) {
            os << "[" << shape[i] << "]";
        }
    }
    os << "])";
    return os;
}

bool operator==(const Shape &, const Shape &);
bool operator!=(const Shape &, const Shape &);

struct MemoryConfig {
    TensorMemoryLayout memory_layout = TensorMemoryLayout::INTERLEAVED;  // Interleave the data across multiple banks
    BufferType buffer_type = BufferType::DRAM;                           // Can be either DRAM or L1
    std::optional<ShardSpec> shard_spec = std::nullopt;
    bool is_sharded() const;
    bool is_l1() const;
    bool is_dram() const;

    static constexpr auto attribute_names = std::make_tuple("memory_layout", "buffer_type", "shard_spec");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->memory_layout), std::cref(this->buffer_type), std::cref(this->shard_spec));
    }
};

bool operator==(const MemoryConfig &config_a, const MemoryConfig &config_b);
bool operator!=(const MemoryConfig &config_a, const MemoryConfig &config_b);

void dump_memory_config(std::ostream &output_stream, const MemoryConfig &memory_config);
void dump_memory_config(const std::string &file_name, const MemoryConfig &memory_config);

MemoryConfig load_memory_config(std::ifstream &input_stream);
MemoryConfig load_memory_config(const std::string &file_name);

using OwnedBuffer = std::variant<
    owned_buffer::Buffer<uint16_t>,
    owned_buffer::Buffer<int32_t>,
    owned_buffer::Buffer<uint32_t>,
    owned_buffer::Buffer<float>,
    owned_buffer::Buffer<bfloat16>>;

// HostDataType supports all types included in OwnedBuffer as well as void*
static_assert(
    std::variant_size_v<OwnedBuffer> + 1 == std::variant_size_v<tt::tt_metal::HostDataType>,
    "The data types supported in OwnedBuffer must match those in HostDataType.");
struct OwnedStorage {
    OwnedBuffer buffer;
    OwnedStorage() = default;

    static constexpr auto attribute_names = std::make_tuple();
    const auto attribute_values() const { return std::make_tuple(); }

    inline void insert_buffer(OwnedBuffer buffer_) {
        this->buffer = buffer_;
    }

    inline OwnedBuffer get_buffer() const {
        return this->buffer;
    }

};

using DeviceBuffer = std::shared_ptr<Buffer>;
struct DeviceStorage {
    DeviceBuffer buffer;
    DeviceStorage() = default;

    const MemoryConfig memory_config() const {
        if (this->buffer.get() == nullptr) {
            TT_THROW("MemoryConfig can only be obtained if the buffer is not null");
        }

        std::optional<ShardSpec> shard_spec = std::nullopt;
        if (is_sharded(this->buffer->buffer_layout())) {
            shard_spec = this->buffer->shard_spec().tensor_shard_spec;
        }
        return MemoryConfig{
            .memory_layout = this->buffer->buffer_layout(),
            .buffer_type = this->buffer->buffer_type(),
            .shard_spec = shard_spec};
    }

    inline void insert_buffer(DeviceBuffer buffer_) {
        this->buffer = buffer_;
    }

    inline DeviceBuffer get_buffer() const {
        return this->buffer;
    }
    static constexpr auto attribute_names = std::make_tuple("memory_config");
    const auto attribute_values() const { return std::make_tuple(this->memory_config()); }
};

using BorrowedBuffer = std::variant<
    borrowed_buffer::Buffer<uint16_t>,
    borrowed_buffer::Buffer<int32_t>,
    borrowed_buffer::Buffer<uint32_t>,
    borrowed_buffer::Buffer<float>,
    borrowed_buffer::Buffer<bfloat16>>;
struct BorrowedStorage {
    BorrowedBuffer buffer;

    BorrowedStorage() = default;
    std::function<void()> on_creation_callback = [] {};
    std::function<void()> on_destruction_callback = [] {};

    explicit BorrowedStorage(
        const BorrowedBuffer &buffer,
        const std::function<void()> &on_creation_callback,
        const std::function<void()> &on_destruction_callback) :
        buffer(buffer), on_creation_callback(on_creation_callback), on_destruction_callback(on_destruction_callback) {
        this->on_creation_callback();
    }

    BorrowedStorage(const BorrowedStorage &other) :
        buffer(other.buffer),
        on_creation_callback(other.on_creation_callback),
        on_destruction_callback(other.on_destruction_callback) {
        this->on_creation_callback();
    }

    BorrowedStorage operator=(const BorrowedStorage &other) {
        this->buffer = other.buffer;
        this->on_creation_callback = other.on_creation_callback;
        this->on_destruction_callback = other.on_destruction_callback;
        this->on_creation_callback();
        return *this;
    }

    BorrowedStorage(BorrowedStorage &&other) :
        buffer(other.buffer),
        on_creation_callback(other.on_creation_callback),
        on_destruction_callback(other.on_destruction_callback) {
        other.on_creation_callback = [] {};
        other.on_destruction_callback = [] {};
    }

    BorrowedStorage operator=(BorrowedStorage &&other) {
        this->buffer = other.buffer;
        this->on_creation_callback = other.on_creation_callback;
        this->on_destruction_callback = other.on_destruction_callback;
        other.on_creation_callback = [] {};
        other.on_destruction_callback = [] {};
        return *this;
    }

    ~BorrowedStorage() { this->on_destruction_callback(); }

    static constexpr auto attribute_names = std::make_tuple();
    const auto attribute_values() const { return std::make_tuple(); }
};

struct MultiDeviceHostStorage {
        DistributedTensorConfig strategy;
        std::vector<OwnedBuffer> buffers;
        std::vector<Shape> shapes;
        mutable std::mutex mtx;
        MultiDeviceHostStorage() = default;
        MultiDeviceHostStorage(DistributedTensorConfig strategy_, std::vector<OwnedBuffer> buffers_, std::vector<Shape> shapes_) : strategy(strategy_), buffers(buffers_), shapes(shapes_) {}
        MultiDeviceHostStorage(MultiDeviceHostStorage &&other) {
            std::lock_guard<std::mutex> lock(mtx);
            strategy = other.strategy;
            buffers = other.buffers;
            shapes = other.shapes;
        }

        MultiDeviceHostStorage(const MultiDeviceHostStorage &other) {
            std::lock_guard<std::mutex> lock(mtx);
            strategy = other.strategy;
            buffers = other.buffers;
            shapes = other.shapes;
        }

        MultiDeviceHostStorage &operator=(const MultiDeviceHostStorage &other) {
            std::lock_guard<std::mutex> lock(mtx);
            strategy = other.strategy;
            buffers = other.buffers;
            shapes = other.shapes;
            return *this;
        }

        MultiDeviceHostStorage &operator=( MultiDeviceHostStorage &&other) {
            std::lock_guard<std::mutex> lock(mtx);
            strategy = other.strategy;
            buffers = other.buffers;
            shapes = other.shapes;
            return *this;
        }

        bool operator == (const MultiDeviceHostStorage& other) {
            return this->strategy == other.strategy and this->buffers == other.buffers and this->shapes == other.shapes;
        }

        static constexpr auto attribute_names = std::make_tuple();
        const auto attribute_values() const { return std::make_tuple(); }

        // Helper Functions - Getters and setters to get/modify storage attributes. These are needed to
        // preinitialize empty tensor handles and use/populate them in the worker threads.
        void insert_buffer_and_shape_for_device(int buffer_index, const OwnedBuffer buffer, const Shape shape) {
            std::lock_guard<std::mutex> lock(mtx);
            buffers[buffer_index] = buffer;
            shapes[buffer_index] = shape;
        }

        OwnedBuffer get_buffer(int buffer_index) const {
            std::lock_guard<std::mutex> lock(mtx);
            TT_ASSERT(buffer_index < buffers.size(), "Buffer not found for buffer_index " + std::to_string(buffer_index));
            return buffers[buffer_index];;
        }

        OwnedBuffer& get_buffer(int buffer_index) {
            std::lock_guard<std::mutex> lock(mtx);
            TT_ASSERT(buffer_index < buffers.size(), "Buffer not found for buffer_index " + std::to_string(buffer_index));
            return buffers[buffer_index];;
        }

        Shape get_tensor_shape(int shape_index) const {
            std::lock_guard<std::mutex> lock(mtx);
            TT_ASSERT(shape_index < shapes.size(), "Buffer not found for device " + std::to_string(shape_index));
            return shapes[shape_index];
        }

        uint32_t num_buffers() const {
            std::lock_guard<std::mutex> lock(mtx);
            return buffers.size();
        }
    };

    struct MultiDeviceStorage {
        DistributedTensorConfig strategy;
        std::vector<int> ordered_device_ids;
        std::unordered_map<int, DeviceBuffer> buffers;
        std::unordered_map<int, Shape> shapes;
        mutable std::mutex buffer_mtx;
        mutable std::mutex shape_mtx;
        MultiDeviceStorage() = default;

        MultiDeviceStorage(
            DistributedTensorConfig strategy_,
            std::vector<int> ordered_device_ids_,
            std::unordered_map<int, DeviceBuffer> buffers_,
            std::unordered_map<int, Shape> shapes_) : strategy(strategy_), ordered_device_ids(ordered_device_ids_), buffers(buffers_), shapes(shapes_) {}

        MultiDeviceStorage(MultiDeviceStorage &&other) {
            std::scoped_lock buf_lock(buffer_mtx, shape_mtx);
            ordered_device_ids = other.ordered_device_ids;
            strategy = other.strategy;
            buffers = other.buffers;
            shapes = other.shapes;
        }
        MultiDeviceStorage(const MultiDeviceStorage &other) {
            std::scoped_lock buf_lock(buffer_mtx, shape_mtx);
            ordered_device_ids = other.ordered_device_ids;
            strategy = other.strategy;
            buffers = other.buffers;
            shapes = other.shapes;
        }

        MultiDeviceStorage &operator=(const MultiDeviceStorage &other) {
            std::scoped_lock buf_lock(buffer_mtx, shape_mtx);
            ordered_device_ids = other.ordered_device_ids;
            strategy = other.strategy;
            buffers = other.buffers;
            shapes = other.shapes;
            return *this;
        }

        MultiDeviceStorage &operator=( MultiDeviceStorage &&other) {
            std::scoped_lock buf_lock(buffer_mtx, shape_mtx);
            ordered_device_ids = other.ordered_device_ids;
            strategy = other.strategy;
            buffers = other.buffers;
            shapes = other.shapes;
            return *this;
        }

        bool operator == (const MultiDeviceStorage& other) {
            return this->ordered_device_ids == other.ordered_device_ids and this->strategy == other.strategy and this->buffers == other.buffers and this->shapes == other.shapes;
        }

        inline const MemoryConfig memory_config() const {
            std::lock_guard<std::mutex> lock(buffer_mtx);
            auto first_device_id = this->ordered_device_ids.at(0);
            if (this->buffers.at(first_device_id).get() == nullptr) {
                TT_THROW("MemoryConfig can only be obtained if the buffer is not null");
            }
            std::optional<ShardSpec> shard_spec = std::nullopt;
            if (is_sharded(this->buffers.at(first_device_id)->buffer_layout())) {
                shard_spec = this->buffers.at(first_device_id)->shard_spec().tensor_shard_spec;
            }
            return MemoryConfig{
                .memory_layout = this->buffers.at(first_device_id)->buffer_layout(),
                .buffer_type = this->buffers.at(first_device_id)->buffer_type(),
                .shard_spec = shard_spec};

        }

        static constexpr auto attribute_names = std::make_tuple();
        const auto attribute_values() const { return std::make_tuple(); }

        // Helper Functions - Getters and setters to get/modify storage attributes. These are needed to
        // preinitialize empty tensor handles and use/populate them in the worker threads.

        inline void insert_buffer_and_shape_for_device(Device* device, const DeviceBuffer buffer, const Shape shape) {
            TT_ASSERT(device == buffer->device(), "Mismatch between device derived from buffer and device derived from MultiDeviceStorage.");
            {
                std::lock_guard<std::mutex> lock(buffer_mtx);
                buffers.insert({device->id(), buffer});
            }
            std::lock_guard<std::mutex> lock(shape_mtx);
            shapes.insert({device->id(), shape});
        }

        inline DeviceBuffer get_buffer_for_device(Device* device) const {
            std::lock_guard<std::mutex> lock(buffer_mtx);
            TT_ASSERT(buffers.find(device->id()) != buffers.end(), "Buffer not found for device " + std::to_string(device->id()));
            TT_ASSERT(buffers.at(device->id())->device() == device, "Mismatch between device derived from buffer and device derived from MultiDeviceStorage.");
            return buffers.at(device->id());
        }

        inline DeviceBuffer& get_buffer_for_device(Device* device) {
            std::lock_guard<std::mutex> lock(buffer_mtx);
            TT_ASSERT(buffers.find(device->id()) != buffers.end(), "Buffer not found for device " + std::to_string(device->id()));
            TT_ASSERT(buffers.at(device->id())->device() == device, "Mismatch between device derived from buffer and device derived from MultiDeviceStorage.");
            return buffers.at(device->id());
        }

        inline DeviceBuffer get_buffer_for_device_id(uint32_t device_id) const {
            std::lock_guard<std::mutex> lock(buffer_mtx);
            return buffers.at(device_id);
        }

        inline Shape get_tensor_shape_for_device(Device* device) const {
            std::lock_guard<std::mutex> lock(shape_mtx);
            TT_ASSERT(shapes.find(device->id()) != shapes.end(), "Shape not found for device " + std::to_string(device->id()));
            return shapes.at(device->id());
        }

        inline uint32_t num_buffers() const {
            std::lock_guard<std::mutex> lock(buffer_mtx);
            return buffers.size();
        }

        inline bool has_buffer_for_device(Device* device) const {
            std::lock_guard<std::mutex> lock(buffer_mtx);
            return buffers.find(device->id()) != buffers.end();
        }

        inline bool has_buffer_for_device_id(uint32_t device_id) const {
            std::lock_guard<std::mutex> lock(buffer_mtx);
            return buffers.find(device_id) != buffers.end();
        }
    };

using Storage = std::variant<OwnedStorage, DeviceStorage, BorrowedStorage, MultiDeviceHostStorage, MultiDeviceStorage>;

template <typename T>
constexpr void raise_unsupported_storage() {
    static_assert(tt::stl::concepts::always_false_v<T>, "Unsupported Storage");
}

inline bool operator==(const Storage &v1, const Storage &v2) {
    return std::visit(
        [](const auto &a, const auto &b) -> bool {
            if constexpr (std::is_same_v<decltype(a), decltype(b)>) {
                return a == b;
            } else {
                return false;
            }
        },
        v1,
        v2);
};

}  // namespace tt_metal

}  // namespace tt

namespace ttnn {
namespace types {

namespace detail {
template <std::size_t Rank>
static tt::tt_metal::Shape compute_ttl_shape(
    const std::array<uint32_t, Rank> &shape, const std::array<std::array<uint32_t, 2>, Rank> &padding) {
    auto ttl_shape = std::array<uint32_t, Rank>{};
    for (auto index = 0; index < Rank; index++) {
        ttl_shape[index] = shape[index] + padding[index][0] + padding[index][1];
    }
    return tt::tt_metal::Shape{
        tt::tt_metal::Shape{ttl_shape}, tt::tt_metal::Padding{padding, tt::tt_metal::Padding::PadValue::Any}};
}

}  // namespace detail

template <std::size_t Rank>
struct RankedShape {
    std::size_t rank;
    tt::tt_metal::Shape value;

    explicit RankedShape(tt::tt_metal::Shape &&shape) : rank{Rank}, value(shape) {}
    explicit RankedShape(const tt::tt_metal::Shape &shape) : rank{Rank}, value(shape) {}

    explicit RankedShape(const std::array<uint32_t, Rank> &shape) : rank{Rank}, value{shape} {}

    explicit RankedShape(
        const std::array<uint32_t, Rank> &shape, const std::array<uint32_t, Rank> &shape_with_tile_padding) :
        rank{Rank}, value{shape, shape_with_tile_padding} {}

    explicit RankedShape(
        const std::array<uint32_t, Rank> &shape, const std::array<std::array<uint32_t, 2>, Rank> &padding) :
        rank{Rank}, value{detail::compute_ttl_shape(shape, padding)} {}

    RankedShape<Rank> with_tile_padding() const {
        return RankedShape{tt::tt_metal::Shape{this->value, tt::tt_metal::Padding{this->value.rank()}}};
    }

    const auto volume() const {
        auto volume = 1;
        for (auto index = 0; index < Rank; index++) {
            volume *= this->value[index];
        }
        return volume;
    }

    bool operator==(const RankedShape<Rank> &other) const {
        const auto &shape_a = this->value;
        const auto &shape_b = other.value;
        // tt::tt_metal::Shape comparison doesn't take padding into account
        return (shape_a == shape_b and shape_a.without_padding() == shape_b.without_padding());
    }

    template <std::size_t OtherRank>
    bool operator==(const RankedShape<OtherRank> &other) const {
        return false;
    }

    const auto operator[](std::int64_t index) const { return this->value.without_padding()[index]; }

    static constexpr auto attribute_names = std::make_tuple("rank", "value");
    const auto attribute_values() const { return std::make_tuple(std::cref(this->rank), std::cref(this->value)); }
};

template <std::size_t Rank>
static std::ostream &operator<<(std::ostream &os, const RankedShape<Rank> &shape) {
    const auto shape_with_tile_padding = shape.with_tile_padding();
    const auto &padding = shape.value.padding();
    os << "ttnn.Shape([";
    for (auto i = 0; i < Rank; ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << shape[i];
        if (padding[i].back > 0) {
            os << "[" << shape_with_tile_padding[i] << "]";
        }
    }
    os << "])";
    return os;
}

struct Shape {
    using RankedShapeVariant = std::variant<
        RankedShape<1>,
        RankedShape<2>,
        RankedShape<3>,
        RankedShape<4>,
        RankedShape<5>,
        RankedShape<6>,
        RankedShape<7>,
        RankedShape<8>>;

    RankedShapeVariant ranked_shape;

   private:
    RankedShapeVariant ttl_shape_to_ttnn_shape(const tt::tt_metal::Shape &shape) {
        switch (shape.rank()) {
            case 1: return RankedShape<1>{shape};
            case 2: return RankedShape<2>{shape};
            case 3: return RankedShape<3>{shape};
            case 4: return RankedShape<4>{shape};
            case 5: return RankedShape<5>{shape};
            case 6: return RankedShape<6>{shape};
            case 7: return RankedShape<7>{shape};
            case 8: return RankedShape<8>{shape};
        };
        TT_THROW("Unsupported rank");
    }

   public:
    explicit Shape(const tt::tt_metal::Shape &shape) : ranked_shape{ttl_shape_to_ttnn_shape(shape)} {}

    template <std::size_t Rank>
    explicit Shape(const RankedShape<Rank> &shape) : ranked_shape{shape} {}

    template <std::size_t Rank>
    explicit Shape(const std::array<uint32_t, Rank> &shape) : ranked_shape{RankedShape<Rank>{shape}} {}

    template <std::size_t Rank>
    explicit Shape(const std::array<uint32_t, Rank> &shape, const std::array<uint32_t, Rank> &shape_tile_padding) :
        ranked_shape{RankedShape<Rank>{shape, shape_tile_padding}} {}

    template <std::size_t Rank>
    explicit Shape(
        const std::array<uint32_t, Rank> &shape, const std::array<std::array<uint32_t, 2>, Rank> &tile_padding) :
        ranked_shape{RankedShape<Rank>{shape, tile_padding}} {}

    const auto rank() const {
        return std::visit(
            []<std::size_t Rank>(const RankedShape<Rank> &shape) -> const auto { return Rank; }, this->ranked_shape);
    }

    const auto size() const { return this->rank(); }

    Shape with_tile_padding() const {
        return std::visit(
            [](const auto &shape) -> Shape { return Shape(shape.with_tile_padding()); }, this->ranked_shape);
    }

    bool operator==(const Shape &other) const {
        return std::visit(
            [](const auto &shape, const auto &other) -> bool { return shape == other; },
            this->ranked_shape,
            other.ranked_shape);
    }

    const auto &value() const {
        return std::visit([](const auto &shape) -> const auto & { return shape.value; }, this->ranked_shape);
    }

    template <std::size_t Rank>
    bool operator==(const std::array<std::uint32_t, Rank> &other) const {
        return Shape{this->value().without_padding()} == Shape{other};
    }

    bool operator!=(const Shape &other) const { return not(*this == other); }

    const auto operator[](std::int64_t index) const {
        return std::visit([index](const auto &shape) -> decltype(auto) { return shape[index]; }, this->ranked_shape);
    }

    const auto volume() const {
        return std::visit([](const auto &shape) -> const auto { return shape.volume(); }, this->ranked_shape);
    }

    template <std::size_t NewRank>
    const Shape to_rank() const {
        return std::visit(
            []<std::size_t Rank>(const RankedShape<Rank> &shape) -> Shape {
                auto shape_with_tile_padding = shape.with_tile_padding();

                std::array<uint32_t, NewRank> new_shape{};
                std::array<uint32_t, NewRank> new_padded_shape{};
                if constexpr (Rank == NewRank) {
                    return Shape(shape);
                } else if constexpr (Rank > NewRank) {
                    auto num_extra_dims = Rank - NewRank;

                    for (auto index = 0; index < num_extra_dims; index++) {
                        TT_ASSERT(shape[index] == 1);
                        TT_ASSERT(shape_with_tile_padding[index] == 1);
                    }

                    for (auto index = 0; index < NewRank; index++) {
                        new_shape[index] = shape[index + num_extra_dims];
                        new_padded_shape[index] = shape_with_tile_padding[index + num_extra_dims];
                    }
                } else {
                    auto num_missing_dims = NewRank - Rank;

                    new_shape.fill(1);
                    new_padded_shape.fill(1);

                    for (auto index = 0; index < Rank; index++) {
                        new_shape[index + num_missing_dims] = shape[index];
                        new_padded_shape[index + num_missing_dims] = shape_with_tile_padding[index];
                    }
                }
                return Shape(RankedShape<NewRank>(new_shape, new_padded_shape));
            },
            this->ranked_shape);
    }

    static constexpr auto attribute_names = std::make_tuple("ranked_shape");
    const auto attribute_values() const { return std::make_tuple(std::cref(this->ranked_shape)); }
};

static std::ostream &operator<<(std::ostream &os, const Shape &self) {
    std::visit([&os](const auto &shape) { os << shape; }, self.ranked_shape);
    return os;
}


struct TensorSchema {
    const std::size_t min_rank;
    const std::size_t max_rank;
    const std::set<DataType> dtypes;
    const std::set<Layout> layouts;
    const bool can_be_on_device;
    const bool can_be_on_cpu;
    const bool can_be_scalar;
    const bool is_optional;

    static constexpr auto attribute_names() {
        return std::make_tuple(
            "min_rank",
            "max_rank",
            "dtypes",
            "layouts",
            "can_be_on_device",
            "can_be_on_cpu",
            "can_be_scalar",
            "is_optional");
    }

    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->min_rank),
            std::cref(this->max_rank),
            std::cref(this->dtypes),
            std::cref(this->layouts),
            std::cref(this->can_be_on_device),
            std::cref(this->can_be_on_cpu),
            std::cref(this->can_be_scalar),
            std::cref(this->is_optional));
    }
};

}  // namespace types

using types::Shape;
using types::TensorSchema;

}  // namespace ttnn
