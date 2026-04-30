#include "infiniops.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef INFINILM_ENABLE_INFINIOPS
#include "ascend/add_rms_norm/kernel.h"
#include "ascend/device_.h"
#include "ascend/embedding/kernel.h"
#include "ascend/linear/kernel.h"
#include "ascend/mha_fwd_kvcache/kernel.h"
#include "ascend/mha_varlen_fwd/kernel.h"
#include "ascend/mul/kernel.h"
#include "ascend/reshape_and_cache/kernel.h"
#include "ascend/rms_norm/kernel.h"
#include "ascend/rotary_embedding/kernel.h"
#include "ascend/swiglu/kernel.h"
#include "ascend/top_k_top_p_sampler/kernel_atb.h"
#endif

namespace infinilm::backends::infiniops {
namespace {

#ifdef INFINILM_ENABLE_INFINIOPS
infini::ops::DataType to_dtype(infinicore::DataType dtype) {
    switch (dtype) {
    case infinicore::DataType::I8:
        return infini::ops::DataType::kInt8;
    case infinicore::DataType::I16:
        return infini::ops::DataType::kInt16;
    case infinicore::DataType::I32:
        return infini::ops::DataType::kInt32;
    case infinicore::DataType::I64:
        return infini::ops::DataType::kInt64;
    case infinicore::DataType::U8:
        return infini::ops::DataType::kUInt8;
    case infinicore::DataType::U16:
        return infini::ops::DataType::kUInt16;
    case infinicore::DataType::U32:
        return infini::ops::DataType::kUInt32;
    case infinicore::DataType::U64:
        return infini::ops::DataType::kUInt64;
    case infinicore::DataType::F16:
        return infini::ops::DataType::kFloat16;
    case infinicore::DataType::BF16:
        return infini::ops::DataType::kBFloat16;
    case infinicore::DataType::F32:
        return infini::ops::DataType::kFloat32;
    case infinicore::DataType::F64:
        return infini::ops::DataType::kFloat64;
    default:
        throw std::runtime_error("InfiniOps adapter: unsupported dtype");
    }
}

infini::ops::Device to_device(const infinicore::Device &device) {
    switch (device.getType()) {
    case infinicore::Device::Type::CPU:
        return {infini::ops::Device::Type::kCpu, static_cast<int>(device.getIndex())};
    case infinicore::Device::Type::ASCEND:
        return {infini::ops::Device::Type::kAscend, static_cast<int>(device.getIndex())};
    default:
        throw std::runtime_error("InfiniOps adapter: unsupported device");
    }
}

infini::ops::Tensor as_tensor(const infinicore::Tensor &tensor) {
    std::vector<std::size_t> shape(tensor->shape().begin(), tensor->shape().end());
    std::vector<std::ptrdiff_t> strides(tensor->strides().begin(), tensor->strides().end());
    return {
        const_cast<std::byte *>(tensor->data()),
        shape,
        to_dtype(tensor->dtype()),
        to_device(tensor->device()),
        strides,
    };
}

std::optional<infini::ops::Tensor> as_optional_tensor(const std::optional<infinicore::Tensor> &tensor) {
    if (!tensor.has_value() || !tensor.value()) {
        return std::nullopt;
    }
    return as_tensor(tensor.value());
}

infini::ops::Handle handle() {
    infini::ops::Handle h;
    h.set_stream(infinicore::context::getStream());
    return h;
}

infini::ops::Config config(std::size_t implementation_index = 0) {
    infini::ops::Config cfg;
    cfg.set_implementation_index(implementation_index);
    return cfg;
}

bool unquantized(const infinicore::nn::BaseLinear &linear) {
    return linear.get_quantization()->get_quant_scheme()
        == infinicore::quantization::QuantScheme::NONE;
}

std::uint16_t f32_to_f16_bits(float value) {
    std::uint32_t f32;
    std::memcpy(&f32, &value, sizeof(f32));
    const std::uint16_t sign = (f32 >> 16) & 0x8000;
    const std::int32_t exponent = static_cast<std::int32_t>((f32 >> 23) & 0xFF) - 127;
    std::uint32_t mantissa = f32 & 0x7FFFFF;

    if (exponent >= 16) {
        if (exponent == 128 && mantissa != 0) {
            return static_cast<std::uint16_t>(sign | 0x7E00);
        }
        return static_cast<std::uint16_t>(sign | 0x7C00);
    }
    if (exponent >= -14) {
        return static_cast<std::uint16_t>(
            sign | ((exponent + 15) << 10) | (mantissa >> 13));
    }
    if (exponent >= -24) {
        mantissa |= 0x800000;
        mantissa >>= (-14 - exponent);
        return static_cast<std::uint16_t>(sign | (mantissa >> 13));
    }
    return sign;
}

std::uint16_t f32_to_bf16_bits(float value) {
    std::uint32_t bits32;
    std::memcpy(&bits32, &value, sizeof(bits32));
    const std::uint32_t rounding_bias = 0x00007FFF + ((bits32 >> 16) & 1);
    return static_cast<std::uint16_t>((bits32 + rounding_bias) >> 16);
}

infinicore::Tensor require_contiguous(const infinicore::Tensor &tensor,
                                      const char *name) {
    if (!tensor->is_contiguous()) {
        throw std::runtime_error(std::string("InfiniOps adapter: ") + name + " must be contiguous");
    }
    return tensor;
}

int64_t max_sequence_length(const infinicore::Tensor &cu_seqlens) {
    if (cu_seqlens->dtype() != infinicore::DataType::I32) {
        throw std::runtime_error("InfiniOps adapter: cu_seqlens must be int32");
    }
    require_contiguous(cu_seqlens, "cu_seqlens");

    const auto source_device = cu_seqlens->device();
    auto host = source_device.getType() == infinicore::Device::Type::CPU
                  ? cu_seqlens
                  : cu_seqlens->to(infinicore::Device::cpu());
    const auto *data = reinterpret_cast<const int32_t *>(host->data());
    int64_t max_len = 0;
    for (std::size_t i = 0; i + 1 < host->size(0); ++i) {
        max_len = std::max<int64_t>(max_len, data[i + 1] - data[i]);
    }
    if (source_device.getType() != infinicore::Device::Type::CPU) {
        infinicore::context::setDevice(source_device);
    }
    return max_len;
}

infinicore::Tensor scalar_tensor(infinicore::DataType dtype,
                                 const infinicore::Device &device,
                                 float value) {
    auto out = infinicore::Tensor::empty({1}, dtype, device);
    auto cpu = infinicore::Device::cpu();
    if (dtype == infinicore::DataType::F16) {
        auto host_value = f32_to_f16_bits(value);
        auto host = infinicore::Tensor::from_blob(&host_value, {1}, dtype, cpu);
        out->copy_from(host);
    } else if (dtype == infinicore::DataType::BF16) {
        auto host_value = f32_to_bf16_bits(value);
        auto host = infinicore::Tensor::from_blob(&host_value, {1}, dtype, cpu);
        out->copy_from(host);
    } else if (dtype == infinicore::DataType::F32) {
        auto host = infinicore::Tensor::from_blob(&value, {1}, dtype, cpu);
        out->copy_from(host);
    } else {
        throw std::runtime_error("InfiniOps adapter: unsupported scalar dtype");
    }
    return out;
}
#endif

void require_enabled() {
#ifndef INFINILM_ENABLE_INFINIOPS
    throw std::runtime_error("InfiniOps adapter was not enabled at build time");
#endif
}

} // namespace

bool should_use(const infinicore::Device &device) {
#ifdef INFINILM_ENABLE_INFINIOPS
    return device.getType() == infinicore::Device::Type::ASCEND;
#else
    (void)device;
    return false;
#endif
}

infinicore::Tensor embedding(const infinicore::Tensor &input_ids,
                             const infinicore::Tensor &weight) {
#ifdef INFINILM_ENABLE_INFINIOPS
    require_contiguous(input_ids, "embedding input_ids");
    require_contiguous(weight, "embedding weight");
    std::vector<std::size_t> out_shape(input_ids->shape().begin(), input_ids->shape().end());
    out_shape.push_back(weight->size(1));
    auto out = infinicore::Tensor::empty(out_shape, weight->dtype(), weight->device());
    infini::ops::Embedding::Call(handle(), config(), as_tensor(input_ids), as_tensor(weight), as_tensor(out));
    return out;
#else
    require_enabled();
    return {};
#endif
}

infinicore::Tensor linear(const infinicore::Tensor &input,
                          const infinicore::Tensor &weight,
                          std::optional<infinicore::Tensor> bias) {
#ifdef INFINILM_ENABLE_INFINIOPS
    auto input_contiguous = require_contiguous(input, "linear input");
    auto weight_contiguous = require_contiguous(weight, "linear weight");
    if (bias.has_value()) {
        require_contiguous(bias.value(), "linear bias");
    }
    std::vector<std::size_t> out_shape(input_contiguous->shape().begin(), input_contiguous->shape().end());
    out_shape.back() = weight_contiguous->size(weight_contiguous->ndim() - 2);
    auto out = infinicore::Tensor::empty(out_shape, input_contiguous->dtype(), input_contiguous->device());
    infini::ops::Linear::Call(
        handle(), config(), as_tensor(input_contiguous), as_tensor(weight_contiguous),
        as_optional_tensor(bias), as_tensor(out));
    return out;
#else
    require_enabled();
    return {};
#endif
}

void rms_norm_(const infinicore::Tensor &out,
               const infinicore::Tensor &input,
               const infinicore::Tensor &weight,
               float eps) {
#ifdef INFINILM_ENABLE_INFINIOPS
    infini::ops::RmsNorm::Call(handle(), config(), as_tensor(input), as_tensor(weight), eps, as_tensor(out));
#else
    (void)out;
    (void)input;
    (void)weight;
    (void)eps;
    require_enabled();
#endif
}

void add_rms_norm_(infinicore::Tensor &input,
                   infinicore::Tensor &residual,
                   const infinicore::Tensor &weight,
                   float eps) {
#ifdef INFINILM_ENABLE_INFINIOPS
    auto residual_out = infinicore::Tensor::empty(residual->shape(), residual->dtype(), residual->device());
    infini::ops::AddRmsNorm::Call(
        handle(), config(), as_tensor(input), as_tensor(residual), as_tensor(weight), eps,
        as_tensor(input), as_tensor(residual_out));
    residual = residual_out;
#else
    (void)input;
    (void)residual;
    (void)weight;
    (void)eps;
    require_enabled();
#endif
}

infinicore::Tensor swiglu(const infinicore::Tensor &up,
                          const infinicore::Tensor &gate) {
#ifdef INFINILM_ENABLE_INFINIOPS
    auto out = infinicore::Tensor::empty(up->shape(), up->dtype(), up->device());
    infini::ops::Swiglu::Call(handle(), config(), as_tensor(up), as_tensor(gate), as_tensor(out));
    return out;
#else
    require_enabled();
    return {};
#endif
}

void reshape_and_cache(const infinicore::Tensor &key,
                       const infinicore::Tensor &value,
                       const infinicore::Tensor &kv_cache,
                       const infinicore::Tensor &slot_mapping) {
#ifdef INFINILM_ENABLE_INFINIOPS
    infini::ops::ReshapeAndCache::Call(
        handle(), config(), as_tensor(key), as_tensor(value), as_tensor(kv_cache),
        as_tensor(slot_mapping), as_tensor(kv_cache));
#else
    (void)key;
    (void)value;
    (void)kv_cache;
    (void)slot_mapping;
    require_enabled();
#endif
}

void rotary_embedding(const infinicore::Tensor &positions,
                      const infinicore::Tensor &query,
                      const infinicore::Tensor &key,
                      const infinicore::Tensor &cos_sin_cache,
                      int64_t head_size,
                      std::optional<infinicore::Tensor> query_out,
                      std::optional<infinicore::Tensor> key_out) {
#ifdef INFINILM_ENABLE_INFINIOPS
    std::optional<infini::ops::Tensor> key_opt = as_tensor(key);
    std::optional<infini::ops::Tensor> query_out_opt = as_optional_tensor(query_out);
    std::optional<infini::ops::Tensor> key_out_opt = as_optional_tensor(key_out);

    infini::ops::RotaryEmbedding::Call(
        handle(), config(), as_tensor(positions), as_tensor(query), key_opt,
        head_size, as_tensor(cos_sin_cache), true, head_size,
        query_out_opt, key_out_opt, false);
#else
    (void)positions;
    (void)query;
    (void)key;
    (void)cos_sin_cache;
    (void)head_size;
    (void)query_out;
    (void)key_out;
    require_enabled();
#endif
}

void mha_varlen_fwd(const infinicore::Tensor &out,
                    const infinicore::Tensor &q,
                    const infinicore::Tensor &k,
                    const infinicore::Tensor &v,
                    const infinicore::Tensor &cu_seqlens_q,
                    const infinicore::Tensor &cu_seqlens_k,
                    const infinicore::Tensor &block_table,
                    int64_t max_seqlen_q,
                    int64_t max_seqlen_k,
                    float softmax_scale) {
#ifdef INFINILM_ENABLE_INFINIOPS
    (void)max_seqlen_q;
    (void)max_seqlen_k;
    std::optional<infini::ops::Tensor> out_opt = as_tensor(out);
    std::optional<infini::ops::Tensor> block_table_opt = as_tensor(block_table);
    std::optional<infini::ops::Tensor> no_tensor;
    std::optional<int64_t> no_generator;
    const auto actual_max_seqlen_q = max_sequence_length(cu_seqlens_q);
    const auto actual_max_seqlen_k = max_sequence_length(cu_seqlens_k);
    infini::ops::MhaVarlenFwd::Call(
        handle(), config(), as_tensor(q), as_tensor(k), as_tensor(v), out_opt,
        as_tensor(cu_seqlens_q), as_tensor(cu_seqlens_k), no_tensor,
        no_tensor, block_table_opt, no_tensor, actual_max_seqlen_q, actual_max_seqlen_k,
        0.0f, softmax_scale, false, true, -1, 0, 0.0f, false, no_generator, 0);
#else
    (void)out;
    (void)q;
    (void)k;
    (void)v;
    (void)cu_seqlens_q;
    (void)cu_seqlens_k;
    (void)block_table;
    (void)max_seqlen_q;
    (void)max_seqlen_k;
    (void)softmax_scale;
    require_enabled();
#endif
}

void mha_fwd_kvcache(const infinicore::Tensor &out,
                     const infinicore::Tensor &q,
                     const infinicore::Tensor &kcache,
                     const infinicore::Tensor &vcache,
                     const infinicore::Tensor &seqlens_k,
                     const infinicore::Tensor &block_table,
                     float softmax_scale) {
#ifdef INFINILM_ENABLE_INFINIOPS
    std::optional<infini::ops::Tensor> seqlens_k_opt = as_tensor(seqlens_k);
    std::optional<infini::ops::Tensor> block_table_opt = as_tensor(block_table);
    std::optional<infini::ops::Tensor> out_opt = as_tensor(out);
    std::optional<infini::ops::Tensor> no_tensor;
    infini::ops::MhaFwdKvcache::Call(
        handle(), config(), as_tensor(q), as_tensor(kcache), as_tensor(vcache),
        no_tensor, no_tensor, seqlens_k_opt, no_tensor, no_tensor,
        no_tensor, no_tensor, block_table_opt, no_tensor, out_opt,
        softmax_scale, true, -1, 0, 0.0f, false, 0);
#else
    (void)out;
    (void)q;
    (void)kcache;
    (void)vcache;
    (void)seqlens_k;
    (void)block_table;
    (void)softmax_scale;
    require_enabled();
#endif
}

infinicore::Tensor sample_from_logits(const infinicore::Tensor &logits,
                                      const infinicore::Tensor &input_offsets,
                                      float temperature,
                                      int top_k,
                                      float top_p) {
#ifdef INFINILM_ENABLE_INFINIOPS
    if (!should_use(logits->device())) {
        throw std::runtime_error("InfiniOps adapter: logits sampling is only implemented for Ascend");
    }
    if (temperature <= 0.0f) {
        throw std::runtime_error("InfiniOps adapter: sampling temperature must be positive");
    }
    if (logits->dtype() != infinicore::DataType::F16 && logits->dtype() != infinicore::DataType::BF16) {
        throw std::runtime_error("InfiniOps adapter: Ascend sampling requires float16 or bfloat16 logits");
    }
    if (logits->ndim() != 3) {
        throw std::runtime_error("InfiniOps adapter: logits must be 3D [batch, seq, vocab]");
    }

    auto offsets_cpu = input_offsets->device().getType() == infinicore::Device::Type::CPU
                         ? input_offsets
                         : input_offsets->to(infinicore::Device::cpu());
    offsets_cpu = offsets_cpu->is_contiguous() ? offsets_cpu : offsets_cpu->contiguous();
    if (offsets_cpu->dtype() != infinicore::DataType::I32) {
        throw std::runtime_error("InfiniOps adapter: input_offsets must be int32");
    }

    const auto batch_size = logits->size(0);
    const auto total_len = logits->size(1);
    const auto vocab_size = logits->size(2);
    const auto n_req = offsets_cpu->size(0) - 1;
    const auto effective_top_k = top_k <= 0 ? static_cast<int64_t>(vocab_size)
                                            : std::min<int64_t>(top_k, static_cast<int64_t>(vocab_size));
    const double scale = 1.0 / static_cast<double>(temperature);
    const bool needs_scale = std::abs(scale - 1.0) > 1e-6;
    std::optional<infinicore::Tensor> scale_value;
    if (needs_scale) {
        scale_value = scalar_tensor(logits->dtype(), logits->device(), static_cast<float>(scale));
    }

    int64_t top_k_value = effective_top_k;
    float top_p_value = top_p;
    std::optional<infini::ops::Tensor> top_k_tensor;
    std::optional<infini::ops::Tensor> top_p_tensor;
    if (top_k > 0) {
        top_k_tensor.emplace(
            &top_k_value, std::vector<std::size_t>{1},
            infini::ops::DataType::kInt64,
            infini::ops::Device{infini::ops::Device::Type::kCpu});
    }
    if (top_p < 1.0f) {
        top_p_tensor.emplace(
            &top_p_value, std::vector<std::size_t>{1},
            infini::ops::DataType::kFloat32,
            infini::ops::Device{infini::ops::Device::Type::kCpu});
    }

    auto logits_2d = logits->view({batch_size * total_len, vocab_size});
    auto sampled_i32 = infinicore::Tensor::empty({n_req}, infinicore::DataType::I32, logits->device());
    const auto *offsets = reinterpret_cast<const int32_t *>(offsets_cpu->data());

    for (std::size_t i = 0; i < n_req; ++i) {
        const auto row = offsets[i + 1] - 1;
        if (row < 0 || static_cast<size_t>(row) >= batch_size * total_len) {
            throw std::runtime_error("InfiniOps adapter: input_offsets contains an invalid row");
        }

        auto score = logits_2d->narrow({{0, static_cast<size_t>(row), 1}});
        require_contiguous(score, "sampling logits row");
        auto sampler_logits = score;
        if (scale_value.has_value()) {
            auto scaled_score = infinicore::Tensor::empty(score->shape(), score->dtype(), score->device());
            infini::ops::Mul::Call(handle(), config(), as_tensor(score), as_tensor(scale_value.value()), as_tensor(scaled_score));
            sampler_logits = scaled_score;
        }

        auto out = sampled_i32->narrow({{0, i, 1}});
        infini::ops::TopKTopPSampler::Call(
            handle(), config(), as_tensor(sampler_logits), top_k_tensor,
            top_p_tensor, as_tensor(out));
    }

    auto sampled_i32_cpu = sampled_i32->to(infinicore::Device::cpu());
    infinicore::context::syncStream();

    auto sampled_i64_cpu = infinicore::Tensor::empty({n_req}, infinicore::DataType::I64, infinicore::Device::cpu());
    const auto *src = reinterpret_cast<const int32_t *>(sampled_i32_cpu->data());
    auto *dst = reinterpret_cast<int64_t *>(sampled_i64_cpu->data());
    for (std::size_t i = 0; i < n_req; ++i) {
        dst[i] = static_cast<int64_t>(src[i]);
    }

    infinicore::context::setDevice(logits->device());
    return sampled_i64_cpu;
#else
    (void)logits;
    (void)input_offsets;
    (void)temperature;
    (void)top_k;
    (void)top_p;
    require_enabled();
    return {};
#endif
}

infinicore::Tensor Embedding::forward(const infinicore::Tensor &indices) const {
    if (should_use(weight()->device())) {
        return embedding(indices, weight());
    }
    return infinicore::nn::Embedding::forward(indices);
}

infinicore::Tensor RMSNorm::forward(const infinicore::Tensor &x) const {
    if (should_use(x->device())) {
        auto out = infinicore::Tensor::empty(x->shape(), x->dtype(), x->device());
        rms_norm_(out, x, weight(), static_cast<float>(eps()));
        return out;
    }
    return infinicore::nn::RMSNorm::forward(x);
}

void RMSNorm::forward_inplace(infinicore::Tensor &x, infinicore::Tensor &residual) const {
    if (!should_use(x->device())) {
        infinicore::nn::RMSNorm::forward_inplace(x, residual);
        return;
    }

    if (!residual) {
        residual = x;
        x = forward(x);
        return;
    }

    add_rms_norm_(x, residual, weight(), static_cast<float>(eps()));
}

infinicore::Tensor ReplicatedLinear::forward(infinicore::Tensor &input) const {
#ifdef INFINILM_ENABLE_INFINIOPS
    if (should_use(input->device()) && unquantized(*this)) {
        std::optional<infinicore::Tensor> bias_opt = has_bias() ? std::make_optional(bias()) : std::nullopt;
        return linear(input, weight(), bias_opt);
    }
#endif
    return infinicore::nn::Linear::forward(input);
}

infinicore::Tensor ColumnParallelLinear::forward(infinicore::Tensor &input) const {
#ifdef INFINILM_ENABLE_INFINIOPS
    if (should_use(input->device()) && unquantized(*this)) {
        std::optional<infinicore::Tensor> bias_opt = has_bias() ? std::make_optional(bias()) : std::nullopt;
        return linear(input, weight(), bias_opt);
    }
#endif
    return infinicore::nn::ColumnParallelLinear::forward(input);
}

infinicore::Tensor RowParallelLinear::forward(infinicore::Tensor &input) const {
#ifdef INFINILM_ENABLE_INFINIOPS
    if (should_use(input->device()) && unquantized(*this)) {
        std::optional<infinicore::Tensor> bias_opt = has_bias() ? std::make_optional(bias()) : std::nullopt;
        auto output = linear(input, weight(), bias_opt);
        if ((tp_size_ > 1) && (communicator_ != nullptr)) {
            infinicore::op::distributed::allreduce_(output, output, INFINICCL_SUM, communicator_);
        }
        return output;
    }
#endif
    return infinicore::nn::RowParallelLinear::forward(input);
}

} // namespace infinilm::backends::infiniops
