#pragma once

#include "infinicore/device.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <optional>
#include <vector>

namespace infinilm::backends::infiniops {

struct GraphTaskUpdate;
using GraphTaskUpdates = std::vector<std::shared_ptr<GraphTaskUpdate>>;

bool should_use(const infinicore::Device &device);

void begin_graph_task_capture();

GraphTaskUpdates end_graph_task_capture();

void update_graph_tasks(const GraphTaskUpdates &updates);

infinicore::Tensor embedding(const infinicore::Tensor &input_ids,
                             const infinicore::Tensor &weight);

infinicore::Tensor linear(const infinicore::Tensor &input,
                          const infinicore::Tensor &weight,
                          std::optional<infinicore::Tensor> bias);

void rms_norm_(const infinicore::Tensor &out,
               const infinicore::Tensor &input,
               const infinicore::Tensor &weight,
               float eps);

void add_rms_norm_(infinicore::Tensor &input,
                   infinicore::Tensor &residual,
                   const infinicore::Tensor &weight,
                   float eps);

infinicore::Tensor swiglu(const infinicore::Tensor &up,
                          const infinicore::Tensor &gate);

void reshape_and_cache(const infinicore::Tensor &key,
                       const infinicore::Tensor &value,
                       const infinicore::Tensor &kv_cache,
                       const infinicore::Tensor &slot_mapping);

void rotary_embedding(const infinicore::Tensor &positions,
                      const infinicore::Tensor &query,
                      const infinicore::Tensor &key,
                      const infinicore::Tensor &cos_sin_cache,
                      int64_t head_size,
                      std::optional<infinicore::Tensor> query_out = std::nullopt,
                      std::optional<infinicore::Tensor> key_out = std::nullopt);

void mha_varlen_fwd(const infinicore::Tensor &out,
                    const infinicore::Tensor &q,
                    const infinicore::Tensor &k,
                    const infinicore::Tensor &v,
                    const infinicore::Tensor &cu_seqlens_q,
                    const infinicore::Tensor &cu_seqlens_k,
                    const infinicore::Tensor &block_table,
                    int64_t max_seqlen_q,
                    int64_t max_seqlen_k,
                    float softmax_scale);

void mha_fwd_kvcache(const infinicore::Tensor &out,
                     const infinicore::Tensor &q,
                     const infinicore::Tensor &kcache,
                     const infinicore::Tensor &vcache,
                     const infinicore::Tensor &seqlens_k,
                     const infinicore::Tensor &block_table,
                     float softmax_scale);

void paged_attention(const infinicore::Tensor &out,
                     const infinicore::Tensor &q,
                     const infinicore::Tensor &kcache,
                     const infinicore::Tensor &vcache,
                     const infinicore::Tensor &seqlens_k,
                     const infinicore::Tensor &block_table,
                     int64_t num_heads,
                     int64_t num_kv_heads,
                     int64_t head_size,
                     float softmax_scale,
                     std::optional<infinicore::Tensor> seqlens_k_host = std::nullopt,
                     std::optional<infinicore::Tensor> block_table_host = std::nullopt);

infinicore::Tensor sample_from_logits(const infinicore::Tensor &logits,
                                      const infinicore::Tensor &input_offsets,
                                      float temperature,
                                      int top_k,
                                      float top_p);

class Embedding : public infinicore::nn::Embedding {
public:
    using infinicore::nn::Embedding::Embedding;

    infinicore::Tensor forward(const infinicore::Tensor &indices) const;
};

class RMSNorm : public infinicore::nn::RMSNorm {
public:
    using infinicore::nn::RMSNorm::RMSNorm;

    infinicore::Tensor forward(const infinicore::Tensor &x) const;
    void forward_inplace(infinicore::Tensor &x, infinicore::Tensor &residual) const;
};

class ReplicatedLinear : public infinicore::nn::Linear {
public:
    using infinicore::nn::Linear::Linear;

    infinicore::Tensor forward(infinicore::Tensor &input) const;
};

class ColumnParallelLinear : public infinicore::nn::ColumnParallelLinear {
public:
    using infinicore::nn::ColumnParallelLinear::ColumnParallelLinear;

    infinicore::Tensor forward(infinicore::Tensor &input) const;
};

class RowParallelLinear : public infinicore::nn::RowParallelLinear {
public:
    using infinicore::nn::RowParallelLinear::RowParallelLinear;

    infinicore::Tensor forward(infinicore::Tensor &input) const;
};

} // namespace infinilm::backends::infiniops
