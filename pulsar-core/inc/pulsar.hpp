#pragma once

#include "pulsar/ops.hpp"
#include "pulsar/profiling.hpp"
#include "pulsar/rope.hpp"

#include "pulsar/runtime/kv/active_buffer.hpp"
#include "pulsar/runtime/kv/file_bucket.hpp"
#include "pulsar/runtime/kv/kv_memory.hpp"
#include "pulsar/runtime/kv/paged_bucket.hpp"
#include "pulsar/runtime/kv/paged_pool.hpp"
#include "pulsar/runtime/kv/page_view.hpp"
#include "pulsar/runtime/kv/recall.hpp"

#include "pulsar/model/compiled_model.hpp"
#include "pulsar/model/interface.hpp"
#include "pulsar/model/paged_forward.hpp"
#include "pulsar/model/qwen2.hpp"
#include "pulsar/model/qwen3.hpp"

#include "pulsar/runtime/engine.hpp"
#include "pulsar/runtime/sampler.hpp"
#include "pulsar/runtime/scheduler.hpp"

// Umbrella header for the engine's public C++ API: KV-cache primitives (kv/),
// model definitions (model/), the serving engine (runtime/), and the op
// declarations, RoPE math and profiling macros they're built on. Excludes
// kernels/: those are .cuh, device code compiled only from .cu translation
// units, not headers a host-side engine consumer includes.
