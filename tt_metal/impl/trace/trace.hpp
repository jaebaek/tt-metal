// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <variant>

#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/impl/dispatch/lock_free_queue.hpp"
#include "tt_metal/impl/program/program.hpp"

namespace tt::tt_metal {

using std::shared_ptr;
using std::unique_ptr;
using std::unordered_map;
using std::vector;

class CommandQueue;

namespace detail {
struct ReadBufferDescriptor;
struct ReadEventDescriptor;
typedef LockFreeQueue<std::variant<ReadBufferDescriptor, ReadEventDescriptor>> CompletionReaderQueue;

struct TraceDescriptor {
    uint32_t num_events;
    std::optional<uint32_t> initial_event_id;
    CompletionReaderQueue issued_completion_q_reads;

    TraceDescriptor() { this->reset(); }

    void reset() {
        this->num_events = 0;
        this->initial_event_id.reset();
        this->issued_completion_q_reads.clear();
    }

    // Calculate relative offset to the initial event ID of the trace
    uint32_t trace_event_id(uint32_t event_id) {
        if (not this->initial_event_id.has_value()) {
            initial_event_id = event_id;
        }
        TT_FATAL(event_id >= initial_event_id.value(), "Traced event ID must be greater or equal to initial event ID");
        return event_id - initial_event_id.value();
    }
};
}

struct TraceBuffer {
    shared_ptr<detail::TraceDescriptor> desc;
    shared_ptr<Buffer> buffer;
};

enum class TraceState {
    EMPTY,
    CAPTURING,
    CAPTURED,
    INSTANTIATING,
    READY
};

class Trace {
   private:
    friend class EnqueueProgramCommand;
    friend void EnqueueTrace(CommandQueue& cq, uint32_t tid, bool blocking);

    TraceState state;

    // Trace queue used to capture commands
    unique_ptr<CommandQueue> tq;

    // Trace instance id to buffer mapping mananged via instantiate and release calls
    // a static map keeps trace buffers alive until explicitly released by the user
    static unordered_map<uint32_t, TraceBuffer> buffer_pool;

    // Thread safe accessor to trace::buffer_pool
    static std::mutex pool_mutex;
    template <typename Func>
    static inline auto _safe_pool(Func func) {
        std::lock_guard<std::mutex> lock(Trace::pool_mutex);
        return func();
    }

    static uint32_t next_id();

   public :
    Trace();
    ~Trace() {
        TT_FATAL(this->state != TraceState::CAPTURING, "Trace capture incomplete before destruction!");
        TT_FATAL(this->state != TraceState::INSTANTIATING, "Trace instantiation incomplete before destruction!");
    }

    // Return the captured trace queue
    CommandQueue& queue() const { return *tq; };

    // Stages a trace buffer into device DRAM via the CQ passed in and returns a unique trace id
    uint32_t instantiate(CommandQueue& tq);

    // Trace capture, validation, and query methods
    void begin_capture();
    void end_capture();
    void validate();

    // Thread-safe accessors to manage trace instances
    static bool has_instance(const uint32_t tid);
    static void add_instance(const uint32_t tid, TraceBuffer buf);
    static void remove_instance(const uint32_t tid);
    static void release_all();  // note all instances across all devices are released
    static TraceBuffer get_instance(const uint32_t tid);
};

}  // namespace tt::tt_metal
