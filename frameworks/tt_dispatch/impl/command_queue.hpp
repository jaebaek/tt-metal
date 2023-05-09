#include <future>
#include <memory>
#include <thread>

#include "frameworks/tt_dispatch/impl/thread_safe_queue.hpp"
#include "tt_metal/common/base.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::tt_metal;
using std::thread;
using std::unique_ptr;

enum class CommandType { ENQUEUE_READ_BUFFER, ENQUEUE_WRITE_BUFFER, ENQUEUE_LAUNCH, FLUSH, FINISH };

class Command {
    CommandType type;

   public:
    Command();
    virtual void handle();
};

class EnqueueReadBufferCommand : public Command {
   private:
    Device* device;
    DramBuffer* buffer;

   public:
    static constexpr CommandType type = CommandType::ENQUEUE_READ_BUFFER;

    EnqueueReadBufferCommand(Device* device, DramBuffer* buffer) {
        this->device = device;
        this->buffer = buffer;
    }

    void handle() {}
};

class EnqueueWriteBufferCommand : public Command {
   private:
    Device* device;
    DramBuffer* buffer;

   public:
    static constexpr CommandType type = CommandType::ENQUEUE_WRITE_BUFFER;

    EnqueueWriteBufferCommand(Device* device, DramBuffer* buffer) {
        this->device = device;
        this->buffer = buffer;
    }

    void handle() {}
};

class EnqueueLaunchCommand : public Command {
   private:
    Device* device;
    Program* program;

   public:
    static constexpr CommandType type = CommandType::ENQUEUE_LAUNCH;

    EnqueueLaunchCommand(Device* device, Program* program) {
        this->device = device;
        this->program = program;
    }

    void handle() {}
};

class CommandQueue {
   public:
    CommandQueue() {
        auto worker_logic = [this]() {
            while (true) {       // Worker thread keeps on flushing
                this->internal_queue.peek()
                    ->handle();  // Only responsible for ensuring that command enqueued onto device... needs to be
                                 // handled prior to popping for 'flush' semantics to work
                this->internal_queue.pop();
            }
        };

        thread(worker_logic).detach();
    }

    ~CommandQueue() { this->finish(); }

   private:
    void enqueue_command(Command& command, bool blocking) {
        unique_ptr<Command> p = std::make_unique<Command>(&command);

        this->internal_queue.push(std::move(p));

        if (blocking) {
            this->finish();
        }
    }

    TSQueue<unique_ptr<Command>> internal_queue;
    void enqueue_read_buffer(Device* device, DramBuffer* buffer, void* dst, bool blocking) {
        EnqueueReadBufferCommand command(device, buffer);
        this->enqueue_command(command, blocking);
    }

    void enqueue_write_buffer(Device* device, DramBuffer* buffer, void* src, bool blocking) {
        EnqueueWriteBufferCommand command(device, buffer);
        this->enqueue_command(command, blocking);
    }

    void enqueue_launch(Device* device, Program* program, bool blocking) {
        EnqueueLaunchCommand command(device, program);
        this->enqueue_command(command, blocking);
    }

    void flush() {
        while (this->internal_queue.size() > 0)
            ;  // Wait until all commands have been enqueued on device
    }

    void finish() { TT_THROW("CommandQueue.finish not yet implemented"); }

    friend void EnqueueReadBuffer(Device* device, CommandQueue& cq, DramBuffer* buffer, void* dst, bool blocking);
    friend void EnqueueWriteBuffer();
    friend void Launch();
    friend void Flush(CommandQueue& cq);
    friend void Finish(CommandQueue& cq);
};

void EnqueueReadBuffer(Device* device, CommandQueue& cq, DramBuffer* buffer, void* dst, bool blocking) {
    cq.enqueue_read_buffer(device, buffer, dst, blocking);
}

void EnqueueWriteBuffer() { TT_THROW("EnqueueWriteBuffer not yet implemented"); }

void EnqueueLaunch() { TT_THROW("EnqueueLaunch not yet implemented"); }

void Flush(CommandQueue& cq) { cq.flush(); }

void Finish(CommandQueue& cq) { cq.finish(); }
