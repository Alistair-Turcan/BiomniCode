#include <ATen/CPUGeneratorImpl.h>
#include <torch/library.h>
#include <algorithm>
#include <cstdint>
#include <mutex>

// Default CPU float uniform consumes one engine word per element. Advance the
// actual PyTorch engine, omitting only unused output tempering and float stores.
void advance_design_cpu_uniform(int64_t count) {
    TORCH_CHECK(count >= 0, "Negative RNG advance count");
    auto generator = at::detail::getDefaultCPUGenerator();
    auto* impl = generator.get<at::CPUGeneratorImpl>();
    std::lock_guard<std::mutex> lock(impl->mutex_);
    auto engine = impl->engine();
    while (count > 0) {
        auto state = engine.data();
        const int64_t available = state.left_ - 1;
        if (available == 0) {
            // The engine performs its original recurrence at the block boundary.
            (void)engine();
            --count;
        } else {
            const int64_t take = std::min(count, available);
            state.left_ -= static_cast<int>(take);
            state.next_ += static_cast<uint32_t>(take);
            engine.set_data(state);
            count -= take;
        }
    }
    impl->set_engine(engine);
}

TORCH_LIBRARY(boltzgen_design_rng, m) {
    m.def("advance(int count) -> ()", &advance_design_cpu_uniform);
}
