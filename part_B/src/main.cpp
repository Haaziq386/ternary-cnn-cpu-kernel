#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(PROFILE_LAYERS) && defined(__linux__)
#include <cstring>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "model.h"
#include "resnet20.h"
#include "ternary_kernel.h"

namespace ternary
{

#if defined(PROFILE_LAYERS) && defined(__linux__)
    namespace
    {
        long long timeval_to_us(const timeval &tv)
        {
            return static_cast<long long>(tv.tv_sec) * 1000000LL + static_cast<long long>(tv.tv_usec);
        }

        struct UsageSnapshot
        {
            rusage ru{};
            bool ok = false;
        };

        UsageSnapshot capture_usage()
        {
            UsageSnapshot snap;
            snap.ok = (getrusage(RUSAGE_SELF, &snap.ru) == 0);
            return snap;
        }

        struct PerfLikeStats
        {
            bool available = false;
            bool hw_available = false;
            std::string hw_reason;
            double task_clock_ms = 0.0;
            double wall_ms = 0.0;
            double cpu_utilized = 0.0;
            long long context_switches = 0;
            long long voluntary_cs = 0;
            long long involuntary_cs = 0;
            long long page_faults = 0;
            long long minor_faults = 0;
            long long major_faults = 0;
            std::uint64_t instructions = 0;
            std::uint64_t cycles = 0;
            double ipc = 0.0;
            double avg_cpu_ghz = 0.0;
        };

        class PerfCounterGroup
        {
        public:
            PerfCounterGroup()
            {
                open_counters();
            }

            ~PerfCounterGroup()
            {
                if (fd_instructions_ >= 0)
                {
                    close(fd_instructions_);
                }
                if (fd_cycles_ >= 0)
                {
                    close(fd_cycles_);
                }
            }

            bool available() const
            {
                return fd_cycles_ >= 0 && fd_instructions_ >= 0;
            }

            const std::string &failure_reason() const
            {
                return reason_;
            }

            void start() const
            {
                if (!available())
                {
                    return;
                }
                (void)ioctl(fd_cycles_, PERF_EVENT_IOC_RESET, 0);
                (void)ioctl(fd_instructions_, PERF_EVENT_IOC_RESET, 0);
                (void)ioctl(fd_cycles_, PERF_EVENT_IOC_ENABLE, 0);
                (void)ioctl(fd_instructions_, PERF_EVENT_IOC_ENABLE, 0);
            }

            void stop() const
            {
                if (!available())
                {
                    return;
                }
                (void)ioctl(fd_cycles_, PERF_EVENT_IOC_DISABLE, 0);
                (void)ioctl(fd_instructions_, PERF_EVENT_IOC_DISABLE, 0);
            }

            std::uint64_t read_cycles() const
            {
                return read_value(fd_cycles_);
            }

            std::uint64_t read_instructions() const
            {
                return read_value(fd_instructions_);
            }

        private:
            int fd_cycles_ = -1;
            int fd_instructions_ = -1;
            std::string reason_;

            static int perf_open(perf_event_attr &attr)
            {
                return static_cast<int>(syscall(__NR_perf_event_open, &attr, 0, -1, -1, 0));
            }

            static std::uint64_t read_value(int fd)
            {
                if (fd < 0)
                {
                    return 0;
                }
                std::uint64_t value = 0;
                const ssize_t n = read(fd, &value, sizeof(value));
                if (n != static_cast<ssize_t>(sizeof(value)))
                {
                    return 0;
                }
                return value;
            }

            void open_counters()
            {
                perf_event_attr cycles_attr{};
                cycles_attr.size = sizeof(perf_event_attr);
                cycles_attr.type = PERF_TYPE_HARDWARE;
                cycles_attr.config = PERF_COUNT_HW_CPU_CYCLES;
                cycles_attr.disabled = 1;
                cycles_attr.exclude_kernel = 1;
                cycles_attr.exclude_hv = 1;
                cycles_attr.inherit = 1;

                fd_cycles_ = perf_open(cycles_attr);
                if (fd_cycles_ < 0)
                {
                    reason_ = std::string("cycles unavailable: ") + std::strerror(errno);
                    return;
                }

                perf_event_attr instr_attr{};
                instr_attr.size = sizeof(perf_event_attr);
                instr_attr.type = PERF_TYPE_HARDWARE;
                instr_attr.config = PERF_COUNT_HW_INSTRUCTIONS;
                instr_attr.disabled = 1;
                instr_attr.exclude_kernel = 1;
                instr_attr.exclude_hv = 1;
                instr_attr.inherit = 1;

                fd_instructions_ = perf_open(instr_attr);
                if (fd_instructions_ < 0)
                {
                    reason_ = std::string("instructions unavailable: ") + std::strerror(errno);
                    close(fd_cycles_);
                    fd_cycles_ = -1;
                    return;
                }
            }
        };

        PerfLikeStats collect_perf_like_stats(const UsageSnapshot &u0,
                                              const UsageSnapshot &u1,
                                              const std::chrono::steady_clock::time_point &t0,
                                              const std::chrono::steady_clock::time_point &t1,
                                              const PerfCounterGroup &hw)
        {
            PerfLikeStats stats;
            if (!u0.ok || !u1.ok)
            {
                return stats;
            }

            const long long cpu_us_0 = timeval_to_us(u0.ru.ru_utime) + timeval_to_us(u0.ru.ru_stime);
            const long long cpu_us_1 = timeval_to_us(u1.ru.ru_utime) + timeval_to_us(u1.ru.ru_stime);
            const long long cpu_us = std::max<long long>(0, cpu_us_1 - cpu_us_0);
            const auto wall_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

            stats.available = true;
            stats.task_clock_ms = static_cast<double>(cpu_us) / 1000.0;
            stats.wall_ms = static_cast<double>(wall_us) / 1000.0;
            stats.cpu_utilized = (stats.wall_ms > 0.0) ? (stats.task_clock_ms / stats.wall_ms) : 0.0;

            stats.voluntary_cs = static_cast<long long>(u1.ru.ru_nvcsw) - static_cast<long long>(u0.ru.ru_nvcsw);
            stats.involuntary_cs = static_cast<long long>(u1.ru.ru_nivcsw) - static_cast<long long>(u0.ru.ru_nivcsw);
            stats.context_switches = stats.voluntary_cs + stats.involuntary_cs;

            stats.minor_faults = static_cast<long long>(u1.ru.ru_minflt) - static_cast<long long>(u0.ru.ru_minflt);
            stats.major_faults = static_cast<long long>(u1.ru.ru_majflt) - static_cast<long long>(u0.ru.ru_majflt);
            stats.page_faults = stats.minor_faults + stats.major_faults;

            stats.hw_available = hw.available();
            if (stats.hw_available)
            {
                stats.cycles = hw.read_cycles();
                stats.instructions = hw.read_instructions();
                if (stats.cycles > 0)
                {
                    stats.ipc = static_cast<double>(stats.instructions) / static_cast<double>(stats.cycles);
                }
                if (stats.task_clock_ms > 0.0)
                {
                    const double task_clock_s = stats.task_clock_ms / 1000.0;
                    stats.avg_cpu_ghz = (task_clock_s > 0.0)
                                            ? (static_cast<double>(stats.cycles) / task_clock_s) / 1.0e9
                                            : 0.0;
                }
            }
            else
            {
                stats.hw_reason = hw.failure_reason();
            }

            return stats;
        }
    } // namespace
#endif

    Tensor slice_first_sample(const Tensor &tensor)
    {
        Tensor sample(1, tensor.c, tensor.h, tensor.w);
        const std::size_t sample_elems = static_cast<std::size_t>(tensor.c) * tensor.h * tensor.w;
        std::copy_n(tensor.ptr(), sample_elems, sample.ptr());
        return sample;
    }

    void softmax_inplace(Tensor &tensor)
    {
        const int classes = tensor.c;
        for (int n = 0; n < tensor.n; ++n)
        {
            float *row = tensor.ptr() + static_cast<std::size_t>(n) * classes;
            float max_value = row[0];
            for (int i = 1; i < classes; ++i)
            {
                max_value = std::max(max_value, row[i]);
            }
            float sum = 0.0f;
            for (int i = 0; i < classes; ++i)
            {
                row[i] = std::exp(row[i] - max_value);
                sum += row[i];
            }
            for (int i = 0; i < classes; ++i)
            {
                row[i] /= sum;
            }
        }
    }

    int argmax_row(const float *row, int length)
    {
        int best_index = 0;
        float best_value = row[0];
        for (int i = 1; i < length; ++i)
        {
            if (row[i] > best_value)
            {
                best_value = row[i];
                best_index = i;
            }
        }
        return best_index;
    }

    float max_abs_diff(const Tensor &lhs, const Tensor &rhs)
    {
        if (lhs.size() != rhs.size())
        {
            throw std::runtime_error("tensor shape mismatch when computing max_abs_diff");
        }
        float diff = 0.0f;
        for (std::size_t i = 0; i < lhs.size(); ++i)
        {
            diff = std::max(diff, std::abs(lhs.data[i] - rhs.data[i]));
        }
        return diff;
    }

    Tensor load_tensor_nchw_f32(const std::string &path, int channels, int height, int width)
    {
        if (channels <= 0 || height <= 0 || width <= 0)
        {
            throw std::runtime_error("invalid tensor shape for sample input");
        }

        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if (!in)
        {
            throw std::runtime_error("failed to open sample input file: " + path);
        }
        const std::streamsize size_bytes = in.tellg();
        if (size_bytes < 0)
        {
            throw std::runtime_error("failed to read sample input file size: " + path);
        }
        in.seekg(0, std::ios::beg);

        const std::size_t item_count = static_cast<std::size_t>(size_bytes) / sizeof(float);
        const std::size_t per_sample = static_cast<std::size_t>(channels) * height * width;
        if (per_sample == 0 || item_count % per_sample != 0)
        {
            throw std::runtime_error("sample input file has invalid size for NCHW float32 layout: " + path);
        }

        const int batch = static_cast<int>(item_count / per_sample);
        Tensor tensor(batch, channels, height, width);
        in.read(reinterpret_cast<char *>(tensor.ptr()), static_cast<std::streamsize>(item_count * sizeof(float)));
        if (!in)
        {
            throw std::runtime_error("failed to read sample input data: " + path);
        }
        return tensor;
    }

    Tensor load_tensor_nc_f32(const std::string &path, int channels, int expected_batch)
    {
        if (channels <= 0)
        {
            throw std::runtime_error("invalid class count for expected outputs");
        }

        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if (!in)
        {
            throw std::runtime_error("failed to open expected output file: " + path);
        }
        const std::streamsize size_bytes = in.tellg();
        if (size_bytes < 0)
        {
            throw std::runtime_error("failed to read expected output file size: " + path);
        }
        in.seekg(0, std::ios::beg);

        const std::size_t item_count = static_cast<std::size_t>(size_bytes) / sizeof(float);
        if (item_count % static_cast<std::size_t>(channels) != 0)
        {
            throw std::runtime_error("expected output file has invalid size for NxC float32 layout: " + path);
        }

        const int batch = static_cast<int>(item_count / static_cast<std::size_t>(channels));
        if (expected_batch > 0 && batch != expected_batch)
        {
            throw std::runtime_error("expected output batch size does not match sample input batch size");
        }

        Tensor tensor(batch, channels, 1, 1);
        in.read(reinterpret_cast<char *>(tensor.ptr()), static_cast<std::streamsize>(item_count * sizeof(float)));
        if (!in)
        {
            throw std::runtime_error("failed to read expected output data: " + path);
        }
        return tensor;
    }

} // namespace ternary

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " model.bin [--validate|--bench] [--sample-input path] [--expected-output path] [--iters N] [--warmup N] [--perf-like]\n";
        return 1;
    }

    const std::string model_path = argv[1];
    bool validate = true;
    bool bench = false;
    bool perf_like = false;
    int warmup = 10;
    int iters = 1000;
    std::string sample_input_path;
    std::string expected_output_path;

    for (int i = 2; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--bench")
        {
            bench = true;
            validate = false;
        }
        else if (arg == "--validate")
        {
            validate = true;
            bench = false;
        }
        else if (arg == "--iters" && i + 1 < argc)
        {
            iters = std::stoi(argv[++i]);
        }
        else if (arg == "--warmup" && i + 1 < argc)
        {
            warmup = std::stoi(argv[++i]);
        }
        else if (arg == "--sample-input" && i + 1 < argc)
        {
            sample_input_path = argv[++i];
        }
        else if (arg == "--expected-output" && i + 1 < argc)
        {
            expected_output_path = argv[++i];
        }
        else if (arg == "--perf-like")
        {
            perf_like = true;
        }
    }

    try
    {
        const ternary::ResNet20Weights model = ternary::load_model(model_path);
        bool requires_vnni = false;
        for (const auto &block : model.blocks)
        {
            if (block.conv1.storage_kind == ternary::TernaryStorageKind::kInt8 ||
                block.conv2.storage_kind == ternary::TernaryStorageKind::kInt8)
            {
                requires_vnni = true;
                break;
            }
        }
        if (requires_vnni && !ternary::cpu_supports_avx_vnni())
        {
            throw std::runtime_error("this model requires AVX-VNNI support");
        }

        if (validate)
        {
            const bool external_requested = !sample_input_path.empty() || !expected_output_path.empty();
            ternary::Tensor validate_input;
            ternary::Tensor expected_output;

            if (external_requested)
            {
                if (sample_input_path.empty() || expected_output_path.empty())
                {
                    throw std::runtime_error("--validate with external tensors requires both --sample-input and --expected-output");
                }
                validate_input = ternary::load_tensor_nchw_f32(sample_input_path, model.input_channels, model.input_h, model.input_w);
                expected_output = ternary::load_tensor_nc_f32(expected_output_path, model.num_classes, validate_input.n);
            }
            else if (model.sample_count > 0)
            {
                validate_input = model.sample_input;
                expected_output = model.sample_outputs;
            }
            else
            {
                throw std::runtime_error("model.bin has no embedded validation tensors; provide --sample-input and --expected-output");
            }

            ternary::InferenceScratch scratch;
            ternary::prepare_scratch(model, validate_input.n, scratch);

            const ternary::Tensor logits = ternary::run_resnet20(model, validate_input, scratch);
            ternary::Tensor probabilities = logits;
            ternary::softmax_inplace(probabilities);
            const float diff = ternary::max_abs_diff(probabilities, expected_output);

            int matches = 0;
            for (int n = 0; n < probabilities.n; ++n)
            {
                const float *predicted_row = probabilities.ptr() + static_cast<std::size_t>(n) * probabilities.c;
                const float *expected_row = expected_output.ptr() + static_cast<std::size_t>(n) * expected_output.c;
                const int predicted = ternary::argmax_row(predicted_row, probabilities.c);
                const int expected = ternary::argmax_row(expected_row, expected_output.c);
                if (predicted == expected)
                {
                    ++matches;
                }
            }

            std::cout << std::fixed << std::setprecision(6)
                      << "OK: " << matches << "/" << probabilities.n
                      << " top-1 matches, max probability diff = " << diff << "\n";
            return 0;
        }

        if (bench)
        {
            const ternary::Tensor single = (model.sample_count > 0)
                                               ? ternary::slice_first_sample(model.sample_input)
                                               : ternary::Tensor(1, model.input_channels, model.input_h, model.input_w);
            ternary::InferenceScratch scratch;
            ternary::prepare_scratch(model, single.n, scratch);

            for (int i = 0; i < warmup; ++i)
            {
                (void)ternary::run_resnet20(model, single, scratch);
            }

            std::vector<double> timings;
            timings.reserve(static_cast<std::size_t>(iters));

#if defined(PROFILE_LAYERS) && defined(__linux__)
            ternary::PerfCounterGroup hw_counters;
            ternary::UsageSnapshot usage_before;
            ternary::UsageSnapshot usage_after;
            std::chrono::steady_clock::time_point bench_t0;
            std::chrono::steady_clock::time_point bench_t1;
            if (perf_like)
            {
                usage_before = ternary::capture_usage();
                hw_counters.start();
                bench_t0 = std::chrono::steady_clock::now();
            }
#endif

            for (int i = 0; i < iters; ++i)
            {
                const auto start = std::chrono::steady_clock::now();
                (void)ternary::run_resnet20(model, single, scratch);
                const auto end = std::chrono::steady_clock::now();
                const std::chrono::duration<double, std::micro> elapsed = end - start;
                timings.push_back(elapsed.count());
            }

#if defined(PROFILE_LAYERS) && defined(__linux__)
            if (perf_like)
            {
                bench_t1 = std::chrono::steady_clock::now();
                hw_counters.stop();
                usage_after = ternary::capture_usage();
            }
#endif

            std::sort(timings.begin(), timings.end());
            const double mean = std::accumulate(timings.begin(), timings.end(), 0.0) / static_cast<double>(timings.size());
            const double median = timings[timings.size() / 2];
            const double p99 = timings[static_cast<std::size_t>(std::floor(0.99 * (timings.size() - 1)))];
            std::cout << std::fixed << std::setprecision(2)
                      << "mean_us=" << mean << " median_us=" << median << " p99_us=" << p99 << "\n";

#if defined(PROFILE_LAYERS) && defined(__linux__)
            if (perf_like)
            {
                const ternary::PerfLikeStats stats = ternary::collect_perf_like_stats(
                    usage_before, usage_after, bench_t0, bench_t1, hw_counters);
                if (stats.available)
                {
                    std::cout << std::fixed << std::setprecision(2)
                              << "[PERF_LIKE] task-clock-ms=" << stats.task_clock_ms
                              << " wall-ms=" << stats.wall_ms
                              << " cpus-utilized=" << stats.cpu_utilized << "\n";
                    std::cout << "[PERF_LIKE] context-switches=" << stats.context_switches
                              << " (voluntary=" << stats.voluntary_cs
                              << ", involuntary=" << stats.involuntary_cs << ")\n";
                    std::cout << "[PERF_LIKE] page-faults=" << stats.page_faults
                              << " (minor=" << stats.minor_faults
                              << ", major=" << stats.major_faults << ")\n";
                    if (stats.hw_available && stats.cycles > 0)
                    {
                        std::cout << std::setprecision(0)
                                  << "[PERF_LIKE] instructions=" << static_cast<double>(stats.instructions)
                                  << " cycles=" << static_cast<double>(stats.cycles) << "\n";
                        std::cout << std::setprecision(3)
                                  << "[PERF_LIKE] ipc=" << stats.ipc
                                  << " avg-cpu-ghz=" << stats.avg_cpu_ghz << "\n";
                    }
                    else
                    {
                        std::cout << "[PERF_LIKE] instructions/cycles unavailable";
                        if (!stats.hw_reason.empty())
                        {
                            std::cout << " (" << stats.hw_reason << ")";
                        }
                        std::cout << "\n";
                    }
                }
                else
                {
                    std::cout << "[PERF_LIKE] unavailable (getrusage failed)\n";
                }
            }
#else
            if (perf_like)
            {
                std::cout << "[PERF_LIKE] unavailable on this platform\n";
            }
#endif
            return 0;
        }

        std::cerr << "Choose either --validate or --bench\n";
        return 1;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}