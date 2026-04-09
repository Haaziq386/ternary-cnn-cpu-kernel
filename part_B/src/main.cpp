#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "model.h"
#include "resnet20.h"

namespace ternary
{

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
        float diff = 0.0f;
        for (std::size_t i = 0; i < lhs.size(); ++i)
        {
            diff = std::max(diff, std::abs(lhs.data[i] - rhs.data[i]));
        }
        return diff;
    }

} // namespace ternary

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " model.bin [--validate|--bench] [--iters N] [--warmup N]\n";
        return 1;
    }

    const std::string model_path = argv[1];
    bool validate = true;
    bool bench = false;
    int warmup = 10;
    int iters = 1000;

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
    }

    try
    {
        const ternary::ResNet20Weights model = ternary::load_model(model_path);
        ternary::InferenceScratch scratch;
        ternary::prepare_scratch(model, model.sample_input.n, scratch);

        if (validate)
        {
            const ternary::Tensor logits = ternary::run_resnet20(model, model.sample_input, scratch);
            ternary::Tensor probabilities = logits;
            ternary::softmax_inplace(probabilities);
            const float diff = ternary::max_abs_diff(probabilities, model.sample_outputs);

            int matches = 0;
            for (int n = 0; n < probabilities.n; ++n)
            {
                const float *predicted_row = probabilities.ptr() + static_cast<std::size_t>(n) * probabilities.c;
                const float *expected_row = model.sample_outputs.ptr() + static_cast<std::size_t>(n) * model.sample_outputs.c;
                const int predicted = ternary::argmax_row(predicted_row, probabilities.c);
                const int expected = ternary::argmax_row(expected_row, model.sample_outputs.c);
                if (predicted == expected)
                {
                    ++matches;
                }
            }

            std::cout << std::fixed << std::setprecision(6)
                      << "OK: " << matches << "/" << probabilities.n
                      << " top-1 matches, max probability diff = " << diff << "\n";
            std::cout << "Note: INT8 path - max diff vs FP32 reference may be 1e-2 to 1e-3\n";
            return 0;
        }

        if (bench)
        {
            const ternary::Tensor single = ternary::slice_first_sample(model.sample_input);
            ternary::prepare_scratch(model, single.n, scratch);

            for (int i = 0; i < warmup; ++i)
            {
                (void)ternary::run_resnet20(model, single, scratch);
            }

            std::vector<double> timings;
            timings.reserve(static_cast<std::size_t>(iters));
            for (int i = 0; i < iters; ++i)
            {
                const auto start = std::chrono::steady_clock::now();
                (void)ternary::run_resnet20(model, single, scratch);
                const auto end = std::chrono::steady_clock::now();
                const std::chrono::duration<double, std::micro> elapsed = end - start;
                timings.push_back(elapsed.count());
            }

            std::sort(timings.begin(), timings.end());
            const double mean = std::accumulate(timings.begin(), timings.end(), 0.0) / static_cast<double>(timings.size());
            const double median = timings[timings.size() / 2];
            const double p99 = timings[static_cast<std::size_t>(std::floor(0.99 * (timings.size() - 1)))];
            std::cout << std::fixed << std::setprecision(2)
                      << "mean_us=" << mean << " median_us=" << median << " p99_us=" << p99 << "\n";
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