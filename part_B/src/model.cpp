#include "model.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ternary
{
    namespace
    {

        constexpr char kMagic[8] = {'T', 'R', 'N', 'C', 'N', 'N', 'B', '1'};

        int round_up(int value, int multiple)
        {
            return ((value + multiple - 1) / multiple) * multiple;
        }

#pragma pack(push, 1)
        struct FileHeader
        {
            char magic[8];
            std::uint32_t version;
            std::uint32_t sample_count;
            std::uint32_t input_channels;
            std::uint32_t input_h;
            std::uint32_t input_w;
            std::uint32_t num_classes;
            std::uint32_t layer_count;
            std::uint32_t reserved[5];
        };

        struct LayerHeader
        {
            std::uint32_t kind;
            std::uint32_t in_channels;
            std::uint32_t out_channels;
            std::uint32_t kernel_h;
            std::uint32_t kernel_w;
            std::uint32_t stride_h;
            std::uint32_t stride_w;
            std::uint32_t padding_h;
            std::uint32_t padding_w;
            std::uint32_t output_h;
            std::uint32_t output_w;
            std::uint32_t k_pad;
            std::uint32_t has_bias;
            std::uint32_t name_len;
            std::uint32_t reserved;
        };
#pragma pack(pop)

        class MappedFile
        {
        public:
            explicit MappedFile(const std::string &path)
            {
                fd_ = ::open(path.c_str(), O_RDONLY);
                if (fd_ < 0)
                {
                    throw std::runtime_error("failed to open model file: " + path);
                }
                struct stat st{};
                if (::fstat(fd_, &st) != 0)
                {
                    ::close(fd_);
                    throw std::runtime_error("failed to stat model file: " + path);
                }
                size_ = static_cast<std::size_t>(st.st_size);
                data_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
                if (data_ == MAP_FAILED)
                {
                    ::close(fd_);
                    throw std::runtime_error("failed to mmap model file: " + path);
                }
            }

            ~MappedFile()
            {
                if (data_ != MAP_FAILED)
                {
                    ::munmap(data_, size_);
                }
                if (fd_ >= 0)
                {
                    ::close(fd_);
                }
            }

            const std::uint8_t *data() const
            {
                return static_cast<const std::uint8_t *>(data_);
            }

            std::size_t size() const
            {
                return size_;
            }

        private:
            int fd_ = -1;
            void *data_ = MAP_FAILED;
            std::size_t size_ = 0;
        };

        class Reader
        {
        public:
            Reader(const std::uint8_t *data, std::size_t size) : data_(data), size_(size) {}

            template <typename T>
            T read()
            {
                if (offset_ + sizeof(T) > size_)
                {
                    throw std::runtime_error("unexpected end of model file");
                }
                T value;
                std::memcpy(&value, data_ + offset_, sizeof(T));
                offset_ += sizeof(T);
                return value;
            }

            void read_bytes(void *dst, std::size_t count)
            {
                if (offset_ + count > size_)
                {
                    throw std::runtime_error("unexpected end of model file");
                }
                std::memcpy(dst, data_ + offset_, count);
                offset_ += count;
            }

            void skip(std::size_t count)
            {
                if (offset_ + count > size_)
                {
                    throw std::runtime_error("unexpected end of model file");
                }
                offset_ += count;
            }

        private:
            const std::uint8_t *data_ = nullptr;
            std::size_t size_ = 0;
            std::size_t offset_ = 0;
        };

        template <typename T>
        void read_vector(Reader &reader, std::vector<T> &values, std::size_t count)
        {
            values.resize(count);
            reader.read_bytes(values.data(), sizeof(T) * count);
        }

        Conv2DWeightsFP32 read_fp32_conv(Reader &reader)
        {
            const LayerHeader header = reader.read<LayerHeader>();
            if (header.kind != static_cast<std::uint32_t>(LayerKind::kFp32Conv))
            {
                throw std::runtime_error("expected fp32 conv layer in model file");
            }
            Conv2DWeightsFP32 weights;
            weights.in_channels = static_cast<int>(header.in_channels);
            weights.out_channels = static_cast<int>(header.out_channels);
            weights.kernel_h = static_cast<int>(header.kernel_h);
            weights.kernel_w = static_cast<int>(header.kernel_w);
            weights.stride_h = static_cast<int>(header.stride_h);
            weights.stride_w = static_cast<int>(header.stride_w);
            weights.padding_h = static_cast<int>(header.padding_h);
            weights.padding_w = static_cast<int>(header.padding_w);
            weights.output_h = static_cast<int>(header.output_h);
            weights.output_w = static_cast<int>(header.output_w);
            weights.has_bias = header.has_bias != 0;
            reader.skip(header.name_len);
            const std::size_t kernel_elements = static_cast<std::size_t>(weights.in_channels) * weights.kernel_h * weights.kernel_w;
            read_vector(reader, weights.weight, static_cast<std::size_t>(weights.out_channels) * kernel_elements);
            if (weights.has_bias)
            {
                read_vector(reader, weights.bias, weights.out_channels);
            }
            return weights;
        }

        TernaryConv2DWeights read_ternary_conv(Reader &reader)
        {
            const LayerHeader header = reader.read<LayerHeader>();
            if (header.kind != static_cast<std::uint32_t>(LayerKind::kTernaryConv))
            {
                throw std::runtime_error("expected ternary conv layer in model file");
            }
            TernaryConv2DWeights weights;
            weights.in_channels = static_cast<int>(header.in_channels);
            weights.out_channels = static_cast<int>(header.out_channels);
            weights.kernel_h = static_cast<int>(header.kernel_h);
            weights.kernel_w = static_cast<int>(header.kernel_w);
            weights.stride_h = static_cast<int>(header.stride_h);
            weights.stride_w = static_cast<int>(header.stride_w);
            weights.padding_h = static_cast<int>(header.padding_h);
            weights.padding_w = static_cast<int>(header.padding_w);
            weights.output_h = static_cast<int>(header.output_h);
            weights.output_w = static_cast<int>(header.output_w);
            weights.k_pad = static_cast<int>(header.k_pad);
            reader.skip(header.name_len);
            const std::size_t packed_bytes_per_row = static_cast<std::size_t>(weights.k_pad) / 8;
            const std::size_t total_packed = static_cast<std::size_t>(weights.out_channels) * packed_bytes_per_row;
            read_vector(reader, weights.pos_bits, total_packed);
            read_vector(reader, weights.neg_bits, total_packed);
            read_vector(reader, weights.scale, weights.out_channels);
            read_vector(reader, weights.bias, weights.out_channels);
            return weights;
        }

        LinearWeights read_linear(Reader &reader)
        {
            const LayerHeader header = reader.read<LayerHeader>();
            if (header.kind != static_cast<std::uint32_t>(LayerKind::kLinear))
            {
                throw std::runtime_error("expected linear layer in model file");
            }
            LinearWeights weights;
            weights.in_features = static_cast<int>(header.in_channels);
            weights.out_features = static_cast<int>(header.out_channels);
            reader.skip(header.name_len);
            read_vector(reader, weights.weight, static_cast<std::size_t>(weights.out_features) * weights.in_features);
            read_vector(reader, weights.bias, weights.out_features);
            return weights;
        }

        Tensor read_tensor(Reader &reader, int batch, int channels, int height, int width)
        {
            Tensor tensor(batch, channels, height, width);
            read_vector(reader, tensor.data, tensor.size());
            return tensor;
        }

    } // namespace

    ResNet20Weights load_model(const std::string &path)
    {
        MappedFile mapped(path);
        Reader reader(mapped.data(), mapped.size());

        const FileHeader header = reader.read<FileHeader>();
        if (std::memcmp(header.magic, kMagic, sizeof(kMagic)) != 0)
        {
            throw std::runtime_error("invalid model magic");
        }
        if (header.version != 1)
        {
            throw std::runtime_error("unsupported model version");
        }

        ResNet20Weights model;
        model.stem = read_fp32_conv(reader);
        for (auto &block : model.blocks)
        {
            block.conv1 = read_ternary_conv(reader);
            block.conv2 = read_ternary_conv(reader);
            const std::uint32_t has_projection = reader.read<std::uint32_t>();
            if (has_projection != 0)
            {
                block.has_projection = true;
                block.projection = read_fp32_conv(reader);
            }
        }
        model.fc = read_linear(reader);

        model.sample_input = read_tensor(reader, static_cast<int>(header.sample_count), static_cast<int>(header.input_channels), static_cast<int>(header.input_h), static_cast<int>(header.input_w));
        model.sample_labels.resize(header.sample_count);
        reader.read_bytes(model.sample_labels.data(), sizeof(std::int64_t) * header.sample_count);
        model.sample_outputs = read_tensor(reader, static_cast<int>(header.sample_count), static_cast<int>(header.num_classes), 1, 1);
        return model;
    }

    void prepare_scratch(const ResNet20Weights &model, int batch_size, InferenceScratch &scratch)
    {
        const std::size_t im2col_capacity = estimate_im2col_capacity(model, batch_size);
        scratch.a.reserve(static_cast<std::size_t>(batch_size) * 16 * 32 * 32);
        scratch.b.reserve(static_cast<std::size_t>(batch_size) * 16 * 32 * 32);
        scratch.c.reserve(static_cast<std::size_t>(batch_size) * 16 * 32 * 32);
        scratch.im2col.reserve(im2col_capacity);
        scratch.im2col_int8.reserve(im2col_capacity);
    }

    std::size_t estimate_im2col_capacity(const ResNet20Weights &model, int batch_size)
    {
        std::size_t capacity = 0;
        const auto consider = [&capacity, batch_size](int in_channels, int kernel_h, int kernel_w, int out_h, int out_w)
        {
            const int kernel_elements = in_channels * kernel_h * kernel_w;
            const int k_pad = round_up(kernel_elements, 32);
            capacity = std::max(capacity, static_cast<std::size_t>(batch_size) * out_h * out_w * k_pad);
        };

        consider(model.stem.in_channels, model.stem.kernel_h, model.stem.kernel_w, model.stem.output_h, model.stem.output_w);
        for (const auto &block : model.blocks)
        {
            consider(block.conv1.in_channels, block.conv1.kernel_h, block.conv1.kernel_w, block.conv1.output_h, block.conv1.output_w);
            consider(block.conv2.in_channels, block.conv2.kernel_h, block.conv2.kernel_w, block.conv2.output_h, block.conv2.output_w);
            if (block.has_projection)
            {
                consider(block.projection.in_channels, block.projection.kernel_h, block.projection.kernel_w, block.projection.output_h, block.projection.output_w);
            }
        }
        consider(model.fc.in_features, 1, 1, 1, 1);
        return capacity;
    }

} // namespace ternary