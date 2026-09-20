#include "inference-bridge.h"

#ifdef XLLAMA_UWP

#include <windows.h>
#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>

#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Imaging.h>
#include <winrt/Windows.Storage.h>
#include <winrt/Windows.Storage.Streams.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "xllama/path_utils.h"
#include "xllama/platform.h"
#include "xllama/utf8_utils.h"

namespace xllama::bridge {
namespace {

using Clock = std::chrono::steady_clock;

long long elapsed_ms(
    Clock::time_point start,
    Clock::time_point end) {

    return std::chrono::duration_cast<
        std::chrono::milliseconds>(end - start).count();
}

std::string json_escape(const std::string& s) {
    std::string out;

    for (char c : s) {
        switch (c) {
        case '\\': out += "\\\\"; break;
        case '"':  out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:   out += c; break;
        }
    }

    return out;
}

std::string shape_json(
    const std::vector<int64_t>& shape) {

    std::ostringstream os;
    os << "[";

    for (size_t i = 0; i < shape.size(); ++i) {
        if (i)
            os << ", ";

        os << shape[i];
    }

    os << "]";
    return os.str();
}

void write_text(
    const std::string& name,
    const std::string& text) {

    const auto path =
        utf8_to_wstring(resolve_local_path(name));

    FILE* fp =
        _wfopen(path.c_str(), L"wb");

    if (!fp)
        return;

    fwrite(
        text.data(),
        1,
        text.size(),
        fp);

    fclose(fp);
}

void write_done(const std::string& text) {
    write_text("vision.done", text);
}

void write_failure(
    const std::string& stage,
    const std::string& error) {

    std::ostringstream os;

    os
        << "{\n"
        << "  \"spike\": \"vision-v2.1\",\n"
        << "  \"backend\": \"onnxruntime-directml\",\n"
        << "  \"device_id\": 0,\n"
        << "  \"status\": \"FAIL\",\n"
        << "  \"stage\": \""
        << json_escape(stage)
        << "\",\n"
        << "  \"error\": \""
        << json_escape(error)
        << "\"\n"
        << "}\n";

    write_text(
        "vision-result.json",
        os.str());

    write_done("error");
}

struct PreparedImage {
    uint32_t source_width = 0;
    uint32_t source_height = 0;

    uint32_t resized_width = 224;
    uint32_t resized_height = 224;

    std::vector<float> tensor;
};

PreparedImage load_and_prepare_image() {
    using namespace winrt::Windows::Graphics::Imaging;
    using namespace winrt::Windows::Storage;

    constexpr uint32_t target_width = 224;
    constexpr uint32_t target_height = 224;

    auto local =
        ApplicationData::Current().LocalFolder();

    auto file =
        local.GetFileAsync(
            L"vision-input.jpg").get();

    auto stream =
        file.OpenAsync(
            FileAccessMode::Read).get();

    auto decoder =
        BitmapDecoder::CreateAsync(
            stream).get();

    PreparedImage image;

    image.source_width =
        decoder.PixelWidth();

    image.source_height =
        decoder.PixelHeight();

    if (image.source_width == 0 ||
        image.source_height == 0) {

        throw std::runtime_error(
            "decoded image has invalid dimensions");
    }

    BitmapTransform transform;

    transform.ScaledWidth(target_width);
    transform.ScaledHeight(target_height);

    auto pixel_data_provider =
        decoder.GetPixelDataAsync(
            BitmapPixelFormat::Rgba8,
            BitmapAlphaMode::Ignore,
            transform,
            ExifOrientationMode::RespectExifOrientation,
            ColorManagementMode::ColorManageToSRgb)
        .get();

    auto pixels =
        pixel_data_provider.DetachPixelData();

    const size_t expected_bytes =
        static_cast<size_t>(target_width) *
        static_cast<size_t>(target_height) *
        4;

    if (pixels.size() != expected_bytes) {
        std::ostringstream error;

        error
            << "unexpected decoded pixel count: "
            << pixels.size()
            << ", expected "
            << expected_bytes;

        throw std::runtime_error(
            error.str());
    }

    const size_t plane_size =
        static_cast<size_t>(target_width) *
        static_cast<size_t>(target_height);

    image.tensor.resize(
        plane_size * 3);

    // ImageNet normalization.
    constexpr float mean_r = 0.485f;
    constexpr float mean_g = 0.456f;
    constexpr float mean_b = 0.406f;

    constexpr float std_r = 0.229f;
    constexpr float std_g = 0.224f;
    constexpr float std_b = 0.225f;

    // RGBA interleaved -> RGB NCHW float32.
    for (size_t i = 0; i < plane_size; ++i) {
        const size_t p = i * 4;

        const float r =
            static_cast<float>(pixels[p + 0]) /
            255.0f;

        const float g =
            static_cast<float>(pixels[p + 1]) /
            255.0f;

        const float b =
            static_cast<float>(pixels[p + 2]) /
            255.0f;

        image.tensor[i] =
            (r - mean_r) / std_r;

        image.tensor[plane_size + i] =
            (g - mean_g) / std_g;

        image.tensor[(plane_size * 2) + i] =
            (b - mean_b) / std_b;
    }

    return image;
}

} // namespace

void run_vision_spike() {
    set_cwd_to_local_folder();

    log_output(
        "[xllama] vision-v2.1: starting real image inference\n");

    std::string stage = "initializing";

    try {
        const std::string model_rel =
            "vision-models\\mobilenetv2-12.onnx";

        const std::string model_path =
            resolve_local_path(model_rel);

        // ------------------------------------------------------------
        // Decode + preprocessing
        // ------------------------------------------------------------

        stage = "image_preprocessing";

        const auto preprocess_start =
            Clock::now();

        PreparedImage image =
            load_and_prepare_image();

        const auto preprocess_end =
            Clock::now();

        log_output(
            "[xllama] vision-v2.1: real image prepared\n");

        // ------------------------------------------------------------
        // ONNX Runtime + DirectML
        // ------------------------------------------------------------

        stage = "ort_environment";

        Ort::Env env(
            ORT_LOGGING_LEVEL_ERROR,
            "vision-v2.1");

        stage = "directml_session";

        Ort::SessionOptions so;

        so.SetExecutionMode(
            ORT_SEQUENTIAL);

        so.DisableMemPattern();

        so.SetGraphOptimizationLevel(
            ORT_ENABLE_EXTENDED);

        Ort::ThrowOnError(
            OrtSessionOptionsAppendExecutionProvider_DML(
                so,
                0));

        const auto load_start =
            Clock::now();

        Ort::Session session(
            env,
            utf8_to_wstring(model_path).c_str(),
            so);

        const auto load_end =
            Clock::now();

        log_output(
            "[xllama] vision-v2.1: DML session created\n");

        // ------------------------------------------------------------
        // Input introspection
        // ------------------------------------------------------------

        stage = "input_introspection";

        if (session.GetInputCount() != 1) {
            throw std::runtime_error(
                "vision-v2.1 expects exactly one model input");
        }

        Ort::AllocatorWithDefaultOptions alloc;

        auto input_name_holder =
            session.GetInputNameAllocated(
                0,
                alloc);

        const std::string input_name =
            input_name_holder.get();

        Ort::TypeInfo input_type =
            session.GetInputTypeInfo(0);

        auto input_info =
            input_type.GetTensorTypeAndShapeInfo();

        auto input_shape =
            input_info.GetShape();

        if (input_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {

            throw std::runtime_error(
                "vision-v2.1 expects FP32 input");
        }

        // Resolve dynamic batch dimension.
        for (auto& d : input_shape) {
            if (d <= 0)
                d = 1;
        }

        if (input_shape.size() != 4) {
            throw std::runtime_error(
                "vision-v2.1 expects a 4D NCHW input");
        }

        if (input_shape[0] != 1 ||
            input_shape[1] != 3 ||
            input_shape[2] != 224 ||
            input_shape[3] != 224) {

            throw std::runtime_error(
                "vision-v2.1 expected model input [1,3,224,224]");
        }

        const size_t input_count =
            image.tensor.size();

        if (input_count !=
            static_cast<size_t>(
                1 * 3 * 224 * 224)) {

            throw std::runtime_error(
                "prepared tensor has unexpected size");
        }

        Ort::MemoryInfo mem =
            Ort::MemoryInfo::CreateCpu(
                OrtDeviceAllocator,
                OrtMemTypeCPU);

        Ort::Value input_tensor =
            Ort::Value::CreateTensor<float>(
                mem,
                image.tensor.data(),
                image.tensor.size(),
                input_shape.data(),
                input_shape.size());

        // ------------------------------------------------------------
        // Outputs
        // ------------------------------------------------------------

        stage = "output_introspection";

        const size_t output_count =
            session.GetOutputCount();

        if (output_count == 0) {
            throw std::runtime_error(
                "model has no outputs");
        }

        std::vector<Ort::AllocatedStringPtr>
            output_name_holders;

        std::vector<const char*>
            output_names;

        output_name_holders.reserve(
            output_count);

        output_names.reserve(
            output_count);

        for (size_t i = 0;
             i < output_count;
             ++i) {

            output_name_holders.push_back(
                session.GetOutputNameAllocated(
                    i,
                    alloc));

            output_names.push_back(
                output_name_holders.back().get());
        }

        const char* input_names[] = {
            input_name_holder.get()
        };

        // ------------------------------------------------------------
        // Real inference
        // ------------------------------------------------------------

        stage = "session_run";

        const auto run_start =
            Clock::now();

        auto outputs =
            session.Run(
                Ort::RunOptions{nullptr},
                input_names,
                &input_tensor,
                1,
                output_names.data(),
                output_names.size());

        const auto run_end =
            Clock::now();

        if (outputs.empty()) {
            throw std::runtime_error(
                "Session.Run returned no outputs");
        }

        // ------------------------------------------------------------
        // Output validation
        // ------------------------------------------------------------

        stage = "output_validation";

        auto output_info =
            outputs[0]
                .GetTensorTypeAndShapeInfo();

        const auto output_shape =
            output_info.GetShape();

        if (output_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {

            throw std::runtime_error(
                "vision-v2.1 expects FP32 output");
        }

        const size_t output_elements =
            output_info.GetElementCount();

        const float* output_data =
            outputs[0]
                .GetTensorData<float>();

        if (!output_data ||
            output_elements == 0) {

            throw std::runtime_error(
                "empty output tensor");
        }

        const auto max_it =
            std::max_element(
                output_data,
                output_data +
                    output_elements);

        const size_t max_index =
            static_cast<size_t>(
                std::distance(
                    output_data,
                    max_it));

        // Top 5 raw output indices.
        std::vector<size_t> indices(
            output_elements);

        for (size_t i = 0;
             i < output_elements;
             ++i) {

            indices[i] = i;
        }

        const size_t top_count =
            std::min<size_t>(
                5,
                indices.size());

        std::partial_sort(
            indices.begin(),
            indices.begin() + top_count,
            indices.end(),
            [output_data](
                size_t a,
                size_t b) {

                return output_data[a] >
                       output_data[b];
            });

        std::ostringstream top5;

        top5 << "[";

        for (size_t i = 0;
             i < top_count;
             ++i) {

            if (i)
                top5 << ", ";

            const size_t index =
                indices[i];

            top5
                << "{"
                << "\"index\": "
                << index
                << ", "
                << "\"value\": "
                << output_data[index]
                << "}";
        }

        top5 << "]";

        // ------------------------------------------------------------
        // Result
        // ------------------------------------------------------------

        stage =
            "real_image_inference_completed";

        std::ostringstream result;

        result
            << "{\n"
            << "  \"spike\": \"vision-v2.1\",\n"
            << "  \"backend\": \"onnxruntime-directml\",\n"
            << "  \"device_id\": 0,\n"
            << "  \"status\": \"PASS\",\n"
            << "  \"stage\": \"real_image_inference_completed\",\n"
            << "  \"image\": \"vision-input.jpg\",\n"

            << "  \"source_width\": "
            << image.source_width
            << ",\n"

            << "  \"source_height\": "
            << image.source_height
            << ",\n"

            << "  \"processed_width\": "
            << image.resized_width
            << ",\n"

            << "  \"processed_height\": "
            << image.resized_height
            << ",\n"

            << "  \"pixel_format\": \"RGBA8 -> RGB FP32 NCHW\",\n"
            << "  \"normalization\": \"ImageNet mean/std\",\n"

            << "  \"input_name\": \""
            << json_escape(input_name)
            << "\",\n"

            << "  \"input_shape\": "
            << shape_json(input_shape)
            << ",\n"

            << "  \"output_shape\": "
            << shape_json(output_shape)
            << ",\n"

            << "  \"preprocess_ms\": "
            << elapsed_ms(
                   preprocess_start,
                   preprocess_end)
            << ",\n"

            << "  \"model_load_ms\": "
            << elapsed_ms(
                   load_start,
                   load_end)
            << ",\n"

            << "  \"inference_ms\": "
            << elapsed_ms(
                   run_start,
                   run_end)
            << ",\n"

            << "  \"input_elements\": "
            << input_count
            << ",\n"

            << "  \"output_elements\": "
            << output_elements
            << ",\n"

            << "  \"max_output_index\": "
            << max_index
            << ",\n"

            << "  \"max_output_value\": "
            << *max_it
            << ",\n"

            << "  \"top5\": "
            << top5.str()
            << "\n"

            << "}\n";

        write_text(
            "vision-result.json",
            result.str());

        write_done("ok");

        log_output(
            "[xllama] vision-v2.1: real image inference completed successfully\n");

    } catch (const winrt::hresult_error& e) {
        const std::string error =
            winrt::to_string(
                e.message());

        log_output(
            std::string(
                "[xllama] vision-v2.1 WinRT error: ") +
            error +
            "\n");

        write_failure(
            stage,
            error);

    } catch (const Ort::Exception& e) {
        log_output(
            std::string(
                "[xllama] vision-v2.1 ORT error: ") +
            e.what() +
            "\n");

        write_failure(
            stage,
            e.what());

    } catch (const std::exception& e) {
        log_output(
            std::string(
                "[xllama] vision-v2.1 error: ") +
            e.what() +
            "\n");

        write_failure(
            stage,
            e.what());
    }
}

} // namespace xllama::bridge

#else

namespace xllama::bridge {

void run_vision_spike() {}

} // namespace xllama::bridge

#endif // XLLAMA_UWP
