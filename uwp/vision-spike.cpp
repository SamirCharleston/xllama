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
#include <cstdio>
#include <exception>
#include <numeric>
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

long long elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
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

std::string shape_json(const std::vector<int64_t>& shape) {
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

void write_text(const std::string& name, const std::string& text) {
    const auto path = utf8_to_wstring(resolve_local_path(name));
    FILE* fp = _wfopen(path.c_str(), L"wb");
    if (!fp)
        return;
    fwrite(text.data(), 1, text.size(), fp);
    fclose(fp);
}

void write_done(const std::string& text) {
    write_text("vision.done", text);
}

void write_failure(const std::string& stage, const std::string& error) {
    std::ostringstream os;
    os << "{\n"
       << "  \"spike\": \"vision-v1\",\n"
       << "  \"backend\": \"onnxruntime-directml\",\n"
       << "  \"device_id\": 0,\n"
       << "  \"status\": \"FAIL\",\n"
       << "  \"stage\": \"" << json_escape(stage) << "\",\n"
       << "  \"error\": \"" << json_escape(error) << "\"\n"
       << "}\n";

    write_text("vision-result.json", os.str());
    write_done("error");
}

} // namespace

void run_vision_spike() {
    set_cwd_to_local_folder();
    log_output("[xllama] vision-v1: starting MobileNetV2 DirectML spike\n");

    std::string stage = "initializing";

    try {
        const std::string model_rel =
            "vision-models\\mobilenetv2-12.onnx";
        const std::string model_path =
            resolve_local_path(model_rel);

        stage = "ort_environment";
        Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "vision-v1");

        stage = "directml_session";

        Ort::SessionOptions so;
        so.SetExecutionMode(ORT_SEQUENTIAL);
        so.DisableMemPattern();
        so.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

        Ort::ThrowOnError(
            OrtSessionOptionsAppendExecutionProvider_DML(so, 0));

        const auto load_start = Clock::now();

        Ort::Session session(
            env,
            utf8_to_wstring(model_path).c_str(),
            so);

        const auto load_end = Clock::now();

        log_output("[xllama] vision-v1: DML session created\n");

        stage = "input_introspection";

        if (session.GetInputCount() != 1)
            throw std::runtime_error(
                "vision-v1 expects exactly one model input");

        Ort::AllocatorWithDefaultOptions alloc;

        auto input_name_holder =
            session.GetInputNameAllocated(0, alloc);

        const std::string input_name =
            input_name_holder.get();

        Ort::TypeInfo input_type =
            session.GetInputTypeInfo(0);

        auto input_info =
            input_type.GetTensorTypeAndShapeInfo();

        auto input_shape =
            input_info.GetShape();

        if (input_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error(
                "vision-v1 expects FP32 input");

        size_t input_count = 1;

        // Resolve dynamic dimensions for the synthetic V1 test.
        // MobileNetV2 may expose a dynamic batch dimension (-1).
        // For this spike we execute a single image/batch.
        for (auto& d : input_shape) {
            if (d <= 0)
                d = 1;

            input_count *= static_cast<size_t>(d);
        }

        // Deterministic synthetic image tensor.
        // Values cover roughly [0,1] repeatedly; this is not intended
        // to classify a meaningful image. It proves real tensor execution.
        std::vector<float> input_data(input_count);

        for (size_t i = 0; i < input_count; ++i)
            input_data[i] =
                static_cast<float>(i % 256) / 255.0f;

        Ort::MemoryInfo mem =
            Ort::MemoryInfo::CreateCpu(
                OrtDeviceAllocator,
                OrtMemTypeCPU);

        Ort::Value input_tensor =
            Ort::Value::CreateTensor<float>(
                mem,
                input_data.data(),
                input_data.size(),
                input_shape.data(),
                input_shape.size());

        stage = "output_introspection";

        const size_t output_count =
            session.GetOutputCount();

        if (output_count == 0)
            throw std::runtime_error(
                "model has no outputs");

        std::vector<Ort::AllocatedStringPtr>
            output_name_holders;

        std::vector<const char*>
            output_names;

        output_name_holders.reserve(output_count);
        output_names.reserve(output_count);

        for (size_t i = 0; i < output_count; ++i) {
            output_name_holders.push_back(
                session.GetOutputNameAllocated(i, alloc));

            output_names.push_back(
                output_name_holders.back().get());
        }

        const char* input_names[] = {
            input_name_holder.get()
        };

        stage = "session_run";

        const auto run_start = Clock::now();

        auto outputs =
            session.Run(
                Ort::RunOptions{nullptr},
                input_names,
                &input_tensor,
                1,
                output_names.data(),
                output_names.size());

        const auto run_end = Clock::now();

        if (outputs.empty())
            throw std::runtime_error(
                "Session.Run returned no outputs");

        stage = "output_validation";

        auto output_info =
            outputs[0].GetTensorTypeAndShapeInfo();

        const auto output_shape =
            output_info.GetShape();

        if (output_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error(
                "vision-v1 expects FP32 output");

        const size_t output_elements =
            output_info.GetElementCount();

        const float* output_data =
            outputs[0].GetTensorData<float>();

        if (!output_data || output_elements == 0)
            throw std::runtime_error(
                "empty output tensor");

        const size_t sample_count =
            std::min<size_t>(10, output_elements);

        std::ostringstream sample;
        sample << "[";

        for (size_t i = 0; i < sample_count; ++i) {
            if (i)
                sample << ", ";
            sample << output_data[i];
        }

        sample << "]";

        const auto max_it =
            std::max_element(
                output_data,
                output_data + output_elements);

        const size_t max_index =
            static_cast<size_t>(
                std::distance(output_data, max_it));

        stage = "session_run_completed";

        std::ostringstream result;

        result
            << "{\n"
            << "  \"spike\": \"vision-v1\",\n"
            << "  \"backend\": \"onnxruntime-directml\",\n"
            << "  \"device_id\": 0,\n"
            << "  \"status\": \"PASS\",\n"
            << "  \"stage\": \"session_run_completed\",\n"
            << "  \"model\": \"mobilenetv2-12.onnx\",\n"
            << "  \"input_name\": \""
            << json_escape(input_name) << "\",\n"
            << "  \"input_shape\": "
            << shape_json(input_shape) << ",\n"
            << "  \"output_shape\": "
            << shape_json(output_shape) << ",\n"
            << "  \"model_load_ms\": "
            << elapsed_ms(load_start, load_end) << ",\n"
            << "  \"inference_ms\": "
            << elapsed_ms(run_start, run_end) << ",\n"
            << "  \"input_elements\": "
            << input_count << ",\n"
            << "  \"output_elements\": "
            << output_elements << ",\n"
            << "  \"max_output_index\": "
            << max_index << ",\n"
            << "  \"max_output_value\": "
            << *max_it << ",\n"
            << "  \"output_sample\": "
            << sample.str() << "\n"
            << "}\n";

        write_text(
            "vision-result.json",
            result.str());

        write_done("ok");

        log_output(
            "[xllama] vision-v1: Session.Run completed successfully\n");

    } catch (const Ort::Exception& e) {
        log_output(
            std::string("[xllama] vision-v1 ORT error: ") +
            e.what() + "\n");

        write_failure(stage, e.what());

    } catch (const std::exception& e) {
        log_output(
            std::string("[xllama] vision-v1 error: ") +
            e.what() + "\n");

        write_failure(stage, e.what());
    }
}

} // namespace xllama::bridge

#else

namespace xllama::bridge {
void run_vision_spike() {}
} // namespace xllama::bridge

#endif // XLLAMA_UWP
