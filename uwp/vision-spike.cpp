// vision-spike.cpp — Xbox vision / DirectML technical spike.
//
// V0 deliberately does not load a vision model yet.
// Its only purpose is to prove that a dedicated vision execution surface
// can initialize ONNX Runtime and the DirectML execution provider inside
// the deployed Xbox UWP/Game package.
//
// Trigger:
//   LocalState\vision.flag
//
// Outputs:
//   LocalState\vision-result.json
//   LocalState\vision.done

#include "inference-bridge.h"

#ifdef XLLAMA_UWP

// clang-format off
    #include <windows.h>
    #include <onnxruntime_cxx_api.h>
    #include <dml_provider_factory.h>
// clang-format on

    #include <chrono>
    #include <cstdio>
    #include <exception>
    #include <string>

    #include "xllama/path_utils.h"
    #include "xllama/platform.h"
    #include "xllama/utf8_utils.h"

namespace xllama::bridge {

namespace {

void write_text(const char* name, const std::string& value) {
    const std::string path = resolve_local_path(name);
    FILE* fp = _wfopen(utf8_to_wstring(path).c_str(), L"wb");
    if (!fp)
        return;
    fwrite(value.data(), 1, value.size(), fp);
    fclose(fp);
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);

    for (unsigned char c : s) {
        switch (c) {
        case '\\': out += "\\\\"; break;
        case '"':  out += "\\\""; break;
        case '\n': out += "\\n";  break;
        case '\r': out += "\\r";  break;
        case '\t': out += "\\t";  break;
        default:
            if (c >= 0x20)
                out += static_cast<char>(c);
            break;
        }
    }

    return out;
}

void write_result(bool ok,
                  const std::string& stage,
                  long long elapsed_ms,
                  const std::string& error = {}) {
    std::string json =
        "{\n"
        "  \"spike\": \"vision-v0\",\n"
        "  \"backend\": \"onnxruntime-directml\",\n"
        "  \"device_id\": 0,\n"
        "  \"status\": \"" + std::string(ok ? "PASS" : "FAIL") + "\",\n"
        "  \"stage\": \"" + json_escape(stage) + "\",\n"
        "  \"elapsed_ms\": " + std::to_string(elapsed_ms);

    if (!error.empty())
        json += ",\n  \"error\": \"" + json_escape(error) + "\"";

    json += "\n}\n";

    write_text("vision-result.json", json);
    write_text("vision.done", ok ? "ok" : "error");
}

} // namespace

void run_vision_spike() {
    set_cwd_to_local_folder();
    log_output("[xllama] vision-spike: V0 starting\n");

    const auto started = std::chrono::steady_clock::now();

    try {
        // Creating the ORT environment validates the plain ORT runtime surface.
        Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "vision-spike");
        log_output("[xllama] vision-spike: ORT environment created\n");

        // Match the DirectML session requirements already used by diffuse.cpp
        // and op-repro.cpp. We intentionally stop before constructing a Session:
        // V1 will do that with a real visual encoder ONNX model.
        Ort::SessionOptions so;
        so.SetExecutionMode(ORT_SEQUENTIAL);
        so.DisableMemPattern();
        so.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

        Ort::ThrowOnError(
            OrtSessionOptionsAppendExecutionProvider_DML(so, /*device_id=*/0));

        log_output("[xllama] vision-spike: DirectML EP registered\n");

        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - started)
                .count();

        write_result(true, "directml_ep_registered", elapsed);
        log_output("[xllama] vision-spike: PASS\n");

    } catch (const Ort::Exception& e) {
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - started)
                .count();

        write_result(false, "onnxruntime", elapsed, e.what());
        log_output(std::string("[xllama] vision-spike ORT error: ") +
                   e.what() + "\n");

    } catch (const std::exception& e) {
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - started)
                .count();

        write_result(false, "exception", elapsed, e.what());
        log_output(std::string("[xllama] vision-spike error: ") +
                   e.what() + "\n");
    }
}

} // namespace xllama::bridge

#else

namespace xllama::bridge {
void run_vision_spike() {}
} // namespace xllama::bridge

#endif // XLLAMA_UWP
