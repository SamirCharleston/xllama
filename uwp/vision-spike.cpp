#include "inference-bridge.h"

#ifdef XLLAMA_UWP

#include <windows.h>
#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>

#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Imaging.h>
#include <winrt/Windows.Storage.h>
#include <winrt/Windows.Storage.Streams.h>

#include <chrono>
#include <cstdio>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>

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
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            out += c;
            break;
        }
    }

    return out;
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
    write_text(
        "vision.done",
        text);
}

void write_failure(
    const std::string& stage,
    const std::string& error) {

    std::ostringstream os;

    os
        << "{\n"
        << "  \"spike\": \"vision-v2\",\n"
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

struct DecodedImageInfo {
    uint32_t width = 0;
    uint32_t height = 0;
};

DecodedImageInfo decode_test_image() {
    using namespace winrt::Windows::Graphics::Imaging;
    using namespace winrt::Windows::Storage;

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

    DecodedImageInfo info;

    info.width =
        decoder.PixelWidth();

    info.height =
        decoder.PixelHeight();

    if (info.width == 0 ||
        info.height == 0) {

        throw std::runtime_error(
            "decoded image has invalid dimensions");
    }

    return info;
}

} // namespace

void run_vision_spike() {
    set_cwd_to_local_folder();

    log_output(
        "[xllama] vision-v2: starting real image decode test\n");

    std::string stage =
        "initializing";

    try {
        stage =
            "image_decode";

        const auto decode_start =
            Clock::now();

        const DecodedImageInfo image =
            decode_test_image();

        const auto decode_end =
            Clock::now();

        stage =
            "image_decoded";

        std::ostringstream result;

        result
            << "{\n"
            << "  \"spike\": \"vision-v2\",\n"
            << "  \"status\": \"PASS\",\n"
            << "  \"stage\": \"image_decoded\",\n"
            << "  \"image\": \"vision-input.jpg\",\n"
            << "  \"source_width\": "
            << image.width
            << ",\n"
            << "  \"source_height\": "
            << image.height
            << ",\n"
            << "  \"decode_ms\": "
            << elapsed_ms(
                   decode_start,
                   decode_end)
            << "\n"
            << "}\n";

        write_text(
            "vision-result.json",
            result.str());

        write_done("ok");

        log_output(
            "[xllama] vision-v2: image decoded successfully\n");

    } catch (const winrt::hresult_error& e) {
        const std::string error =
            winrt::to_string(e.message());

        log_output(
            std::string(
                "[xllama] vision-v2 WinRT error: ") +
            error +
            "\n");

        write_failure(
            stage,
            error);

    } catch (const Ort::Exception& e) {
        log_output(
            std::string(
                "[xllama] vision-v2 ORT error: ") +
            e.what() +
            "\n");

        write_failure(
            stage,
            e.what());

    } catch (const std::exception& e) {
        log_output(
            std::string(
                "[xllama] vision-v2 error: ") +
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
