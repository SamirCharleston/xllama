#include "inference-bridge.h"

#if defined(XLLAMA_UWP) && defined(XLLAMA_USE_LLAMA)

    #include <llama.h>
    #include <mtmd-helper.h>
    #include <mtmd.h>

    #include <winrt/Windows.Foundation.h>
    #include <winrt/Windows.Graphics.Imaging.h>
    #include <winrt/Windows.Storage.Streams.h>
    #include <winrt/Windows.Storage.h>

    #include <chrono>
    #include <cstdio>
    #include <memory>
    #include <sstream>
    #include <stdexcept>
    #include <string>
    #include <vector>

    #include "xllama/json_utils.h"
    #include "xllama/path_utils.h"
    #include "xllama/platform.h"
    #include "xllama/utf8_utils.h"

namespace xllama::bridge {
namespace {

using Clock = std::chrono::steady_clock;

long long elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

void write_text(const std::string& name, const std::string& contents) {
    const auto path = utf8_to_wstring(resolve_local_path(name));
    FILE* fp = _wfopen(path.c_str(), L"wb");
    if (!fp)
        throw std::runtime_error("cannot write " + name);
    fwrite(contents.data(), 1, contents.size(), fp);
    fclose(fp);
}

struct Image {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<unsigned char> rgb;
};

Image load_image() {
    using namespace winrt::Windows::Graphics::Imaging;
    using namespace winrt::Windows::Storage;

    auto file = ApplicationData::Current().LocalFolder().GetFileAsync(L"vision-input.jpg").get();
    auto stream = file.OpenAsync(FileAccessMode::Read).get();
    auto decoder = BitmapDecoder::CreateAsync(stream).get();

    Image image;
    image.width = decoder.PixelWidth();
    image.height = decoder.PixelHeight();
    if (!image.width || !image.height)
        throw std::runtime_error("invalid image dimensions");

    auto provider =
        decoder
            .GetPixelDataAsync(BitmapPixelFormat::Rgba8, BitmapAlphaMode::Ignore, BitmapTransform{},
                               ExifOrientationMode::RespectExifOrientation,
                               ColorManagementMode::ColorManageToSRgb)
            .get();
    auto pixels = provider.DetachPixelData();
    const size_t count = static_cast<size_t>(image.width) * image.height;
    if (pixels.size() != count * 4)
        throw std::runtime_error("unexpected decoded pixel count");

    image.rgb.resize(count * 3);
    for (size_t i = 0; i < count; ++i) {
        image.rgb[i * 3] = pixels[i * 4];
        image.rgb[i * 3 + 1] = pixels[i * 4 + 1];
        image.rgb[i * 3 + 2] = pixels[i * 4 + 2];
    }
    return image;
}

std::string format_prompt(const llama_model* model) {
    const std::string content = std::string(mtmd_default_marker()) +
                                "Describe the main subject of this image in one short sentence.";
    const llama_chat_message message{"user", content.c_str()};
    const char* tmpl = llama_model_chat_template(model, nullptr);
    const int32_t needed = llama_chat_apply_template(tmpl, &message, 1, true, nullptr, 0);
    if (needed <= 0)
        throw std::runtime_error("model chat template is unsupported");
    std::string prompt(static_cast<size_t>(needed), '\0');
    if (llama_chat_apply_template(tmpl, &message, 1, true, prompt.data(), needed) != needed)
        throw std::runtime_error("failed to format vision prompt");
    return prompt;
}

std::string token_piece(const llama_vocab* vocab, llama_token token) {
    char small[128];
    int32_t n = llama_token_to_piece(vocab, token, small, sizeof(small), 0, false);
    if (n >= 0)
        return std::string(small, static_cast<size_t>(n));
    std::vector<char> large(static_cast<size_t>(-n));
    n = llama_token_to_piece(vocab, token, large.data(), static_cast<int32_t>(large.size()), 0,
                             false);
    if (n < 0)
        throw std::runtime_error("failed to decode output token");
    return std::string(large.data(), static_cast<size_t>(n));
}

struct Batch {
    llama_batch value = llama_batch_init(1, 0, 1);
    ~Batch() {
        llama_batch_free(value);
    }
};

} // namespace

void run_vision_caption() {
    set_cwd_to_local_folder();
    std::string stage = "initializing";

    try {
        stage = "image_decode";
        const auto image_start = Clock::now();
        Image image = load_image();
        const auto image_end = Clock::now();

        stage = "model_load";
        const auto load_start = Clock::now();
        llama_backend_init();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        model_params.load_mode = LLAMA_LOAD_MODE_NONE;
        const std::string model_path =
            resolve_local_path("vision-models\\SmolVLM-256M-Instruct-Q8_0.gguf");
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(model_path.c_str(), model_params), llama_model_free);
        if (!model)
            throw std::runtime_error("failed to load SmolVLM text model");

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = 2048;
        context_params.n_batch = 512;
        context_params.n_ubatch = 512;
        context_params.n_threads = 6;
        context_params.n_threads_batch = 6;
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model.get(), context_params), llama_free);
        if (!context)
            throw std::runtime_error("failed to create SmolVLM context");

        mtmd_context_params vision_params = mtmd_context_params_default();
        vision_params.use_gpu = false;
        vision_params.n_threads = 6;
        vision_params.warmup = false;
        const std::string projector_path =
            resolve_local_path("vision-models\\mmproj-SmolVLM-256M-Instruct-Q8_0.gguf");
        std::unique_ptr<mtmd_context, decltype(&mtmd_free)> vision(
            mtmd_init_from_file(projector_path.c_str(), model.get(), vision_params), mtmd_free);
        if (!vision || !mtmd_support_vision(vision.get()))
            throw std::runtime_error("failed to load SmolVLM vision projector");
        const auto load_end = Clock::now();

        stage = "vision_encode";
        const std::string prompt = format_prompt(model.get());
        std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(
            mtmd_bitmap_init(image.width, image.height, image.rgb.data()), mtmd_bitmap_free);
        std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> chunks(
            mtmd_input_chunks_init(), mtmd_input_chunks_free);
        if (!bitmap || !chunks)
            throw std::runtime_error("failed to allocate vision input");
        const mtmd_bitmap* bitmaps[] = {bitmap.get()};
        const mtmd_input_text input{prompt.data(), prompt.size(), true, true};
        if (mtmd_tokenize(vision.get(), chunks.get(), &input, bitmaps, 1) != 0)
            throw std::runtime_error("failed to tokenize image and prompt");

        const auto encode_start = Clock::now();
        llama_pos n_past = 0;
        if (mtmd_helper_eval_chunks(vision.get(), context.get(), chunks.get(), 0, 0, 512, true,
                                    &n_past) != 0)
            throw std::runtime_error("failed to encode image and prompt");
        const auto encode_end = Clock::now();

        stage = "text_generation";
        std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(
            llama_sampler_init_greedy(), llama_sampler_free);
        if (!sampler)
            throw std::runtime_error("failed to create greedy sampler");
        Batch batch;
        if (!batch.value.token)
            throw std::runtime_error("failed to allocate decode batch");

        const llama_vocab* vocab = llama_model_get_vocab(model.get());
        const auto generation_start = Clock::now();
        std::string description;
        for (int i = 0; i < 64; ++i) {
            const llama_token token = llama_sampler_sample(sampler.get(), context.get(), -1);
            llama_sampler_accept(sampler.get(), token);
            if (llama_vocab_is_eog(vocab, token))
                break;
            description += token_piece(vocab, token);

            batch.value.n_tokens = 1;
            batch.value.token[0] = token;
            batch.value.pos[0] = n_past++;
            batch.value.n_seq_id[0] = 1;
            batch.value.seq_id[0][0] = 0;
            batch.value.logits[0] = 1;
            if (llama_decode(context.get(), batch.value) != 0)
                throw std::runtime_error("failed to decode generated token");
        }
        const auto generation_end = Clock::now();
        if (description.empty())
            throw std::runtime_error("SmolVLM returned an empty description");

        std::ostringstream result;
        result << "{\n"
               << "  \"spike\": \"vision-caption-v0\",\n"
               << "  \"backend\": \"llama.cpp-mtmd-cpu\",\n"
               << "  \"model\": \"SmolVLM-256M-Instruct-Q8_0\",\n"
               << "  \"status\": \"PASS\",\n"
               << "  \"stage\": \"vision_caption_completed\",\n"
               << "  \"image\": \"vision-input.jpg\",\n"
               << "  \"source_width\": " << image.width << ",\n"
               << "  \"source_height\": " << image.height << ",\n"
               << "  \"description\": \"" << json_escape(description) << "\",\n"
               << "  \"image_decode_ms\": " << elapsed_ms(image_start, image_end) << ",\n"
               << "  \"model_load_ms\": " << elapsed_ms(load_start, load_end) << ",\n"
               << "  \"vision_encode_ms\": " << elapsed_ms(encode_start, encode_end) << ",\n"
               << "  \"generation_ms\": " << elapsed_ms(generation_start, generation_end) << "\n"
               << "}\n";
        write_text("vision-caption-result.json", result.str());
        write_text("vision-caption.done", "ok");
        log_output("[xllama] vision caption completed\n");
    } catch (const winrt::hresult_error& error) {
        const std::string message = wstring_to_utf8(error.message().c_str());
        write_text("vision-caption-result.json", "{\"status\":\"FAIL\",\"stage\":\"" +
                                                     json_escape(stage) + "\",\"error\":\"" +
                                                     json_escape(message) + "\"}\n");
        write_text("vision-caption.done", "error");
    } catch (const std::exception& error) {
        write_text("vision-caption-result.json", "{\"status\":\"FAIL\",\"stage\":\"" +
                                                     json_escape(stage) + "\",\"error\":\"" +
                                                     json_escape(error.what()) + "\"}\n");
        write_text("vision-caption.done", "error");
    }
}

} // namespace xllama::bridge

#else

namespace xllama::bridge {
void run_vision_caption() {}
} // namespace xllama::bridge

#endif
