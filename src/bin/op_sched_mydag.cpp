#include <tensorflow/lite/simple_memory_arena.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <hmcos/sched/life.hpp>
#include <hmcos/sched/pass.hpp>
#include <hmcos/sched/plan.hpp>
#include <hmcos/sched/sched.hpp>
#include <hmcos/util/viz.hpp>

#include <nlohmann/json.hpp>

using namespace hmcos;
using namespace std::chrono;
using json = nlohmann::json;

#define TIME_CODE(code)                                                        \
    {                                                                          \
        auto _begin = system_clock::now();                                     \
        code;                                                                  \
        auto _dur =                                                            \
            duration_cast<milliseconds>(system_clock::now() - _begin).count(); \
        LOG(INFO) << fmt::format("{} ms", _dur);                               \
    }

static uint64_t computeArenaSize(const LifetimeStat &stat) {
    std::vector<tflite::ArenaAllocWithUsageInterval> allocs(stat.values.size());
    TfLiteContext ctx;
    tflite::SimpleMemoryArena arena(64);
    for (auto [i, val] : EnumRange(stat.values))
        arena.Allocate(&ctx, 64, val.value->type.Size(), i, val.gen,
                       val.kill - 1, &allocs[i]);
    return arena.RequiredBufferSize();
}

static void dumpSchedule(const std::vector<OpRef> &sched,
                         const std::string &path) {
    json j;
    for (auto &op : sched) {
        j.push_back(op->name);
    }
    std::ofstream ofs(path);
    ofs << j.dump(4);
    ofs.close();
}

int main(int argc, char const *argv[]) {
    // Initialize glog
    FLAGS_minloglevel = 0;
    google::LogToStderr();
    google::InitGoogleLogging(argv[0]);

    // // Build compitation graph from ONNX model
    // std::ifstream ifs(argv[1], std::ifstream::binary);
    // onnx::ModelProto model;
    // model.ParseFromIstream(&ifs);
    // ifs.close();
    // Graph graph(model, std::filesystem::path(argv[1]).stem().string());
    // model.Clear();

    // parse the json file from the first argument
    std::ifstream dag_ifs(argv[1]);
    json dag_json;
    dag_ifs >> dag_json;
    dag_ifs.close();
    Graph graph(dag_json);

    // if user provide a budget, use the user-provided budget
    // otherwise, use the default budget
    static constexpr auto MAX_BUDGET = INT64_MAX / 2;
    uint64_t budget = MAX_BUDGET;
    if (argc >= 4) {
        budget = std::stoull(argv[3]);
    }

    // Schedule hierarchical graph
    std::vector<OpRef> sched;
    TIME_CODE(sched = HierarchicalSchedule(graph, budget);)

    // dump the result schedule to a json file
    std::string json_path = std::filesystem::path(argv[2]).string() + "/" + graph.name + ".json";
    LOG(INFO) << "Dumping schedule to " << json_path;
    dumpSchedule(sched, json_path);

    LOG(INFO) << "HMCOS Peak: " << EstimatePeak(sched, graph.inputs) << " Byte";
    LOG(INFO) << "HMCOS Arena Size: " << computeArenaSize(ComputeLifetime(sched, graph)) << " Byte";
    sched = ReversePostOrder(graph);
    LOG(INFO) << "RPO Peak: " << EstimatePeak(sched, graph.inputs) << " Byte";
    LOG(INFO) << "RPO Arena Size: " << computeArenaSize(ComputeLifetime(sched, graph)) << " Byte";

    return 0;
}
