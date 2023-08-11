#include <hmcos/core/graph.hpp>
#include <hmcos/util/fmt.hpp>
#include <hmcos/util/viz.hpp>
#include <unordered_map>

#include <nlohmann/json.hpp>

namespace hmcos {

Graph::Graph(const onnx::ModelProto &model, const std::string &name) {
    // Create name of this graph
    auto &graph = model.graph();
    this->name = name.size() == 0 ? graph.name() : name;

    // Build name-value map
    std::unordered_map<std::string, ValueRef> nameToVal;
    // Inputs
    for (auto &info : graph.input()) {
        auto val = std::make_shared<Value>(Value::CreateInput(info));
        auto in = std::make_shared<Input>(val);
        val->input = in;
        inputs.push_back(in);
        nameToVal.insert({info.name(), val});
        LOG(INFO) << "graph input: " << info.name() << " " << info.type().tensor_type().elem_type();
    }
    // Outputs
    for (auto &info : graph.output()) {
        auto val = std::make_shared<Value>(Value::CreateResult(info));
        outputs.push_back(std::make_shared<Output>(val));
        nameToVal.insert({info.name(), val});
        LOG(INFO) << "graph output: " << info.name() << " " << info.type().tensor_type().elem_type();
    }
    // Parameters
    for (auto &tensor : graph.initializer()) {
        auto val = std::make_shared<Value>(Value::CreateParam(tensor));
        params.push_back(val);
        nameToVal.insert({tensor.name(), val});
    }
    // Intermediates
    for (auto &info : graph.value_info()) {
        auto val = std::make_shared<Value>(Value::CreateResult(info));
        nameToVal.insert({info.name(), val});
        LOG(INFO) << "graph intermediates: " << info.name() << " " << info.type().tensor_type().elem_type();
    }

    // Build ops
    for (auto &node : graph.node()) {
        auto op = std::make_shared<Op>(&node);
        LOG(INFO) << "op: " << op->name << " type:" << op->type;
        // Input values
        for (auto &in : node.input()) {
            if (!Contains(nameToVal, in))
                LOG(FATAL) << fmt::format(
                    "Cannot find information of value {}.", in);
            auto &inVal = nameToVal[in];
            op->inputs.push_back(inVal);
            inVal->uses.push_back(op);
        }
        // Output values
        for (auto &out : node.output()) {
            if (!Contains(nameToVal, out))
                LOG(FATAL) << fmt::format(
                    "Cannot find information of value {}.", out);
            auto &outVal = nameToVal[out];
            op->outputs.push_back(outVal);
            outVal->def = op;
        }
        ops.push_back(op);
    }

    // Connect vertices
    ConnectVerts();
}

Graph::Graph(const nlohmann::json &dag_json) {
    // Create the graph from the json file
    this->name = dag_json["name"].get<std::string>();

    // Build name-value map
    std::unordered_map<std::string, ValueRef> nameToVal;
    
    // Inputs
    LOG(INFO) << "creating dummy graph inputs";
    for (auto &tensor : dag_json["dummy_input_tensors"]) {
        auto val = std::make_shared<Value>(
            Value::CreateInput(tensor.get<std::string>(), 0));
        auto in = std::make_shared<Input>(val);
        val->input = in;
        inputs.push_back(in);
        nameToVal.insert({tensor.get<std::string>(), val});
        LOG(INFO) << "graph input: " << val->name;
    }
    // Outputs
    LOG(INFO) << "creating outputs";
    for (auto &tensor : dag_json["graph_output_tensors"]) {
        auto val = std::make_shared<Value>(
            Value::CreateResult(
                tensor.get<std::string>(), dag_json["tensor_sizes"][tensor].get<int>()));
        outputs.push_back(std::make_shared<Output>(val));
        nameToVal.insert({tensor.get<std::string>(), val});
        LOG(INFO) << "graph output: " << val->name;
    }

    // Parameters
    // Since HMCOS does not support scheduling with parameters, we don't create 
    // the parameters

    // Intermediates
    LOG(INFO) << "creating tensors";
    for (auto &tensor : dag_json["tensor_list"]) {
        // skip the input and output tensors that have been added before
        if (Contains(nameToVal, tensor.get<std::string>())) continue;
        auto val = std::make_shared<Value>(
            Value::CreateResult(
                tensor.get<std::string>(), dag_json["tensor_sizes"][tensor].get<int>()));
        nameToVal.insert({tensor.get<std::string>(), val});
    }

    LOG(INFO) << "creating ops";
    // Build ops
    for (auto &node : dag_json["dag"]) {
        std::string node_type;
        node_type = "unknown";
        auto op = std::make_shared<Op>(
            node["name"].get<std::string>(), node_type);

        LOG(INFO) << "\t created ops.." << op->name;
        // skip input nodes
        if (node["input_nodes"].size() == 0) {
            // add the dummy input tensors to these input tenosr nodes
            std::string dummy_tensor = "dummy_" + node["name"].get<std::string>();
            if (!Contains(nameToVal, dummy_tensor))
                LOG(FATAL) << fmt::format(
                    "Cannot find information of value {}.", dummy_tensor);
            auto &inVal = nameToVal[dummy_tensor];
            op->inputs.push_back(inVal);
            inVal->uses.push_back(op);
        } else{
            // Input tensors
            for (auto &in_tensor : node["input_tensors"]) {
                if (!Contains(nameToVal, in_tensor.get<std::string>()))
                    LOG(FATAL) << fmt::format(
                        "Cannot find information of value {}.", in_tensor);
                auto &inVal = nameToVal[in_tensor.get<std::string>()];
                op->inputs.push_back(inVal);
                inVal->uses.push_back(op);
            }
        }
        
        // Output tensors
        for (auto &out_tensor : node["output_tensors"]) {
            if (!Contains(nameToVal, out_tensor.get<std::string>()))
                LOG(FATAL) << fmt::format(
                    "Cannot find information of value {}.", out_tensor);
            auto &outVal = nameToVal[out_tensor.get<std::string>()];
            op->outputs.push_back(outVal);
            outVal->def = op;
        }
        ops.push_back(op);
    }

    LOG(INFO) << "connecting vertices";
    // Connect vertices
    ConnectVerts();
    
}

void Graph::ConnectVerts() {
    for (auto &op : ops) {
        for (auto &in : op->inputs) {
            if (in->kind == ValueKind::PARAM) continue;
            Vertex::Connect(in->Vertex(), op);
        }
    }
    for (auto &out : outputs) Vertex::Connect(out->value->Vertex(), out);
}

VertexRef VertexCloner::VisitInput(const InputRef &input) {
    auto newVal = VisitValue(input->value);
    auto newInput = std::make_shared<Input>(newVal);
    newVal->input = newInput;
    return newInput;
}

VertexRef VertexCloner::VisitOutput(const OutputRef &output) {
    auto &val = output->value;
    auto newVal = VisitValue(val);
    Visit(val->Vertex());
    return std::make_shared<Output>(newVal);
}

VertexRef VertexCloner::VisitOp(const OpRef &op) {
    auto newOp = std::make_shared<Op>(*op);
    for (auto &in : op->inputs) {
        auto newIn = VisitValue(in);
        newOp->inputs.push_back(newIn);
        newIn->uses.push_back(newOp);
        if (in->kind != ValueKind::PARAM) Visit(in->Vertex());
    }
    for (auto &out : op->outputs) {
        auto newOut = VisitValue(out);
        newOp->outputs.push_back(newOut);
        newOut->def = newOp;
    }
    return newOp;
}

ValueRef VertexCloner::VisitValue(const ValueRef &value) {
    if (Contains(valueMap, value)) return valueMap[value];
    auto newVal = std::make_shared<Value>(*value);
    valueMap.insert({value, newVal});
    return newVal;
}

class GraphCloner : public VertexCloner {
public:
    GraphCloner(const Graph &src, Graph &dst) : src(src), dst(dst) {}

    void Clone() {
        dst.name = src.name;
        for (auto &out : src.outputs) Visit(out);
        dst.ConnectVerts();
    }

    VertexRef VisitInput(const InputRef &input) override {
        auto newInput = VertexCloner::VisitInput(input);
        dst.inputs.push_back(As<Input>(newInput));
        return newInput;
    }

    VertexRef VisitOutput(const OutputRef &output) override {
        auto newOutput = VertexCloner::VisitOutput(output);
        dst.outputs.push_back(As<Output>(newOutput));
        return newOutput;
    }

    VertexRef VisitOp(const OpRef &op) override {
        auto newOp = VertexCloner::VisitOp(op);
        dst.ops.push_back(As<Op>(newOp));
        return op;
    }

    ValueRef VisitValue(const ValueRef &value) override {
        if (Contains(valueMap, value)) return valueMap[value];
        auto newVal = VertexCloner::VisitValue(value);
        if (newVal->kind == ValueKind::PARAM) dst.params.push_back(newVal);
        return newVal;
    }

protected:
    const Graph &src;
    Graph &dst;
};

Graph Graph::Clone() const {
    Graph dst;
    GraphCloner(*this, dst).Clone();
    return dst;
}

class SubgraphExtractor : public VertexVisitor<VertexRef, bool> {
public:
    SubgraphExtractor(const Graph &src, Graph &dst,
                      std::function<bool(OpRef)> isOutput)
        : src(src), dst(dst), isOutput(isOutput) {}

    void Extract() {
        for (auto &out : src.outputs) Visit(out, false);
        dst.ConnectVerts();
    }

    VertexRef VisitInput(const InputRef &input, bool inGraph) override {
        if (!inGraph) return nullptr;
        auto newVal = VisitValue(input->value);
        auto newInput = std::make_shared<Input>(newVal);
        newVal->input = newInput;
        dst.inputs.push_back(newInput);
        return newInput;
    }

    VertexRef VisitOutput(const OutputRef &output, bool) override {
        Visit(output->value->Vertex(), false);
        return nullptr;
    }

    VertexRef VisitOp(const OpRef &op, bool inGraph) override {
        auto isOut = this->isOutput(op);
        inGraph |= isOut;
        if (inGraph) {
            auto newOp = std::make_shared<Op>(*op);
            dst.ops.push_back(newOp);
            for (auto &in : op->inputs) {
                auto newIn = VisitValue(in);
                newOp->inputs.push_back(newIn);
                newIn->uses.push_back(newOp);
                if (in->kind != ValueKind::PARAM) Visit(in->Vertex(), true);
            }
            for (auto &out : op->outputs) {
                auto newOut = VisitValue(out);
                newOp->outputs.push_back(newOut);
                newOut->def = newOp;
                if (isOut)
                    dst.outputs.push_back(std::make_shared<Output>(newOut));
            }
            return newOp;
        } else {
            for (auto &in : op->inputs)
                if (in->kind == ValueKind::RESULT) Visit(in->Vertex(), false);
            return nullptr;
        }
    }

    ValueRef VisitValue(const ValueRef &value) {
        if (Contains(valueMap, value)) return valueMap[value];
        auto newVal = std::make_shared<Value>(*value);
        valueMap.insert({value, newVal});
        if (newVal->kind == ValueKind::PARAM) dst.params.push_back(newVal);
        return newVal;
    }

private:
    std::unordered_map<ValueRef, ValueRef> valueMap;
    const Graph &src;
    Graph &dst;
    std::function<bool(OpRef)> isOutput;
};

Graph Graph::Subgraph(std::function<bool(const OpRef &)> isOutput,
                      const std::string &subName) const {
    Graph sub;
    sub.name = subName;
    SubgraphExtractor(*this, sub, isOutput).Extract();
    return sub;
}

void Graph::Plot(const std::string &dir, const std::string &format) const {
    // Define DOT graph
    DotCreator<VertexRef> creator(name);

    // Add vertices
    for (auto &in : inputs) creator.Node(in, in->value->name);
    for (auto &op : ops) creator.Node(op, op->type);
    for (auto &out : outputs) creator.Node(out, out->value->name);

    // Add edges
    for (auto &op : ops)
        for (auto &pred : op->preds) creator.Edge(pred.lock(), op);
    for (auto &out : outputs) creator.Edge(out->Def(), out);

    // Compile
    creator.Render(dir, format);
}

}  // namespace hmcos