
import onnx
from typing import Iterable

def print_tensor_data(initializer: onnx.TensorProto) -> None:

    if initializer.data_type == onnx.TensorProto.DataType.FLOAT:
        print(initializer.float_data)
    elif initializer.data_type == onnx.TensorProto.DataType.INT32:
        print(initializer.int32_data)
    elif initializer.data_type == onnx.TensorProto.DataType.INT64:
        print(initializer.int64_data)
    elif initializer.data_type == onnx.TensorProto.DataType.DOUBLE:
        print(initializer.double_data)
    elif initializer.data_type == onnx.TensorProto.DataType.UINT64:
        print(initializer.uint64_data)
    else:
        raise NotImplementedError

    return


def dims_prod(dims: Iterable) -> int:

    prod = 1
    for dim in dims:
        prod *= dim

    return prod

def load_model(model_path):
    model = onnx.load(model_path)
    print(f"check {model_path}")
    # onnx.checker.check_model(model)

    graph_dict = {}
    link_in_dict = {}
    link_out_dict = {}
    graph_def = model.graph
    # Modify nodes
    nodes = graph_def.node
    for node in nodes:
        # print(node)
        graph_dict[node.name] = node
        for i in node.input:
            if i not in link_in_dict.keys():
                link_in_dict[i] = []
            link_in_dict[i].append(node.op_type)
        for i in node.output:
            if i not in link_out_dict.keys():
                link_out_dict[i] = []
            link_out_dict[i].append(node.op_type)

    inputs = graph_def.input
    for graph_input in inputs:
        input_shape = []
        for d in graph_input.type.tensor_type.shape.dim:
            if d.dim_value == 0:
                input_shape.append(None)
            else:
                input_shape.append(d.dim_value)
        # print(
        #   f"Input Name: {graph_input.name}, Input Data Type: {graph_input.type.tensor_type.elem_type}, Input Shape: {input_shape}"
        # )

    outputs = graph_def.output
    for graph_output in outputs:
        output_shape = []
        for d in graph_output.type.tensor_type.shape.dim:
            if d.dim_value == 0:
                output_shape.append(None)
            else:
                output_shape.append(d.dim_value)
        # print(
        #     f"Output Name: {graph_output.name}, Output Data Type: {graph_output.type.tensor_type.elem_type}, Output Shape: {output_shape}"
        # )

    # To modify inputs and outputs, we would rather create new inputs and outputs.
    # Using onnx.helper.make_tensor_value_info and onnx.helper.make_model

    # onnx.checker.check_model(model)
    # onnx.save(model, "convnets_modified.onnx")
    return graph_dict, link_in_dict, link_out_dict

def model_compare(g0, li0, lo0, g1, li1, lo1):
    diff_res = dict()
    for name, node in g0.items():
        if node.op_type in ["DequantizeLinear", "QuantizeLinear"]:
            continue

        if not name in g1.keys():
            if name not in diff_res.keys():
                diff_res[name] = [node.op_type]
            diff_res[name].append("not in")
            continue
        
        node1 = g1[name]
        for i0, i1 in zip(node.input, node1.input):
            if li0[i0] == li1[i1]:
                continue
            if name not in diff_res.keys():
                diff_res[name] = [node.op_type]
            diff_res[name].append(f"In {i0}({li0[i0]}) != {i1}({li1[i1]})")
        for i0, i1 in zip(node.output, node1.output):
            if lo0[i0] == lo1[i1]:
                continue
            if name not in diff_res.keys():
                diff_res[name] = [node.op_type]
            diff_res[name].append(f"Out {i0}({lo0[i0]}) != {i1}({lo1[i1]})")

    return diff_res

if __name__ == "__main__":
    m0 = "/data/home/vacingfang/RVM_old/vmodel/Output/video.onnx"
    m1 = "/data/home/vacingfang/RVM_old/RobustVideoMatting_onnx/model_seg_sim.onnx"
    g0, li0, lo0 = load_model(m0)
    g1, li1, lo1 = load_model(m1)
    diffs = model_compare(g0, li0, lo0, g1, li1, lo1)
    print(f"total nodes: {len(g0), len(g1)}, diff nodes: {len(diffs)}")
    for name, value in diffs.items():
        print(f"[{name}]: {value[0]}")
        for v in value[1:]:
            print(f"\t{v}")
