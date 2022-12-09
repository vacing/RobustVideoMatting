import onnx
from typing import Iterable
from onnx import helper

def process_initializer(model):
    dequant_outs = []
    dequant_in_out = {}
    graph_def = model.graph
    nodes = graph_def.node
    for node in nodes:
        if node.op_type in ["DequantizeLinear"]:
            for out in node.output:
                dequant_outs.append(out)
                dequant_in_out[out] = node.input[0]

    tensor_in = {}
    for info in graph_def.initializer:
        print(info)
        break
    return model

    # Modify initializer
    value_info = graph_def.value_info
    new_infos = []
    rm_infos = []
    for info in graph_def.value_info:
        if info.name in ["PPQ_Variable_1404", "PPQ_Variable_1401"]:
            print(info)

        if info.name not in dequant_outs:
            continue

        # skip static initializer as input
        in_name = dequant_in_out[info.name]
        if in_name not in tensor_in.keys():
            continue

        in_tensor = tensor_in[in_name]
        shape = []
        for dim in in_tensor.type.tensor_type.shape.dim:
            shape.append(dim.dim_value)
        
        if info.name in ["PPQ_Variable_1404"]:
            print(in_tensor)
            print("shape", shape)
        info2 = helper.make_tensor_value_info(info.name, info.type.tensor_type.elem_type, shape=shape)
        rm_infos.append(info)
        new_infos.append(info2)
        if info2.name in ["PPQ_Variable_1404"]:
            print(info2)

    for info in rm_infos:
        value_info.remove(info)
    for info in new_infos:
        value_info.append(info)

    print("-" * 60)
    for info in graph_def.value_info:
        if info.name in ["PPQ_Variable_1404"]:
            print(info)
    
    return model

def process_dequant(model):
    dequant_outs = []
    dequant_in_out = {}
    graph_def = model.graph
    nodes = graph_def.node
    for node in nodes:
        if node.op_type in ["DequantizeLinear"]:
            for out in node.output:
                dequant_outs.append(out)
                dequant_in_out[out] = node.input[0]

    tensor_in = {}
    for info in graph_def.value_info:
        tensor_in[info.name] = info

    # Modify initializer
    value_info = graph_def.value_info
    new_infos = []
    rm_infos = []
    for info in graph_def.value_info:
        if info.name in ["PPQ_Variable_1404", "PPQ_Variable_1401"]:
            print(info)

        if info.name not in dequant_outs:
            continue

        # skip static initializer as input
        in_name = dequant_in_out[info.name]
        if in_name not in tensor_in.keys():
            continue

        in_tensor = tensor_in[in_name]
        shape = []
        for dim in in_tensor.type.tensor_type.shape.dim:
            shape.append(dim.dim_value)
        
        if info.name in ["PPQ_Variable_1404"]:
            print(in_tensor)
            print("shape", shape)
        info2 = helper.make_tensor_value_info(info.name, info.type.tensor_type.elem_type, shape=shape)
        rm_infos.append(info)
        new_infos.append(info2)
        if info2.name in ["PPQ_Variable_1404"]:
            print(info2)

    for info in rm_infos:
        value_info.remove(info)
    for info in new_infos:
        value_info.append(info)

    print("-" * 60)
    for info in graph_def.value_info:
        if info.name in ["PPQ_Variable_1404"]:
            print(info)
    
    return model

def process_quant(model):
    quant_outs = []
    quant_in_out = {}
    # Modify nodes
    graph_def = model.graph
    nodes = graph_def.node
    for node in nodes:
        if node.op_type in ["QuantizeLinear"]:
            for out in node.output:
                quant_outs.append(out)
                quant_in_out[out] = node.input[0]

    tensor_in = {}
    for info in graph_def.value_info:
        tensor_in[info.name] = info

    static_ins = {
        "r4i": [1, 64, 10, 12],
        "r3i": [1, 40, 20, 23],
        "r2i": [1, 20, 40, 45],
        "r1i": [1, 16, 80, 90],
    }

    # Modify initializer
    value_info = graph_def.value_info
    new_infos = []
    rm_infos = []
    for info in graph_def.value_info:
        if info.name in ["PPQ_Variable_1401"]:
            print(info)

        if info.name not in quant_outs:
            continue

        in_name = quant_in_out[info.name]
        in_tensor = tensor_in[in_name]
        shape = []
        if in_tensor.name in static_ins.keys():
            shape = static_ins[in_tensor.name]
        else:
            for dim in in_tensor.type.tensor_type.shape.dim:
                shape.append(dim.dim_value)
        
        if info.name in ["PPQ_Variable_1401"]:
            print("shape", shape)
        info2 = helper.make_tensor_value_info(info.name, onnx.TensorProto.INT8, shape=shape)

        rm_infos.append(info)
        new_infos.append(info2)

    for info in rm_infos:
        value_info.remove(info)
    for info in new_infos:
        value_info.append(info)

    print("-" * 60)
    for info in graph_def.value_info:
        if info.name in ["r1i"]:
            print(info)
    
    return model

if __name__ == "__main__":
    m0 = "/data/home/vacingfang/RVM_old/RobustVideoMatting_onnx/model/model_seg_sim_dr1_ppq.onnx"
    model = onnx.load(m0)
    # print(model)
    # onnx.checker.check_model(model)
    for i in range(3):
        model = process_quant(model)
        print("*" * 120)
        model = process_dequant(model)
        print("*" * 120)
    # model = process_initializer(model)
    # onnx.checker.check_model(model)
    onnx.save(model, "convnets_modified.onnx")

