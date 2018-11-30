"""
(c) @thomaswilley, 2018
ONNXManager.py: A ModelManager with the ability to save/load
in the Open Neural Network Exchange Format (ONNX - https://onnx.ai)
"""

from simpledl.ModelManager import ModelManager
import onnx
from onnx import helper, numpy_helper, TensorProto

class ONNXManager(ModelManager):

    def _make_nodes_for_layer(self, layer_num):
        "create ONNX nodes for layer_num layer in the network"
        l_output = len(self.model['shape'])-1
        dim_output = self.model['shape'][::-1][0]
        assert((layer_num > 0) and (layer_num <= l_output))

        op = "Relu" if 'relu' in self.model['activation{}'.format(layer_num)].__name__ else "Sigmoid"

        input_name = 'X' if layer_num == 1 else 'A{}'.format(layer_num-1)

        nodes = [
            helper.make_node('MatMul',
                ['W{}'.format(layer_num), input_name],
                ['_Z{}'.format(layer_num)]),
            helper.make_node('Add',
                ['_Z{}'.format(layer_num), 'B{}'.format(layer_num)],
                ['Z{}'.format(layer_num)]),
            helper.make_node(op,
                ['Z{}'.format(layer_num)],
                ['A{}'.format(layer_num)]),
        ]

        # We're basically hard wiring this to DL Trainer's implementation initially...
        if layer_num == l_output:
            if dim_output > 1:
                nodes.append(helper.make_node("ArgMax", ['A{}'.format(layer_num)], ['Y']))
            else:
                nodes.append(helper.make_node("Greater", ['A{}'.format(layer_num)], [0.5]))

        return nodes

    def _make_nodes_for_input(self, layer_num):
        "create ONNX input nodes so the graph knows which initializers are required"
        input_nodes = [
            helper.make_tensor_value_info('W{}'.format(layer_num),
                TensorProto.DOUBLE, self.model['w{}'.format(layer_num)].shape),
            helper.make_tensor_value_info('B{}'.format(layer_num),
                TensorProto.DOUBLE, self.model['b{}'.format(layer_num)].shape),
        ]
        return input_nodes

    def make_graph(self, name="simpledl model (https://github.com/thomaswilley/pysimpledl)"):
        "create the ONNX graph representation of the ModelManager model"
        nodes = []
        l_output = len(self.model['shape'])-1
        dim_input = self.model['shape'][0]

        # the first input node is literally the input, X; rest added by layer..
        inputs = [helper.make_tensor_value_info('X' , TensorProto.DOUBLE, [dim_input, 1]),]

        initializer = []
        for l in range(l_output):
            layer = l + 1
            nodes += self._make_nodes_for_layer(layer)
            inputs += self._make_nodes_for_input(layer)
            initializer += [
                numpy_helper.from_array(self.model['w{}'.format(layer)], 'W{}'.format(layer)),
                numpy_helper.from_array(self.model['b{}'.format(layer)], 'B{}'.format(layer)),
            ]

        # TODO: multi-dimensional outputs
        outputs = [helper.make_tensor_value_info('Y', TensorProto.DOUBLE, [1]),]

        # Just create and return the ONNX graph; leave it to the caller to enforce validity
        graph_proto = helper.make_graph(nodes, name, inputs, outputs, initializer)

        return graph_proto

    def make_model(self, graph_name="simpledl model (https://github.com/thomaswilley/pysimpledl)",
            producer_name=None):
        "create the overall ONNX model representation of the Model"
        onnx_graph = self.make_graph(graph_name)
        producer_name = producer_name if producer_name else graph_name
        onnx_model = helper.make_model(onnx_graph, producer_name=producer_name)
        return onnx_model
