import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
from pathlib import Path

fp = Path(os.path.dirname(__file__))
os.chdir(fp)


import tensorflow.keras as keras
from jinja2 import Template, Environment, FileSystemLoader
import numpy as np

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
from pathlib import Path

fp = Path(os.path.dirname(__file__))
os.chdir(fp)
from soeampc.mpcproblem import MPCQuadraticCostLxLu

def np2cpp( arr ):
    if arr.ndim == 1:
        return '{'+', '.join(str(num) for num in arr)+'}'
    if arr.ndim == 2:
        return '{'+','.join('{'+','.join(str(num) for num in arr[:,i])+'}' for i in range(len(arr[0,:]))) + '}'
        
    

if __name__=='__main__':
    model_path = Path("/share/mihaela-larisa.clement/soeampc-data/models").joinpath('2-50-50-10_mu=1.00_20230106-170148')
    model = keras.models.load_model(model_path)
    
    # model = keras.Sequential(
    #     [
    #         keras.layers.Normalization(input_shape=[2,],axis=None),
    #         keras.layers.Dense(3, activation="tanh"),
    #         keras.layers.Dense(6, activation="linear"),
    #         keras.layers.Reshape((2,3))
    #     ]
    # )
    # model.layers[0].adapt(np.array([[1,2],[3,4],[5,6],[7,8]]))
    
    jinja_environment = Environment(loader=FileSystemLoader('benchmark/templates'))
    template_hpp = jinja_environment.get_template('neural_network.hpp.jinja')
    # template_cpp = jinja_environment.get_template('neural_network.cpp.jinja')
    template_main = jinja_environment.get_template('main.cpp.jinja')
    
    # in normalization calculated as (input-offset)/scale
    # in this case, initial layer of model is normalization layer
    assert type(model.layers[0]) == keras.layers.Normalization
    input_size = model.layers[0].input.shape[1]
    input_offset = model.layers[0].get_weights()[0]
    input_scale  = model.layers[0].get_weights()[1]
    
    input_offset = np2cpp(np.array([input_offset for i in range(input_size)]))
    input_scale  = np2cpp(np.sqrt(np.array([input_scale for i in range(input_size)])))
    
    
    if len(model.layers[-1].output.shape)==3:
        output_size     = model.layers[-1].output.shape[2]
        output_horizont = model.layers[-1].output.shape[1]
    else:
        output_size = model.layers[-1].output.shape[1]
        output_horizont = 1
    
    # in denormalization, output is calculated as (output * scale + offset)
    # in this case, network is trained without output normalization
    output_scale =  np2cpp(np.array([1 for i in range(output_size)]))
    output_offset = np2cpp(np.array([0 for i in range(output_size)]))
    
    activation = {keras.activations.get("tanh"): "TANH",
                  keras.activations.get("linear"): "LINEAR"
                  }
    
    layer_list = [{
            "activation":  activation[layer.activation],
            "input_size":   layer.input.shape[1],
            "output_size":  layer.output.shape[1],
            "weights":     np2cpp(layer.get_weights()[0]),
            "bias":        np2cpp(layer.get_weights()[1]),
    } for layer in model.layers[1:-1]]
    
    test_inputs = np.random.rand(input_size)
    test_data = np2cpp(test_inputs)
    test_outputs = model(test_inputs).numpy().reshape(output_horizont,output_size)
    test_result = np2cpp(test_outputs)
    
    print(f"{test_inputs=}")
    print(f"{test_outputs=}")
    
    template_values = {
        'name':          'stirtank_nn',
        'input_size':    input_size,
        'input_offset':  input_offset,
        'input_scale':   input_scale,
        'output_size':   output_size,
        'output_horizont':   output_horizont,
        'output_offset': output_offset,
        'output_scale':  output_scale,
        'layer_list':    layer_list,
        'test_data':     test_data,
        'test_result':   test_result
        }


    output = template_hpp.render(template_values)
    with open('benchmark/neural_network.hpp', 'w') as f:
        f.write(output)
    
    # output = template_cpp.render(template_values)
    # with open('embedded_nn_inference/neural_network.cpp', 'w') as f:
    #     f.write(output)
    
    mpcclass=MPCQuadraticCostLxLu
    p = model_path.joinpath("mpc_parameters")
    mpc = mpcclass.genfromtxt(p)
    
    u_min, u_max = mpc.get_uminmax()
    x_min, x_max = mpc.get_xminmax()
    
    Lx = mpc.Lx
    Lu = mpc.Lu
    
    nconstr = 5
    x0 = np.array([-1.578192466893326928e+00,1.101893171497852819e+00,-2.417856791676831207e-02,2.057156335867898811e+00,-1.769806397658753028e+00,-4.351553925093814001e-01,2.230494161571518474e-01,2.602737975852264185e+00,1.366460681094356922e-01,5.305138309769214189e+00])
        
    template_values = {
        'P': np2cpp(mpc.P.transpose()),
        'Q': np2cpp(mpc.Q.transpose()),
        'R': np2cpp(mpc.R.transpose()),
        'K': np2cpp(mpc.K.transpose()),
        'Kdelta': np2cpp(mpc.Kdelta.transpose()),
        'alpha': mpc.alpha,
        'u_min': np2cpp(u_min),
        'u_max': np2cpp(u_max),
        'Lx': np2cpp(Lx[:5, :].transpose()),
        'nconstr': nconstr,
        'x0': np2cpp(x0),
        'name': "stirtank"
    }
    output = template_main.render(template_values)
    with open('benchmark/main.cpp', 'w') as f:
        f.write(output)