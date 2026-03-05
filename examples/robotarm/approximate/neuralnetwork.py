import jax
import equinox as eqx
import math
import json
import os
import functools
import numpy as np
import h5py
import fire

def init_weight(dim_in, dim_out, key):
    stdv = 1. / math.sqrt(dim_out)
    return jax.random.uniform(key, (dim_out, dim_in)) * 2 * stdv - stdv


def init_bias(dim_out, key):
    stdv = 1. / math.sqrt(dim_out)
    return jax.random.uniform(key, (dim_out,)) * 2 * stdv - stdv

def normalize(val, scale, offset):
    return jax.numpy.divide(val-offset, scale)

def denormalize(val, scale, offset):
    return jax.numpy.multiply(val, scale)+offset

def L1(y, y_pred):
    return jax.numpy.mean(jax.numpy.abs(y-y_pred))

def L2(y, y_pred):
    return jax.numpy.mean((y - y_pred) ** 2)

# quadratic cost term
def L_quadratic_sum(y, y_pred, Q):
    return (y-y_pred).transpose() @ Q @ (y-y_pred)

# cost for a single mpc timestep
def L_mpc_stage_imitate(u, u_pred, x, x_pred, Q, R):
    return L_quadratic_sum(x, x_pred, Q) + L_quadratic_sum(u, u_pred, R)
    
# cost for a single trajectory
def L_mpc_imitate(U,U_pred,X,X_pred,Q,R,P):
    return jax.numpy.sum(
        jax.vmap(
            functools.partial(L_mpc_stage_imitate, Q=Q, R=R))(u=U, u_pred=U_pred, x=X[:-1,:], x_pred=X_pred[:-1,:])
    ) + L_quadratic_sum(X[-1,:], X_pred[-1,:], P)


def in_ellipse(x, P, alpha):
    return x.transpose() @ P @ x <= alpha**2

def in_box(x, x_min, x_max):
    return jax.numpy.all( jax.numpy.logical_and(x >= x_min,  x <= x_max ))

class LinearLayer(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    input_dim: int
    output_dim: int

    def __init__(self, input_dim, output_dim, rng_key):

        super().__init__()

        key_weight, key_bias = jax.random.split(rng_key)
        self.weight = init_weight(input_dim, output_dim, key_weight)
        self.bias = init_bias(output_dim, key_bias)

        self.input_dim = input_dim
        self.output_dim = output_dim

    def __call__(self, x):
        return self.weight @ x + self.bias

class NonLinearLayer(LinearLayer):
    def __call__(self, x):
        return self.__class__.activation_function(self.weight @ x + self.bias)


class TanhLayer(NonLinearLayer):
    activation_function = jax.nn.tanh
        
    
class ReluLayer(NonLinearLayer):
    activation_function = jax.nn.relu
    
class ReshapeLayer(eqx.Module):
    input_dim: int
    output_dim: tuple
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
    
    def __call__(self, x):
        return jax.numpy.reshape(x, self.output_dim)
    
class MLP(eqx.Module):
    layers: list
    
    def __init__(self, layers, rng_key):
        super().__init__()
        self.layers = layers
    
    def __call__(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
        else:
            return x

def make_standard_model(*, key, nx, nu, N):
   layer_keys = jax.random.split(key, 20)
   return MLP(
       layers=[
           LinearLayer(input_dim=nx+ny, output_dim=100,   rng_key=layer_keys[0]),
           ReluLayer(input_dim=100, output_dim=100,    rng_key=layer_keys[1]),
           ReluLayer(input_dim=100, output_dim=100,    rng_key=layer_keys[2]),
           TanhLayer(input_dim=100, output_dim=100,    rng_key=layer_keys[3]),
           TanhLayer(input_dim=100, output_dim=100,    rng_key=layer_keys[4]),
           LinearLayer(input_dim=100, output_dim=nu*N, rng_key=layer_keys[5]),
           ReshapeLayer(input_dim=nu*N, output_dim=(N, nu))
           ],
       rng_key=layer_keys[6]
       )
   
def make_standard_model_cartpole(*, key, nx, nu, ny, N):
   layer_keys = jax.random.split(key, 20)
   return MLP(
       layers=[
           LinearLayer(input_dim=nx+ny, output_dim=100,   rng_key=layer_keys[0]),
           ReluLayer(input_dim=100, output_dim=100,    rng_key=layer_keys[1]),
           ReluLayer(input_dim=100, output_dim=100,    rng_key=layer_keys[2]),
           ReluLayer(input_dim=100, output_dim=100,    rng_key=layer_keys[2]),
           TanhLayer(input_dim=100, output_dim=100,    rng_key=layer_keys[3]),
           TanhLayer(input_dim=100, output_dim=100,    rng_key=layer_keys[4]),
           TanhLayer(input_dim=100, output_dim=100,    rng_key=layer_keys[4]),
           ReluLayer(input_dim=100, output_dim=100,    rng_key=layer_keys[2]),
           ReluLayer(input_dim=100, output_dim=100,    rng_key=layer_keys[2]),
           ReluLayer(input_dim=100, output_dim=100,    rng_key=layer_keys[2]),
           LinearLayer(input_dim=100, output_dim=nu*N, rng_key=layer_keys[5]),
           ReshapeLayer(input_dim=nu*N, output_dim=(N, nu))
           ],
       rng_key=layer_keys[6]
       )

def make_standard_model_franka(*, key, nx, nu, ny, N):
   layer_keys = jax.random.split(key, 20)
   return MLP(
       layers=[
           LinearLayer(input_dim=nx+ny, output_dim=256,   rng_key=layer_keys[0]),
           ReluLayer(input_dim=256, output_dim=256,    rng_key=layer_keys[1]),
           ReluLayer(input_dim=256, output_dim=256,    rng_key=layer_keys[2]),
           ReluLayer(input_dim=256, output_dim=256,    rng_key=layer_keys[4]),
           ReluLayer(input_dim=256, output_dim=256,    rng_key=layer_keys[4]),
           ReluLayer(input_dim=256, output_dim=256,    rng_key=layer_keys[2]),
           ReluLayer(input_dim=256, output_dim=256,    rng_key=layer_keys[2]),
           ReluLayer(input_dim=256, output_dim=256,    rng_key=layer_keys[2]),
           ReluLayer(input_dim=256, output_dim=256,    rng_key=layer_keys[2]),
           ReluLayer(input_dim=256, output_dim=256,    rng_key=layer_keys[2]),
           LinearLayer(input_dim=256, output_dim=nu*N, rng_key=layer_keys[5]),
           ReshapeLayer(input_dim=nu*N, output_dim=(N, nu))
           ],
       rng_key=layer_keys[6]
       )

def make_model(*, key, model_fcn_name, model_hyperparams, normalization_parameters):
    model_dict = {
            "standard": make_standard_model,
            "standard_stirtank": make_standard_model,
            "standard_cartpole": make_standard_model_cartpole,
            "standard_quadcopterten": make_standard_model_cartpole,
            # "standard_franka": make_standard_model_cartpole,
            "standard_franka": make_standard_model_franka,
            }
    model_generator_fcn = model_dict[model_fcn_name]
    return model_generator_fcn(key=key,**model_hyperparams)


def save_model(filename, hyperparams, model):
    output_folder = os.path.dirname(filename)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)    
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)
        
def load_model(filename):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = make_model(key=jax.random.PRNGKey(123), **hyperparams)
        return eqx.tree_deserialise_leaves(f, model), hyperparams
    

def compute_learning_rate(step, initial_learning_rate=1e-2, drop_rate=0.8, wait_epochs=30, min_learning_rate=1e-4):
    new_learning_rate = drop_rate**( step // wait_epochs )*initial_learning_rate
    return max(new_learning_rate, min_learning_rate)


def export_model_as_h5(filename, hyperparams, model):
    nx = hyperparams['model_hyperparams']['nx']
    ny = hyperparams['model_hyperparams']['ny']
    nu = hyperparams['model_hyperparams']['nu']
    input_norm_offset =  np.concatenate([hyperparams['normalization_parameters']['x_offset'], hyperparams['normalization_parameters']['y_offset']])
    input_norm_scale =   np.concatenate([hyperparams['normalization_parameters']['x_scale'], hyperparams['normalization_parameters']['y_scale']])
    output_norm_offset = np.array(hyperparams['normalization_parameters']['u_offset'])
    output_norm_scale =  np.array(hyperparams['normalization_parameters']['u_scale'])
    test_input = np.arange(nx+ny)+1
    test_input_norm = normalize(test_input,input_norm_scale, input_norm_offset)
    test_output_norm = model(test_input_norm)
    test_output = jax.vmap(functools.partial(denormalize, scale=output_norm_scale, offset=output_norm_offset))(test_output_norm)
    layer_output = []
    output_dim = None
    for layer in model.layers:
        if isinstance(layer, ReshapeLayer):
            output_dim = np.array(layer.output_dim)
            continue
        elif isinstance(layer, TanhLayer):
            activation = 'tanh'
        elif isinstance(layer, ReluLayer):
            activation = 'relu'
        elif isinstance(layer, LinearLayer): # needs to be last because nonlinear inherets from linear lol
            activation = 'linear'
        else: 
            raise Exception(f'Exporting layer of type {type(layer)} not supported!')
        layer_output.append({
            'weights':   np.array(jax.numpy.asarray(layer.weight)),
            'biases':    np.array(jax.numpy.asarray(layer.bias)),
            'activation': activation ,
        })
        
    with h5py.File(filename, 'w') as f:
        for i, layer in enumerate(layer_output):
            grp = f.create_group(f'layer_{i}')
            grp.create_dataset('weights', data=layer['weights'])
            grp.create_dataset('biases',  data=layer['biases'])
            grp.create_dataset('activation', data=layer['activation'].encode('utf-8'))
        
        grp = f.create_group(f'input_norm')
        grp.create_dataset('offset', data=input_norm_offset)
        grp.create_dataset('scale',  data=input_norm_scale)
                
        grp = f.create_group(f'output_norm')
        grp.create_dataset('offset', data=output_norm_offset)
        grp.create_dataset('scale',  data=output_norm_scale)
        
        f.create_dataset('output_shape', data=output_dim)
        
        grp = f.create_group(f'test')
        grp.create_dataset('test_input', data=test_input)
        grp.create_dataset('test_output', data=test_output)
    
    for idx, layer in enumerate(layer_output):
        print(f"Layer {idx}:")
        print(f"\t{layer['weights']}")
        print(f"\t{layer['biases']}")
        print(f"\t{layer['activation']}")
    print(f"\nTest input:  {test_input}")
    print(f"Test output:   {test_output}")
    print(f"Input offset:  {input_norm_offset}")
    print(f"Input scale:   {input_norm_scale}")
    print(f"Output offset: {output_norm_offset}")
    print(f"Output scale:  {output_norm_scale}")
    

def load_eqx_export_h5(filename):
    model, hyperparameters = load_model(filename=filename)
    outfilename = f"{filename[:-4]}.h5"
    export_model_as_h5(outfilename, hyperparams=hyperparameters, model=model)
    

if __name__=="__main__":
    # fire.Fire({
        # "load_eqx_export_h5": load_eqx_export_h5
    # })
    
    load_eqx_export_h5("/share/mihaela-larisa.clement/soeampc-data/models/20250510-140709/399.eqx")