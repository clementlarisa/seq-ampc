#ifndef MULTI_LAYER_PERCEPTRON_HPP
#define MULTI_LAYER_PERCEPTRON_HPP

#include <unordered_set>

#include <eigen3/Eigen/Dense>
// #include <HighFive/H5Easy.hpp>
#include "highfive/H5Easy.hpp"

struct NormParam{
    Eigen::VectorXf offset;
    Eigen::VectorXf scale;
};

struct Layer {
    Eigen::MatrixXf weights;
    Eigen::VectorXf biases;
    std::string activation;
};

struct MLP {
    NormParam input_norm;
    std::vector<Layer> layers;
    NormParam output_norm;
    Eigen::Vector2i output_shape;
} ;

inline
Eigen::VectorXf
normalize(const Eigen::VectorXf& x, const Eigen::VectorXf& offset, const Eigen::VectorXf& scale) {
    return (x - offset).cwiseQuotient(scale);
}

inline
Eigen::VectorXf
denormalize(const Eigen::VectorXf& x, const Eigen::VectorXf& offset, const Eigen::VectorXf& scale) {
    return x.cwiseProduct(scale) + offset;
}

inline
Eigen::VectorXf
relu(const Eigen::VectorXf& x) {
    return x.cwiseMax(0);
}

inline
Eigen::VectorXf
leakyrelu(const Eigen::VectorXf& x, float alpha = 0.01) {
    return x.unaryExpr([alpha](float v) { return v > 0 ? v : alpha * v; });
}

inline
Eigen::VectorXf
elu(const Eigen::VectorXf& x, float alpha = 1.0) {
    return x.unaryExpr([alpha](float v) { return v > 0 ? v : alpha * (std::exp(v) - 1); });
}

inline
Eigen::VectorXf
softmax(const Eigen::VectorXf& x) {
    Eigen::VectorXf exp_x = (x.array() - x.maxCoeff()).exp();
    return exp_x / exp_x.sum();
}

inline
Eigen::VectorXf
tanh(const Eigen::VectorXf& x) {
    return x.array().tanh();
}

inline
Eigen::VectorXf
apply_activation(const Eigen::VectorXf& x, const std::string& activation) {
    if (activation == "relu") {
        return relu(x);
    } else if (activation == "softmax") {
        return softmax(x);
    } else if (activation == "tanh") {
        return tanh(x);
    } else if (activation == "leakyrelu"){
        return leakyrelu(x);
    } else if (activation == "elu"){
        return elu(x);
    } else if (activation == "linear"){
        return x;
    }
    
    else {
        throw std::invalid_argument("Unknown activation function: " + activation);
    }
}

inline
bool
is_valid_activation(const std::string& activation) {
    static const std::unordered_set<std::string> valid_activations = {"relu","leakyrelu", "elu", "softmax", "tanh", "linear"};
    return valid_activations.find(activation) != valid_activations.end();
}

inline
Eigen::MatrixXf
forward_pass(const MLP& mlp, const Eigen::VectorXf& input) {
    Eigen::VectorXf output = normalize(input, mlp.input_norm.offset, mlp.input_norm.scale);

    for (const auto& layer : mlp.layers) {
        output = apply_activation(layer.weights * output + layer.biases, layer.activation);
    }

    
    Eigen::MatrixXf output_reshaped = Eigen::Map<const Eigen::MatrixXf>(
        output.data(),
        mlp.output_shape(1),
        mlp.output_shape(0)
    ).transpose();
    for (int i = 0; i < output_reshaped.rows(); ++i) {
        output_reshaped.row(i) = denormalize(output_reshaped.row(i).transpose(), 
                                             mlp.output_norm.offset, 
                                             mlp.output_norm.scale).transpose();
    }
    return output_reshaped;
}

MLP
load_layers_from_h5(const std::string model_path, const bool test){
    std::cout << "[MLP] loading model from " << model_path << std::endl;
    HighFive::File file(model_path, HighFive::File::ReadOnly);       

    MLP mlp;

    // Assuming you know the number of layers or iterate until no more layers are found
    for (int i = 0; ; ++i) {
        std::string layer_name = "layer_" + std::to_string(i);

        if (!file.exist(layer_name)) break;

        Eigen::MatrixXf weights = H5Easy::load<Eigen::MatrixXf>(file, layer_name + "/weights");
        Eigen::VectorXf biases = H5Easy::load<Eigen::VectorXf>(file, layer_name + "/biases");

        std::string activation = H5Easy::load<std::string>(file, layer_name + "/activation");

        if (!is_valid_activation(activation)) {
            throw std::invalid_argument("Unknown activation function: " + activation + " in layer " + std::to_string(i));
        }

        mlp.layers.push_back({weights, biases, activation});
    }

    mlp.input_norm.offset = H5Easy::load<Eigen::VectorXf>(file, "input_norm/offset");
    mlp.input_norm.scale = H5Easy::load<Eigen::VectorXf>(file, "input_norm/scale");
    mlp.output_norm.offset = H5Easy::load<Eigen::VectorXf>(file, "output_norm/offset");
    mlp.output_norm.scale = H5Easy::load<Eigen::VectorXf>(file, "output_norm/scale");
    mlp.output_shape = H5Easy::load<Eigen::Vector2i>(file, "output_shape");

    if (test){
        const Eigen::VectorXf test_input = H5Easy::load<Eigen::VectorXf>(file, "test/test_input");
        const Eigen::MatrixXf test_output = H5Easy::load<Eigen::MatrixXf>(file, "test/test_output");
        std::cout << "Test input size: " << test_input.size() << std::endl;
        std::cout << "Test output size: " << test_output.rows() << " by " << test_output.cols() << std::endl;
        std::cout << "Model output size: " << mlp.output_shape(0) << " by " << mlp.output_shape(1) << std::endl;
        const Eigen::MatrixXf predicted_output = forward_pass(mlp, test_input);
        if (!predicted_output.isApprox(test_output, 1e-4)){
            std::cout << "\nTest input is: \n\t" << test_input.transpose() << std::endl;
            std::cout << "\nTest output is: \n\t" << test_output.transpose() << std::endl;
            
            std::cout << "\nPredicted output is: \n\t" << predicted_output.transpose() << std::endl;
            std::cout << "\nLayers: " << std::endl;
            for (size_t i=0; i<mlp.layers.size(); i++){
                std::cout 
                    << "Layer " << std::to_string(i)
                    << ": \t activation " << mlp.layers[i].activation
                    << ", weights [" << mlp.layers[i].weights.rows() << ", " << mlp.layers[i].weights.cols() << "]"
                    << ", bias [" << mlp.layers[i].biases.rows() << ", " << mlp.layers[i].biases.cols() << "]" << std::endl;
                // std::cout << "Weights:" << mlp.layers[i].weights << std::endl;
                // std::cout << "Biases:" << mlp.layers[i].biases.transpose() << std::endl;
            }

            std::cout << "input offset:" << mlp.input_norm.offset.transpose() << std::endl;
            std::cout << "input scale:" << mlp.input_norm.scale.transpose() << std::endl;
            std::cout << "output offset:" << mlp.output_norm.offset.transpose() << std::endl;
            std::cout << "output scale:" << mlp.output_norm.scale.transpose() << std::endl;


            std::cout << "\n" << std::endl;
            throw std::runtime_error("Error: Predicted output is not approximately equal to expected output within tolerance.");
        }
        std::cout << "[MLP] tests passed!" << std::endl;
    }

    return mlp;
}


#endif // MULTI_LAYER_PERCEPTRON