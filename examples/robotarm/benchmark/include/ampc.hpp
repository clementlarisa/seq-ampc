#include <iostream>
#include <fstream>
#include <array>
#include <string>

#include <chrono>

#include <Eigen/Dense>
#include "mlp.hpp"
#include "kinematics.hpp"
#include "sdf.hpp"

static constexpr std::size_t NX{14};
static constexpr std::size_t NU{7};
static constexpr std::size_t NY{7};
static constexpr std::size_t N{20};
static constexpr float dt{0.05f};
static constexpr float eps{1e-4f};

static const Eigen::Vector<float, NX>Q{1, 1, 1, 1, 1, 1, 1, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001};
static const Eigen::Matrix<double, NX, NX> P_double{{ 5.14770897e+02, -1.39115713e-09,  0.00000000e+00,  2.32454153e-09,
    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  4.58827885e+01,
    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00},
  {-1.39115713e-09,  1.20638447e+03,  0.00000000e+00,  0.00000000e+00,
   -1.24112232e-09,  1.44937266e-09,  0.00000000e+00,  0.00000000e+00,
    7.69986518e+01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00},
  { 0.00000000e+00,  0.00000000e+00,  7.27412848e+02,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  5.56645811e+01,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00},
  { 2.32454153e-09,  0.00000000e+00,  0.00000000e+00,  5.78129368e+02,
    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00,  4.88483513e+01,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00},
  { 0.00000000e+00, -1.24112232e-09,  0.00000000e+00,  0.00000000e+00,
    5.14770781e+02,  0.00000000e+00, -1.34099234e-09,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  4.58827765e+01,
    0.00000000e+00,  0.00000000e+00},
  { 0.00000000e+00,  1.44937266e-09,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  4.61993804e+02,  1.25171796e-09,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    4.33614277e+01,  0.00000000e+00},
  { 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
   -1.34099234e-09,  1.25171796e-09,  4.61993791e+02,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  4.33614263e+01},
  { 4.58827885e+01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  4.58827885e+00,
    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00},
  { 0.00000000e+00,  7.69986518e+01,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    7.69986518e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00},
  { 0.00000000e+00,  0.00000000e+00,  5.56645811e+01,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  5.56645811e+00,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00},
  { 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  4.88483513e+01,
    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00,  4.88483513e+00,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00},
  { 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    4.58827765e+01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  4.58827765e+00,
    0.00000000e+00,  0.00000000e+00},
  { 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  4.33614277e+01,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    4.33614276e+00,  0.00000000e+00},
  { 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00,  4.33614263e+01,  0.00000000e+00,
    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
    0.00000000e+00,  4.33614262e+00}};
static const Eigen::Matrix<float, NX, NX> P=P_double.cast<float>();
static const Eigen::Vector<float, NU>R{{0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001}};
static const float alpha{ 0.6 };
static const float alphaSquared = alpha*alpha;
static const Eigen::Vector<float, NX> x_max{{2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671, 2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61}};
static const Eigen::Vector<float, NX> x_min{{-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671, -2.175, -2.175, -2.175, -2.175, -2.61, -2.61, -2.61}};
static const Eigen::Vector<float, NU> u_max{ {15, 7.5, 10, 12.5, 15, 20, 20} };
static const Eigen::Vector<float, NU> u_min{ {-15, -7.5, -10, -12.5, -15, -20, -20} };

static SDF3D sdf("models/sdf.h5");
// static SDF3D sdf("/home/hose/projects/dsme/online-validation-extra-examples/franka/tmp/sdf.h5");

void
forwardSimulateTrajectoryClippedInputs(Eigen::Matrix<float, NX, N+1>& X, Eigen::Matrix<float, NU, N>& U, Eigen::Matrix<float, NU, N>& V){
    for (std::size_t ii=0; ii < N; ii++){
        U.col(ii) = (prestabilization(X.col(ii),V.col(ii))).cwiseMin(u_max).cwiseMax(u_min);
        const Eigen::Vector<float, 7> dq = X.col(ii).tail(7);
        const Eigen::Vector<float, NX> x_dot = (Eigen::VectorXf(14) << dq, U.col(ii)).finished();
        X.col(ii+1) = X.col(ii) + dt*x_dot;
    }
}


void shiftAppendTerminalInPlace(Eigen::Matrix<float, NX, N+1>& X, Eigen::Matrix<float, NU, N>& U){
    X.block(0,0,NX,N) = X.block(0, 1, NX, N); // shift
    U.block(0,0,NU,N-1) = U.block(0, 1, NU, N-1); //shift
    Eigen::VectorXf x_terminal = Eigen::VectorXf::Zero(X.rows());
    x_terminal.tail(7) = X.col(N - 1).tail(7);
    U.col(N-1) = (prestabilization(x_terminal, Eigen::Vector<float,7>::Zero())).cwiseMin(u_max).cwiseMax(u_min); // terminal controller
    Eigen::Vector<float,NX> xdot;
    xdot << X.col(N-1).tail(7), U.col(N-1);
    X.col(N) = X.col(N-1)+xdot*dt;
}


bool inStateConstr(const Eigen::Matrix<float, NX, N>& X) {
    return ((X.colwise() - x_min).array() >= eps).all() &&
           ((X.colwise() - x_max).array() <= eps).all();
}

bool inTerminalConstr( const Eigen::Vector<float, NX>& x){
    return ( x.transpose() * P * x - alphaSquared ) <= eps;
};

bool inOutputConstr(const Eigen::Matrix<float, NX, N+1>& X, SDF3D& sdf) {
    for (int i = 0; i < X.cols(); ++i) {
      Eigen::Matrix<float, 7, 1> q = X.col(i).template head<7>();
      auto [position, orientation] = kinematics(q);
      if (sdf.sdf(position) + 0.02f > 0.0f) {
        return false;
      }
    }
    return true;
}

bool feasible(const Eigen::Matrix<float, NX, N+1>& X){
    bool feasInState = inStateConstr(X.block(0,0,NX,N));
    Eigen::Vector<float,14> x_term; 
    x_term << Eigen::Vector<float,7>::Zero(), X.col(N).bottomRows(7);
    bool feasInTerm  = inTerminalConstr(x_term);
    bool feasInOutput = inOutputConstr(X, sdf);
    // std::cout << "feasInState: " << feasInState << "      feasInOtuput: " << feasInOutput << "      feasInTerm " << feasInTerm << std::endl;
    return feasInState && feasInOutput && feasInTerm;
}

float stateCost(const Eigen::Matrix<float, NX, N>& X, const Eigen::Vector<float, NX>& Q){
    const Eigen::Matrix<float, NX, N> QX = Q.asDiagonal()*X;
    Eigen::Matrix<float, 1, N> XtQX;
    for (size_t ii=0; ii<N; ii++){
        XtQX(0,ii) = X.col(ii).dot(QX.col(ii));
    }
    return XtQX.sum();
};

float inputCost(const Eigen::Matrix<float, NU, N>& U, const Eigen::Vector<float, NU>& R){
    const Eigen::Matrix<float, NU, N> RU = R.asDiagonal()*U;
    Eigen::Matrix<float, 1, N> UtRU;
    for (size_t ii=0; ii<N; ii++){
         UtRU(0,ii) = U.col(ii).dot(RU.col(ii));
    }
    return UtRU.sum();
};

float terminalCost(const Eigen::Vector<float, NX> x, const Eigen::Matrix<float, NX, NX>& P){
    return x.transpose()*P*x;
};

float cost(const Eigen::Matrix<float, NX, N+1>& X, const Eigen::Matrix<float, NU, N>& U)
{
    return stateCost(X.block(0,0,NX,N),Q) + inputCost(U,R) + terminalCost(X.col(N), P);
};


class Controller{
public:
    MLP mlp;
    Eigen::Matrix<float, NX, N+1> X_cand;
    Eigen::Matrix<float, NX, N+1> X_pred;

    Eigen::Matrix<float, NU, N> U_cand;
    Eigen::Matrix<float, NU, N> U_pred;

    Eigen::Matrix<float, NU, N> V_pred;

    float candidate_cost = 0;
    bool candidate_feasible = false;

    std::chrono::duration<double> total_inf_time{0};
    std::chrono::duration<double> total_aug_time{0};
    std::size_t total_iterations = 0;

    bool safety;
    
    Controller(const std::string model_path, const bool safety=false) :
    mlp(load_layers_from_h5(model_path, true)), safety(safety)
    {
    }

    Eigen::Vector<float, 7>
    update(const Eigen::Vector<float, NX>& x0, const Eigen::Vector<float, NY>& y){
        const auto start_inference = std::chrono::high_resolution_clock::now();
        Eigen::VectorXf in_(NX+NY);
        in_ << x0, y;
        const auto res = forward_pass(mlp, in_);
        V_pred = res.transpose();
        const auto end_inference = std::chrono::high_resolution_clock::now();
        total_inf_time += end_inference-start_inference;
        
        const auto start_aug = std::chrono::high_resolution_clock::now();
        X_pred.col(0) = x0;
        forwardSimulateTrajectoryClippedInputs(X_pred, U_pred, V_pred);
        
        bool predicted_feasible = feasible(X_pred);
        float predicted_cost = cost(X_pred, U_pred);

        // if ( (predicted_feasible && (predicted_cost <= candidate_cost))
        if ( (predicted_feasible )
                || !candidate_feasible
                || !safety
            ){
                X_cand = X_pred;
                U_cand = U_pred;
                candidate_feasible = predicted_feasible;
        };

        Eigen::Vector<float, 7> u0 = U_cand.col(0);

        shiftAppendTerminalInPlace(X_cand, U_cand);
        const auto end_aug = std::chrono::high_resolution_clock::now();
        total_aug_time += end_aug - start_aug;
        total_iterations++;
        
        return u0;
    }

    std::chrono::duration<double>
    get_mean_inf_time(){
        return (total_inf_time/total_iterations);
    }

    std::chrono::duration<double>
    get_mean_aug_time(){
        return total_aug_time/total_iterations;
    }
};