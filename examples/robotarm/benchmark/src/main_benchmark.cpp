#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>
 
#include <Eigen/Dense>

#define H5_USE_EIGEN 
#include "highfive/H5Easy.hpp"

#include "ampc.hpp"
#include "utils.hpp"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <h5-model-path>" << std::endl;
    return -1;
  }
  std::string mlp_model_path = argv[1];
  std::cout << "Loading model: " << mlp_model_path << std::endl;
  auto now_string = time_stamp();
  
  const bool safety = true;
  Controller controller(mlp_model_path, safety);

  constexpr float T_sim = 5;
  constexpr float dt = 0.05f;
  constexpr size_t N_sim = static_cast<size_t>(T_sim/dt);
  
  Eigen::Vector<float, 14> x0;
  x0 << -1.57079, -0.785398, 0.0, -2, 0, 1.5708, 0.785398, 0,0,0,0,0,0,0;


  Eigen::MatrixXf Ys(NY, 3);
  Ys.col(0) << -0.09376911, -0.71724763,  0.35187067,      0.18915935,  0.68137134, -0.68130571,  0.18914113;
  Ys.col(1) << -0.09376911, -0.21724763,  0.25187067,      0.18915935,  0.68137134, -0.68130571,  0.18914113;
  Ys.col(2) <<  0.07928567, -0.67870293,  0.44347463,      0.18915935,  0.68137134, -0.68130571,  0.18914113;
  Eigen::VectorXf Ys_times(3);
  Ys_times << 0, 2, 3;
  size_t Y_idx_now = 0;

  Eigen::MatrixXf X_sim(N_sim, 14);
  Eigen::MatrixXf Y_sim(N_sim, 7);
  Eigen::MatrixXf U_sim(N_sim, 7);

  bool firsttimefeasible = false;
  
  for (size_t t = 0; t<N_sim; t++ ){
    if (Y_idx_now+1 < Ys_times.size() && Ys_times[Y_idx_now+1] <= t*dt){
      Y_idx_now++;
      std::cout << "New reference: " << Ys.col(Y_idx_now).transpose() << std::endl;
    }
    Eigen::Vector<float,NY> y = Ys.col(Y_idx_now);

    Eigen::Vector<float,7> u = controller.update(x0, y);
    if (!firsttimefeasible && controller.candidate_feasible){
      std::cout << "Controller became feasible at t=" << t << " that is " << t*dt << "s" << std::endl;
      firsttimefeasible = true;
    }
    
    X_sim.row(t) = x0;
    U_sim.row(t) = u;
    Y_sim.row(t) = y;

    Eigen::Vector<float, 14> x_dot;
    x_dot << x0.bottomRows(7), u;
    x0 += x_dot*dt;
  }

  std::cout << std::endl;
  std::cout << "Mean inference time:           " << controller.get_mean_inf_time().count()*1000.f << "ms" << std::endl;
  std::cout << "Mean safety augmentation time: " << controller.get_mean_aug_time().count()*1000.f << "ms" << std::endl;
  std::cout << std::endl;

  {
    const std::string output_filename = "logs/sim_" + time_stamp() + ".h5";
    save_simulation(output_filename, X_sim, Y_sim, U_sim);
    std::cout << "Saved simulation results to " << output_filename << std::endl;
  }
  {
    const std::string output_filename = "logs/sim_latest.h5";
    save_simulation(output_filename, X_sim, Y_sim, U_sim);
    std::cout << "Saved simulation results to " << output_filename << std::endl;

  }
}