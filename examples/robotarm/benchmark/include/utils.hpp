#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>

inline std::string time_stamp()
{
    const auto  now = std::chrono::system_clock::now();
    const auto   tt = std::chrono::system_clock::to_time_t(now);
    std::tm      tm;
    localtime_r(&tt, &tm);
    std::ostringstream os;
    os << std::put_time(&tm, "%Y%m%d-%H%M%S");        // e.g. 20250507-141231
    return os.str();
}

inline 
void
save_simulation(
  const std::string filename,
  const Eigen::MatrixXf& X_sim,
  const Eigen::MatrixXf& Y_sim,
  const Eigen::MatrixXf& U_sim)
{
  H5Easy::File file(filename, H5Easy::File::Overwrite);
  H5Easy::dump(file, "/X_sim", X_sim);
  H5Easy::dump(file, "/Y_sim", Y_sim);
  H5Easy::dump(file, "/U_sim", U_sim);
  file.flush();
}
