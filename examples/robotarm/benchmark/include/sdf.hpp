#include "highfive/H5Easy.hpp"
#include <Eigen/Dense>
#include <stdexcept>
#include <cassert>
#include <cmath>


class SDF3D {
public:
    SDF3D(const std::string& h5_filename) {
        HighFive::File file(h5_filename, HighFive::File::ReadOnly);

        std::cout << "Loading SDF from " << h5_filename << std::endl;
        // Load resolution
        file.getDataSet("resolution").read(resolution_);

        std::cout << "Resolution: " << resolution_ << std::endl;

        // Load SDF 3D data
        std::vector<int> dims;
        file.getDataSet("dims").read(dims);
        nx_ = dims[0];
        ny_ = dims[1];
        nz_ = dims[2];
        std::cout << "nx: " << nx_ << "    ny: " << ny_ << "    nz: " << nz_ <<std::endl;

        sdf_data_ = H5Easy::load<Eigen::VectorXf>(file, "sdf_flat");

        // Compute physical extents
        dx_ = (nx_ - 1) * resolution_;
        dy_ = (ny_ - 1) * resolution_;
        dz_ = (nz_ - 1) * resolution_;

        x0_ = -dx_ / 2.0f;
        y0_ = -dy_ / 2.0f;
        z0_ = -dz_ / 2.0f;

        // Load and validate test positions (optional but enabled by default)
        Eigen::MatrixXf test_points = H5Easy::load<Eigen::MatrixXf>(file, "test_positions");
        Eigen::VectorXf test_outputs = H5Easy::load<Eigen::VectorXf>(file, "sdf_outputs");
        std::cout << "Test Points: " << test_points.rows()  << " by " << test_points.cols() << std::endl;
        std::cout << "Test Outputs: " << test_points.size()  << std::endl;

        assert(test_points.cols() == 3);
        assert(test_points.rows() == test_outputs.size());
    

        for (int i = 0; i < test_points.rows(); ++i) {
            Eigen::Vector3f p = test_points.row(i).transpose();
            float v_computed = sdf(p);
            float v_expected = test_outputs(i);
    
            if (std::abs(v_computed - v_expected) > 1e-5f) {
                std::cout << "Mismatch at index " << i
                    << ": computed = " << v_computed
                    << ", expected = " << v_expected
                    << std::endl;
            }    
        }
    
        std::cout << "All test points passed validation.\n";
    }

    float sdf(const Eigen::Vector3f& position) const {
        // Convert position to grid indices
        float fx = (position.x() - x0_) / resolution_;
        float fy = (position.y() - y0_) / resolution_;
        float fz = (position.z() - z0_) / resolution_;

        int x = static_cast<int>(std::floor(fx));
        int y = static_cast<int>(std::floor(fy));
        int z = static_cast<int>(std::floor(fz));

        float dx = fx - x;
        float dy = fy - y;
        float dz = fz - z;

        // Clamp indices to be within the valid range
        x = std::clamp(x, 0, static_cast<int>(nx_) - 2);
        y = std::clamp(y, 0, static_cast<int>(ny_) - 2);
        z = std::clamp(z, 0, static_cast<int>(nz_) - 2);

        // Compute indices for the 8 surrounding grid points
        size_t idx000 = index(x, y, z);
        size_t idx100 = index(x + 1, y, z);
        size_t idx010 = index(x, y + 1, z);
        size_t idx110 = index(x + 1, y + 1, z);
        size_t idx001 = index(x, y, z + 1);
        size_t idx101 = index(x + 1, y, z + 1);
        size_t idx011 = index(x, y + 1, z + 1);
        size_t idx111 = index(x + 1, y + 1, z + 1);

        // Retrieve the SDF values at the 8 surrounding grid points
        float c000 = sdf_data_(idx000);
        float c100 = sdf_data_(idx100);
        float c010 = sdf_data_(idx010);
        float c110 = sdf_data_(idx110);
        float c001 = sdf_data_(idx001);
        float c101 = sdf_data_(idx101);
        float c011 = sdf_data_(idx011);
        float c111 = sdf_data_(idx111);

        // Perform trilinear interpolation
        float c00 = c000 * (1 - dx) + c100 * dx;
        float c10 = c010 * (1 - dx) + c110 * dx;
        float c01 = c001 * (1 - dx) + c101 * dx;
        float c11 = c011 * (1 - dx) + c111 * dx;

        float c0 = c00 * (1 - dy) + c10 * dy;
        float c1 = c01 * (1 - dy) + c11 * dy;

        float c = c0 * (1 - dz) + c1 * dz;

        return c;
    }

private:
    size_t index(int x, int y, int z) const {
        return static_cast<size_t>(z) * ny_ * nx_ + static_cast<size_t>(y) * nx_ + static_cast<size_t>(x);
    }

    Eigen::Matrix<float, Eigen::Dynamic, 1> sdf_data_;
    size_t nx_, ny_, nz_;
    float resolution_;
    float dx_, dy_, dz_;
    float x0_, y0_, z0_;
};