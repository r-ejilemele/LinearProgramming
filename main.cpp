#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <thread>
#include <utility>
#include <vector>

namespace py = pybind11;

float some_fn(float arg1, float arg2) { return arg1 + arg2; }
class LP {

public:
  /**
   * Creates an Instance of a Linear Program maximizer
   *
   * @param A The n x m linear constraint matrix in the equation Ax <= b
   * @param B The n x 1 constraint vector in the equation Ax <= b
   * @param C The m x 1 objective function coefficients in the equation c^Tx
   */
  LP(Eigen::MatrixXd A, Eigen::MatrixXd B, Eigen::MatrixXd C, bool maxim) {
    m_a = A;
    m_b = B;
    m_c = C;
    // m_maxim = maximizer;
    int number_of_constrants = m_b.size();
    m_b.conservativeResize(m_b.size() + 1);
    m_push_back(m_b, 0);
    Eigen::MatrixXd stacked;
    stacked.resize(m_a.rows() + m_c.cols(), m_a.cols());
    m_v_stack(m_a, (maxim ? -1 : 1) * m_c.transpose(), stacked);
    Eigen::MatrixXd stacked2;
    Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(
        number_of_constrants + 1, number_of_constrants + 1);
    stacked2.resize(stacked.rows(), stacked.cols() + identity.cols());
    // std::cout << stacked << std::endl << std::endl;
    // std::cout << identity << std::endl << std::endl;
    m_h_stack(stacked, identity, stacked2);
    // std::cout << stacked2 << std::endl << std::endl;
    m_table.resize(stacked2.rows(), stacked2.cols() + m_b.cols());
    // std::cout << m_b << std::endl << std::endl;
    m_h_stack(stacked2, m_b, m_table);
    std::cout << m_table << std::endl << std::endl;
    m_num_variables = m_c.size();
  }

  // /**
  //  * Creates an Instance of a Linear Program maximizer
  //  *
  //  * Takes in rvalues for the Inputs
  //  *
  //  * @param A The n x m linear constraint matrix in the equation Ax <= b
  //  * @param B The n x 1 constraint vector in the equation Ax <= b
  //  * @param C The m x 1 objective function coefficients in the equation c^Tx
  //  */
  // LP(MatrixXd &&A, VectorXd &&B, VectorXd &&C, bool maxim)
  //     : m_a(std::move(A)), m_b(std::move(B)), m_c(std::move(C)) {
  //   int number_of_constrants = m_b.size();
  //   m_push_back(m_b, 0);
  //   MatrixXd stacked;
  //   m_v_stack(m_a, (maxim ? 1 : -1) * m_c.transpose(), stacked);
  //   MatrixXd stacked2;
  //   m_h_stack(
  //       stacked,
  //       MatrixXd::Identity(number_of_constrants + 1, number_of_constrants +
  //       1), stacked2);
  //   m_h_stack(stacked2, m_b, m_table);
  //   m_num_variables = m_c.size();
  // }

  /**
   * @brief  an Instance of a Linear Program
   *
   *
   * @returns the feasible solution this Linear Program using the Simplex
   * algorithm
   */
  Eigen::VectorXd simplexSolver() {
    bool exists_negative_in_last_row =
        (m_table.row(m_table.rows() - 1).array() < 0).any();
    while (exists_negative_in_last_row) {
      Eigen::Index pivot_column = m_argmin(m_table.row(m_table.rows() - 1));
      Eigen::Index pivot_row =
          m_argmin((m_table.col(m_table.cols() - 1).array() /
                    m_table.col(pivot_column).array())
                       .segment(0, m_table.rows() - 1));

      double pivot_elem = m_table(pivot_row, pivot_column);
      m_table.row(pivot_row) = m_table.row(pivot_row) / pivot_elem;
      m_elimination(pivot_row, pivot_column);

      exists_negative_in_last_row =
          (m_table.row(m_table.rows() - 1).array() < 0).any();
    }
    // std::cout << m_table << std::endl << endl;
    Eigen::VectorXd output(m_num_variables, 1);
    for (int i = 0; i < m_num_variables; i++) {
      auto [hot, index] = m_is_one_hot(m_table.col(i));

      if (hot) {
        output(i) = m_table(index, m_table.cols() - 1);
      } else {
        output(i) = 0;
      }
    }
    std::cout << m_table << std::endl << std::endl;
    std::cout << output << std::endl << std::endl;
    return output;
  }

private:
  Eigen::MatrixXd m_a;
  Eigen::VectorXd m_b;
  Eigen::MatrixXd m_c;
  Eigen::MatrixXd m_table;
  int m_num_variables;

  void m_elimination(Eigen::Index pivot_row, Eigen::Index pivot_column) {
    for (int i = 0; i < m_table.rows(); i++) {
      if (i != pivot_row) {

        double factor = m_table(i, pivot_column);
        m_table.row(i) -= factor * m_table.row(pivot_row);
      }
    }
    return;
  }

  std::pair<bool, int> m_is_one_hot(const Eigen::Ref<Eigen::VectorXd> v) {

    bool found = false;
    int index = 0;
    for (int i = 0; i < v.size(); i++) {
      if (v(i) != 0 && found == true) {
        return std::pair<bool, int>(false, index);
      } else if (v(i) != 0 && v(i) != 1) {
        return std::pair<bool, int>(false, index);
      } else if (v(i) == 1 && found == false) {
        found = true;
        index = i;
      }
    }
    return std::pair<bool, int>(true, index);
  }
  /**
   *
   * @brief mutatues `vec` to have `value` at the end
   *
   * @param vec The vector that will have ```value``` appended to it
   *
   *
   */
  void m_push_back(Eigen::Ref<Eigen::VectorXd> vec, double value) {
    auto n = vec.size();
    // vec.conservativeResize(n + 1);
    vec[n - 1] = value;
  }

  /**
   *
   * @brief stacks the two matrices `matrix` and `matrix2` vertically on top of
   *each other
   *
   *@param matrix1 the matrix that will be on top
   *@param matrix2 the matrix that will be on the bottom
   *@param out the matrix that the output will be put into
   */
  void m_v_stack(Eigen::MatrixXd matrix1, Eigen::MatrixXd matrix2,
                 Eigen::Ref<Eigen::MatrixXd> out) {

    out << matrix1, matrix2;
    return;
  }

  /**
   * @brief stacks the two matrices `matrix` and `matrix2` horizontally next to
   *each other
   *
   *@param matrix1 the matrix that will be on the left
   *@param matrix2 the matrix that will be on the right
   *@param out the matrix that the output will be put into
   */
  void m_h_stack(Eigen::MatrixXd matrix1, Eigen::MatrixXd matrix2,
                 Eigen::Ref<Eigen::MatrixXd> out) {
    out.resize(matrix1.rows(), matrix1.cols() + matrix2.cols());
    out << matrix1, matrix2;
    return;
  }

  int m_argmin(const Eigen::VectorXd v) {
    return std::distance(v.data(),
                         std::min_element(v.data(), v.data() + v.size()));
  }
};

PYBIND11_MODULE(linear, handle) {
  handle.doc() = "This is the doc";
  py::class_<LP>(handle, "LinearProgram")
      .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, bool>())
      .def("simplexSolver", &LP::simplexSolver);
}
