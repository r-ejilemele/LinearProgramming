#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <vector>

using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::VectorXd;
using namespace std;

class LP {

public:
  /**
   * Creates an Instance of a Linear Program maximizer
   *
   * @param A The n x m linear constraint matrix in the equation Ax <= b
   * @param B The n x 1 constraint vector in the equation Ax <= b
   * @param C The m x 1 objective function coefficients in the equation c^Tx
   */
  LP(MatrixXd &A, VectorXd &B, VectorXd &C, bool maxim) {
    m_a = A;
    m_b = B;
    m_c = C;
    // m_maxim = maximizer;
    int number_of_constrants = m_b.size();
    m_push_back(m_b, 0);
    MatrixXd stacked;
    m_v_stack(m_a, (maxim ? 1:-1) * m_c.transpose(), stacked);
    MatrixXd stacked2;
    m_h_stack(
        stacked,
        MatrixXd::Identity(number_of_constrants + 1, number_of_constrants + 1),
        stacked2);
    m_h_stack(stacked2, m_b, m_table);
    m_num_variables = m_c.size();
  }
  /**
   * @brief  an Instance of a Linear Program
   *
   *
   * @returns the feasible solution this Linear Program using the Simplex
   * algorithm
   */
  VectorXd simplexSolver() {
    bool exists_negative_in_last_row =
        (m_table.row(m_table.rows() - 1).array() < 0).any();
    while (exists_negative_in_last_row) {
      Eigen::Index pivot_column = m_argmin(m_table.row(m_table.rows() - 1));
      Eigen::Index pivot_row = m_argmin((m_table.col(m_table.cols() - 1).array() /
                                       m_table.col(pivot_column).array())
                                          .segment(0, m_table.rows() - 1));

      double pivot_elem = m_table(pivot_row, pivot_column);
      m_table.row(pivot_row) = m_table.row(pivot_row) / pivot_elem;
      m_elimination(pivot_row, pivot_column);

      exists_negative_in_last_row =
          (m_table.row(m_table.rows() - 1).array() < 0).any();
    }
    std::cout << m_table << std::endl << endl;
    VectorXd output(m_num_variables, 1);
    for (int i = 0; i < m_num_variables; i++) {
      auto [hot, index] = m_is_one_hot(m_table.col(i));

      if (hot) {
        output(i) = m_table(index, m_table.cols() - 1);
      } else {
        output(i) = 0;
      }
    }
    return output;
  }

private:
  MatrixXd m_a;
  VectorXd m_b;
  VectorXd m_c;
  MatrixXd m_table;
  int m_num_variables;

  void m_elimination(Eigen::Index &pivot_row, Eigen::Index &pivot_column) {
    for (int i = 0; i < m_table.rows(); i++) {
      if (i != pivot_row) {

        double factor = m_table(i, pivot_column);
        m_table.row(i) -= factor * m_table.row(pivot_row);
      }
    }
    return;
  }

  std::pair<bool, int> m_is_one_hot(const Eigen::VectorXd &v) {

    bool found = false;
    int index = 0;
    for (int i = 0; i < v.size(); i++) {
      if (v(i) != 0 && found == true) {
        return pair<bool, int>(false, index);
      } else if (v(i) != 0 && v(i) != 1) {
        return pair<bool, int>(false, index);
      } else if (v(i) == 1 && found == false) {
        found = true;
        index = i;
      }
    }
    return pair<bool, int>(true, index);
  }
  /**
   *
   * @brief mutatues `vec` to have `value` at the end
   *
   * @param vec The vector that will have ```value``` appended to it
   *
   *
   */
  template <class T, class U> void m_push_back(T &vec, const U &value) {
    auto n = vec.size();
    vec.conservativeResize(n + 1);
    vec[n] = value;
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
  void m_v_stack(const MatrixXd &matrix1, const MatrixXd &matrix2,
               MatrixXd &out) {

    out.resize(matrix1.rows() + matrix2.rows(), matrix1.cols());
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
  void m_h_stack(const MatrixXd &matrix1, const MatrixXd &matrix2,
               MatrixXd &out) {
    out.resize(matrix1.rows(), matrix1.cols() + matrix2.cols());
    out << matrix1, matrix2;
    return;
  }

  int m_argmin(const Eigen::VectorXd &v) {
    return std::distance(v.data(),
                         std::min_element(v.data(), v.data() + v.size()));
  }
};

int main() {
  // vector<string> msg {"Hello", "C++", "World", "from", "VS Code", "and the
  // C++ extension!"};

  // for (const string& word : msg)
  // {
  //     cout << word << " ";
  // }
  // cout << endl;
  //      VectorXd m(2,1);
  //   m(0,0) = 3;
  //   m(1,0) = 4;
  //   push_back(m,0);
  //   cout << m << endl;
  //   cout << m.rows() << endl;
  //   cout << m.cols() << endl;
  //   std::cout << m.size() << std::endl;
  MatrixXd A(2, 3);
  A << 3, 2, 1, 2, 5, 3;

  VectorXd B(2, 1);
  B << 10,15;
  VectorXd C(3, 1);
  C << -2, -3, -4;
  LP linear(A, B, C, true);

  //   MatrixXd stacked;
  //   v_stack(A, -1 * C.transpose(), stacked);
  //   MatrixXd stacked2;
  //   h_stack(stacked, MatrixXd::Identity(4, 4), stacked2);
  //   MatrixXd stacked3;
  //   h_stack(stacked2, B, stacked3);

  //   Eigen::Index pivot_column = argmin(stacked3.row(stacked3.rows() - 1));
  //   Eigen::Index pivot_row = argmin(
  //       (stacked3.col(stacked3.cols() - 1).array() / stacked3.col(0).array())
  //           .segment(0, stacked3.rows() - 1));
  //   double pivot_elem = stacked3(pivot_row, pivot_column);
  //   stacked3.row(0) = stacked3.row(0) / pivot_elem;

  //   stacked << B, A;

  std::cout << linear.simplexSolver() << std::endl;
}
