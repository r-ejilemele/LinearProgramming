#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#define TOLERANCE 1e-15

using Eigen::placeholders::last;
namespace py = pybind11;

float some_fn(float arg1, float arg2) { return arg1 + arg2; }

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
  os << "["; // Start with an opening bracket
  for (size_t i = 0; i < v.size(); ++i) {
    os << v[i]; // Print each element
    if (i != v.size() - 1) {
      os << ", "; // Add a comma and space between elements
    }
  }
  os << "]"; // End with a closing bracket
  return os; // Return the ostream reference
}

class LP {

public:
  /**
   * Creates an Instance of a Linear Program maximizer
   *
   * @param A The n x m linear constraint matrix in the equation Ax <= b
   * @param B The n x 1 constraint vector in the equation Ax <= b
   * @param C The m x 1 objective function coefficients in the equation c^Tx
   * @param maxim is True if this is a maximization problem otherwise it's False
   */
  LP(Eigen::MatrixXd A, Eigen::MatrixXd B, Eigen::MatrixXd C, bool maxim,
     std::vector<std::string> constraint) {
    m_a = A;
    m_La.resize(A.rows(), A.cols());
    m_b = B;
    m_Lb.resize(B.rows(), B.cols());
    m_c = (maxim ? -1 : 1) * C;
    types = constraint;
    for (auto i = 0; i < A.rows(); ++i) {
      if (constraint[i] == "G") {
        m_Lb(i) = -m_b(i);
        m_La.row(i) = -m_a.row(i);
      } else {
        m_Lb(i) = m_b(i);
        m_La.row(i) = m_a.row(i);
      }
    }
    // Eigen::VectorXd rowNorms = m_a.rowwise().norm();
    // Eigen::VectorXd colNorms = m_a.colwise().norm();

    // std::cout << "rowNorms shape: (" << rowNorms.rows() << ", "
    //           << rowNorms.cols() << ")" << std::endl;
    // std::cout << "colNorms shape: (" << colNorms.rows() << ", "
    //           << colNorms.cols() << ")" << std::endl;
    // rowNorms = rowNorms.unaryExpr([](double x) { return x == 0.0 ? 1.0 : x;
    // }); colNorms = colNorms.unaryExpr([](double x) { return x == 0.0 ? 1.0 :
    // x; }); Eigen::DiagonalMatrix<double, Eigen::Dynamic>
    // rowNormsDiag(rowNorms); Eigen::DiagonalMatrix<double, Eigen::Dynamic>
    // colNormsDiag(colNorms); Eigen::DiagonalMatrix<double, Eigen::Dynamic>
    // rowNormsDiagInv =
    //     rowNormsDiag.inverse();
    // Eigen::DiagonalMatrix<double, Eigen::Dynamic> colNormsDiagInv =
    //     colNormsDiag.inverse();

    // m_a = rowNormsDiagInv * m_a * colNormsDiagInv;
    // m_b = rowNormsDiagInv * m_b;
    // m_c = colNormsDiagInv * m_c;

    // std::cout << "m_a shape: (" << m_a.rows() << ", " << m_a.cols() << ")"
    //           << std::endl;
    // std::cout << "colNormsMat shape: (" << colNormsMat.rows() << ", "
    //           << colNormsMat.cols() << ")" << std::endl;
    // std::cout << "rowNormsMat shape: (" << rowNormsMat.rows() << ", "
    //           << rowNormsMat.cols() << ")" << std::endl;
    // m_a = m_a.array() / rowNormsMat.array() / colNormsMat.array();
    // Eigen::VectorXd rowMax = m_a.rowwise().maxCoeff();
    // Eigen::DiagonalMatrix<double, Eigen::Dynamic> rowMaxDiag(rowMax);
    // rowMaxDiag = rowMaxDiag.inverse();
    // m_a = rowMaxDiag * m_a;
    // m_b = rowMaxDiag * m_b;
    // for (int i = 0; i < m_a.rows(); i++) {
    //   auto rowMax = A.row(i).cwiseAbs().maxCoeff();
    //   if (rowMax > 0) {

    //     m_a.row(i) /= rowMax;
    //     m_b.row(i) /= rowMax;
    //   }
    // }
    m_build_phase_one_table();
    m_num_variables = m_c.size();
    // m_maxim = maximizer;
    // std::cout << "Constraint: " << constraint << std::endl;
    // int number_of_constrants = m_b.size();
    // // m_b.conservativeResize(m_b.size() + 1);
    // // m_push_back(m_b, 0);

    // Eigen::MatrixXd stacked;
    // stacked.resize(m_a.rows() + m_c.cols(), m_a.cols());
    // m_v_stack(m_a, (maxim ? -1 : 1) * m_c.transpose(), stacked);
    // Eigen::MatrixXd stacked2;
    // Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(
    //     number_of_constrants + 1, number_of_constrants + 1);
    // stacked2.resize(stacked.rows(), stacked.cols() + identity.cols());
    // // std::cout << stacked << std::endl << std::endl;
    // // std::cout << identity << std::endl << std::endl;
    // m_h_stack(stacked, identity, stacked2);
    // // std::cout << stacked2 << std::endl << std::endl;
    // m_table.resize(stacked2.rows(), stacked2.cols() + m_b.cols());
    // // std::cout << m_b << std::endl << std::endl;
    // m_h_stack(stacked2, m_b, m_table);
    // // std::cout << "Table1: " << std::endl << m_table << std::endl;
    // auto n = C.size();
    // m_basis.resize(A.rows());
    // for (int i = 0; i < A.rows(); ++i) {
    //   m_basis(i) = n + i;
    // }
    // m_preProcess();
    // // std::cout << m_table << std::endl << std::endl;
    // // std::cout << "Table2: " << std::endl << m_table << std::endl;
    // // std::cout << "Basis: " << std::endl << m_basis << std::endl;
    // // std::cout << "Basis Size: " << std::endl << m_basis.size() <<
    // std::endl;
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
    phaseOneSimplexSolver();
    // std::cout << "Basis: " << std::endl << m_basis << std::endl;
    // std::cout << "Table: \n" << m_table << std::endl;

    bool exists_negative_in_last_row =
        (m_table(m_table.rows() - 1, Eigen::seq(0, last - 1)).array() <
         -TOLERANCE)
            .any();
    int counter = 0;
    while (exists_negative_in_last_row) {
      std::cout << std::endl
                << "Phase 2: "
                << m_table(m_table.rows() - 1, m_table.cols() - 1) << std::endl;
      // std::cout << "Table: \n" << m_table << std::endl;
      //  std::cout << "Table: \n" << m_table << std::endl;
      Eigen::Index pivot_column =
          m_find_pivot_column(m_table.row(m_table.rows() - 1));
      // std::cout << "After pivot Column" << std::endl;

      int pivot_row = m_find_pivot_row(pivot_column, m_table);
      // std::cout << "Position " << pivot_row << ", " << pivot_column <<
      // std::endl
      //           << std::endl;

      m_pivot(pivot_row, pivot_column, m_table);

      exists_negative_in_last_row =
          (m_table(m_table.rows() - 1, Eigen::seq(0, last - 1)).array() <
           -TOLERANCE)
              .any();
      counter++;
    }

    std::cout << std::endl
              << std::endl
              << m_table(m_table.rows() - 1, m_table.cols() - 1) << std::endl
              << std::endl;
    Eigen::VectorXd output(m_num_variables, 1);
    for (int i = 0; i < m_num_variables; i++) {
      auto [hot, index] = m_is_one_hot(m_table.col(i));

      if (hot) {
        output(i) = m_table(index, m_table.cols() - 1);
      } else {
        output(i) = 0;
      }
    }

    // std::cout << "HELLOOO" << std::endl << std::endl;
    // // std::cout << output << std::endl << std::endl;
    // std::cout << "AFTER PHASE TWO ASSIGNMENT VALID:"
    //           << m_check_assignment_is_valid(m_table) << std::endl
    //           << std::endl;
    return output;
  }

private:
  Eigen::MatrixXd m_a;
  Eigen::MatrixXd m_La;
  Eigen::VectorXd m_b;
  Eigen::VectorXd m_Lb;
  Eigen::MatrixXd m_c;
  Eigen::MatrixXd m_table;
  Eigen::MatrixXd m_phase_one_table;
  std::vector<int> m_basis;
  Eigen::VectorXd m_objective;
  int64_t m_num_artificial;
  std::vector<std::string> types;
  int m_num_variables;

  void m_build_phase_one_table() {
    // Extend rhs vector by one and add zero

    m_push_back(m_b, 0);

    // Stack phase 1 objective function below constraint vector
    Eigen::MatrixXd stacked;
    // stacked.resize(m_a.rows() + m_c.cols(), m_a.cols());
    m_v_stack(
        m_a, m_make_vector_with_one_non_zero_elem(0, 0, m_a.cols()).transpose(),
        stacked);

    // Create basis tracking
    m_basis.resize(m_a.rows());

    // Create objective vector that will be used in phase two of simplex
    m_objective.resize(m_c.size() + m_a.rows() + 2, 1);
    m_objective.setZero();
    m_objective.topRows(m_c.size()) = m_c;
    m_objective(m_objective.size() - 2) = 1;
    // std::cout << "Objective: \n" << m_objective << std::endl;
    //  Add slack variables
    Eigen::MatrixXd stacked2 = stacked;
    auto height = stacked2.rows();
    std::vector<int64_t> greater_than_rows;

    for (auto i = 0; i < m_a.rows(); ++i) {
      stacked = stacked2;
      auto vec = m_make_vector_with_one_non_zero_elem(
          i, (types[i] == "G" ? -1 : 1), height);

      m_h_stack(stacked, vec, stacked2);
      if (types[i] == "G") {

        greater_than_rows.push_back(i);
      } else {
        m_basis[i] = stacked2.cols() - 1;
      }

      // std::cout << "5" << std::endl;
    }
    // std::cout << "Done adding slack variables" << std::endl;

    // Add in artificial variables
    m_num_artificial = 0;
    for (auto i = 0; i < greater_than_rows.size(); ++i) {
      stacked = stacked2;
      auto row = greater_than_rows[i];
      auto vec = m_make_vector_with_one_non_zero_elem(row, 1, height);
      vec(height - 1) = 1;
      m_h_stack(stacked, vec, stacked2);
      m_basis[row] = stacked2.cols() - 1;
      m_num_artificial++;
    }

    // std::cout << "Initial Basis: " << m_basis << std::endl;
    //  std::cout << "m_b size: " << m_b.size() << std::endl;
    //  std::cout << "m_b: " << m_b << std::endl;
    //  std::cout << "table size: " << stacked2.rows() << std::endl;
    // Makes sure that every column that has a basic variable is actually basic
    // std::cout << "Done resizing" << std::endl;
    m_h_stack(stacked2, m_b, m_phase_one_table);
    // std::cout << "Table before correcting basic columns: \n"
    //           << m_phase_one_table << std::endl
    //           << std::endl;
    // std::cout << "Done stacking the two:" << std::endl;
    for (auto i = 0; i < m_basis.size(); ++i) {
      auto col = m_basis[i];
      Eigen::VectorXd col_copy = m_phase_one_table.col(col);

      col_copy(i) = 0;
      if ((col_copy.array() != 0).any()) {
        // std::cout << "Basis Row: " << i << std::endl;
        // std::cout << "Basis Col: " << col << std::endl;
        // std::cout << "Before elimination:\n"
        //           << m_phase_one_table.col(col) << std::endl;
        m_elimination(i, col, m_phase_one_table);
        // std::cout << "After elimination:\n"
        //           << m_phase_one_table.col(col) << std::endl;
      }
    }

    // std::cout << m_phase_one_table << std::endl << std::endl;
  }

  void phaseOneSimplexSolver() {
    // std::cout << "Num artificial: \n" << m_num_artificial << std::endl;
    // std::cout << "Initial Table: \n" << m_phase_one_table << std::endl;
    bool exists_negative_in_last_row =
        (m_phase_one_table(m_phase_one_table.rows() - 1,
                           Eigen::seq(0, Eigen::placeholders::last - 1))
             .array() < -TOLERANCE)
            .any();
    int counter = 0;
    while (exists_negative_in_last_row) {
      bool is_feasible = m_check_assignment_is_valid(m_phase_one_table);
      std::cout << "Feasibility Check: " << is_feasible << std::endl;
      if (is_feasible)
        break;
      std::cout << "After phase one1: \n"
                << m_phase_one_table << std::endl
                << std::endl;

      std::cout << "Phase One: " << std::endl
                << m_phase_one_table(m_phase_one_table.rows() - 1,
                                     m_phase_one_table.cols() - 1)
                << std::endl;
      Eigen::Index pivot_column = m_find_pivot_column(
          m_phase_one_table.row(m_phase_one_table.rows() - 1));
      std::cout << "Pivot Column: " << pivot_column << std::endl;
      int pivot_row = m_find_pivot_row(pivot_column, m_phase_one_table);
      std::cout << "Pivot Row: " << pivot_row << std::endl;
      m_pivot(pivot_row, pivot_column, m_phase_one_table);
      exists_negative_in_last_row =
          (m_phase_one_table(m_phase_one_table.rows() - 1,
                             Eigen::seq(0, last - 1))
               .array() < -TOLERANCE)
              .any();
      counter++;
    }

    std::cout << "Phase ONe table after: \n"
              << m_phase_one_table << std::endl
              << std::endl;
    // Remove artificial variables from the basis
    // std::cout << "m_phase_one_table shape: (" << m_phase_one_table.rows()
    //           << ", " << m_phase_one_table.cols() << ")" << std::endl;
    // std::cout << "Num artificial: " << m_num_artificial << std::endl;
    auto numColsInTable = m_phase_one_table.cols();
    for (int i = m_basis.size() - 1; i >= 0; i--) {
      // std::cout << "Index: " << i << std::endl;
      // std::cout << "Basis column: " << m_basis[i] << std::endl;
      // std::cout << "Basis column is artificial: "
      //           << (m_basis[i] >= m_phase_one_table.cols() -
      //           m_num_artificial)
      //           << std::endl;
      if (m_basis[i] >= m_phase_one_table.cols() - (m_num_artificial + 1)) {
        auto row = m_phase_one_table.row(i);
        bool foundOne = false;
        for (int j = 0; j < row.size(); ++j) {
          if (row(j) != 0 &&
              (std::find(m_basis.begin(), m_basis.end(), row(j)) ==
               m_basis.end()) &&
              j <= (numColsInTable - (m_num_artificial + 2))) {
            m_pivot(i, j, m_phase_one_table);
            foundOne = true;
            break;
          }
        }
        if (foundOne == false) {
          std::cout
              << "WAS UNABLE TO FIND VARIABLE TO REPLACE ARTIFICIAL VARIABLE"
              << std::endl;
        }
      }
    }

    // std::cout << "Basis: \n" << m_basis << std::endl;
    //  std::cout << "Basic column 26: \n"
    //            << m_phase_one_table.col(26) << std::endl;
    // std::cout << "Final Table: \n" << m_phase_one_table << std::endl;

    // std::cout << "Phase One: " << std::endl
    //           << m_phase_one_table(m_phase_one_table.rows() - 1,
    //                                m_phase_one_table.cols() - 1)
    //           << std::endl;

    // std::cout << "After phase one Counter: " << counter << std::endl;

    // std::cout << "After phase one1: \n"
    //           << m_phase_one_table << std::endl
    //           << std::endl;
    auto temp8 = m_phase_one_table.cols() - 1;
    Eigen::VectorXd rhs =
        m_phase_one_table(Eigen::seq(0, Eigen::placeholders::last - 1),
                          Eigen::placeholders::last);
    Eigen::MatrixXd temp = m_phase_one_table(
        Eigen::seq(0, Eigen::placeholders::last - 1),
        Eigen::seq(0, Eigen::placeholders::last - (m_num_artificial + 1)));
    auto vec = m_make_vector_with_one_non_zero_elem(0, 0, temp.rows());
    Eigen::MatrixXd temp2;
    m_h_stack(temp, vec, temp2);
    // std::cout << "stacked the vector of zeros" << std::endl;

    m_h_stack(temp2, static_cast<Eigen::MatrixXd>(rhs), temp);
    // std::cout << "stacked the B" << std::endl;
    // std::cout << "Objective Size: " << m_objective.size() << std::endl;
    m_v_stack(temp, m_objective.transpose(), m_table);

    // std::cout << "m_table shape: (" << m_table.rows() << ", " <<
    // m_table.cols()
    //           << ")" << std::endl;
    // std::cout << "Basis: [ ";
    for (auto i = 0; i < m_basis.size(); ++i) {
      // std::cout << m_basis[i] << ", ";
    }
    // std::cout << "] ";
    // std::cout << "Last column before artificial: "
    //           << temp8 - (m_num_artificial + 1) << std::endl;

    for (auto i = 0; i < m_basis.size(); ++i) {
      auto col = m_basis[i];
      Eigen::VectorXd col_copy = m_table.col(col);

      col_copy(i) = 0;
      if ((col_copy.array() != 0).any()) {
        // std::cout << "Basis Row: " << i << std::endl;
        // std::cout << "Basis Col: " << col << std::endl;
        // std::cout << "Before elimination:\n"
        //           << m_phase_one_table.col(col) << std::endl;
        m_elimination(i, col, m_table);
        // std::cout << "After elimination:\n"
        //           << m_phase_one_table.col(col) << std::endl;
      }
    }
    // std::cout << "AFTER PHASE ONE ASSIGNMENT VALID:"
    //           << m_check_assignment_is_valid(m_table) << std::endl
    //           << std::endl;
  }

  Eigen::VectorXd m_make_vector_with_one_non_zero_elem(Eigen::Index position,
                                                       double value,
                                                       int64_t size) {
    Eigen::VectorXd vec = Eigen::VectorXd::Zero(size);
    // std::cout << "Before: " << position << ", " << size << std::endl;
    vec(position) = value;
    // std::cout << "After here" << std::endl;
    return vec;
  }

  bool m_check_assignment_is_valid(Eigen::Ref<Eigen::MatrixXd> table) {
    Eigen::VectorXd output(m_num_variables, 1);
    output.setZero();
    for (int i = 0; i < m_num_variables; i++) {
      auto [hot, index] = m_is_one_hot(table.col(i));

      if (hot) {
        output(i) = table(index, table.cols() - 1);
      } else {
        output(i) = 0;
      }
    }
    // std::cout << "m_a shape: (" << m_a.rows() << ", " << m_a.cols() << ")"
    //           << std::endl;
    // std::cout << "output shape: (" << output.rows() << ", " << output.cols()
    //           << ")" << std::endl;
    // std::cout << "m_b shape: ("
    //           << m_b(Eigen::seq(0, Eigen::placeholders::last - 1),
    //                  Eigen::placeholders::all)
    //                  .rows()
    //           << ", "
    //           << m_b(Eigen::seq(0, Eigen::placeholders::last - 1),
    //                  Eigen::placeholders::all)
    //                  .cols()
    //           << ")" << std::endl;
    // std::cout << "m_La:  " << std::endl << m_La << std::endl;
    // std::cout << "output:  " << std::endl << output << std::endl;
    // std::cout << "m_Lb:  " << std::endl << m_Lb << std::endl << std::endl;
    // std::cout << "product:  " << std::endl
    //           << (m_La * output) << std::endl
    //           << std::endl;
    return ((m_La * output).array() <= m_Lb.array() + 1e-8).all();
  }

  void m_pivot(Eigen::Index pivot_row, Eigen::Index pivot_column,
               Eigen::Ref<Eigen::MatrixXd> table) {
    // std::cout << "Dimensions position: (" << m_table.rows() << ", "
    //           << m_table.cols() << ")" << std::endl;
    //  std::cout << "Pivot position: (" << pivot_row << ", " << pivot_column <<
    //  ")"
    //            << std::endl;
    auto pivot_elem = table(pivot_row, pivot_column);
    // std::cout << "checkpoint0" << std::endl;
    table.row(pivot_row) = table.row(pivot_row) / pivot_elem;
    // std::cout << "checkpoint1" << std::endl;
    m_elimination(pivot_row, pivot_column, table);
    // std::cout << "checkpoint3" << std::endl;
    m_basis[pivot_row] = pivot_column;
    // std::cout << "checkpoint4" << std::endl;
  }

  void m_elimination(Eigen::Index pivot_row, Eigen::Index pivot_column,
                     Eigen::Ref<Eigen::MatrixXd> table) {
    for (int i = 0; i < table.rows(); i++) {
      if (i != pivot_row) {

        double factor = table(i, pivot_column);
        if (factor != 0)
          table.row(i) -= factor * table.row(pivot_row);
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
  void m_push_back(Eigen::VectorXd &vec, double value) {
    auto n = vec.size();
    vec.conservativeResize(vec.size() + 1);
    // vec.conservativeResize(n + 1);
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
  void m_v_stack(Eigen::MatrixXd matrix1, Eigen::MatrixXd matrix2,
                 Eigen::MatrixXd &out) {
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
  void m_h_stack(Eigen::MatrixXd matrix1, Eigen::MatrixXd matrix2,
                 Eigen::MatrixXd &out) {
    // std::cout << "HELLO" << std::endl;
    // std::cout << "HELLO2" << std::endl;
    out.resize(matrix1.rows(), matrix1.cols() + matrix2.cols());
    out << matrix1, matrix2;
    return;
  }

  int m_argmin(const Eigen::VectorXd v) {

    return std::distance(v.data(),
                         std::min_element(v.data(), v.data() + v.size()));
  }
  int m_find_pivot_column(const Eigen::VectorXd v) {
    for (int j = 0; j < v.size() - 1; ++j) {
      if (v(j) < -TOLERANCE) {
        return j;
      }
    }

    // std::cout << "Returning from finding pivot column" << std::endl;
    return -1;
  }
  int m_find_pivot_row(const Eigen::Index pivot_column,
                       Eigen::Ref<Eigen::MatrixXd> table) {
    // std::cout << "Entering from finding pivot row" << std::endl;

    // std::cout << "Inside Pivot Row: " << std::endl;
    auto rhs = table.col(table.cols() - 1).array();
    // std::cout << "Inside Pivot Row2: " << std::endl;
    // std::cout << "PIVOT COLUMN: " << std::endl
    //           << pivot_column << std::endl
    //           << std::endl;
    // std::cout << "m_table shape: (" << m_table.rows() << ", " <<
    // m_table.cols()
    //           << ")" << std::endl;
    auto col = table.col(pivot_column).array();
    // std::cout << "Inside Pivot Row3: " << std::endl;
    // std::cout << "RHS: " << std::endl << rhs << std::endl << std::endl;
    Eigen::Index n = table.rows() - 1;
    Eigen::ArrayXd ratio =
        (col > 0).select(rhs / col, std::numeric_limits<double>::infinity());

    // std::cout << "After finding the ratios" << std::endl;
    Eigen::VectorXd ratio_segment = ratio.segment(0, n);
    // std::cout << "Ratio Segment1: \n" << ratio_segment << std::endl;
    ratio_segment =
        (ratio_segment.array() > -TOLERANCE)
            .select(ratio_segment, std::numeric_limits<double>::infinity());
    double min_ratio = ratio_segment.minCoeff();
    // std::cout << "Ratio : " << std::endl << ratio_segment << std::endl;
    // for (int i = 0; i < n; ++i) {
    //   if (col(i) > 0) {
    //     // std::cout << "Positive Elem : " << std::endl << col(i) <<
    //     std::endl;
    //     //  ratios(i) = rhs(i) / col(i);
    //     //  positive_rows.push_back(i);
    //   } else {
    //     // ratios(i) = std::numeric_limits<double>::infinity();
    //   }
    // }
    std::vector<int> candidate_rows;
    for (int i = 0; i < ratio_segment.size(); ++i) {
      // std::cout << "Ratio Segment: " << ratio_segment(i) << std::endl;
      if (std::abs(ratio_segment(i) - min_ratio) < TOLERANCE) {
        candidate_rows.push_back(i);
      }
    }
    if (candidate_rows.size() == 1)
      return candidate_rows[0];
    // std::cout << "After finding candidate : " << candidate_rows.size()
    //           << std::endl;
    // std::cout << "Basis: " << std::endl << m_basis << std::endl;
    double min_index = std::numeric_limits<double>::max();
    int leaving_row = -1;
    // std::cout << "After finding basis size: " << std::endl
    //           << m_basis << std::endl;
    for (int row : candidate_rows) {
      // std::cout << "row: " << row << std::endl;
      //  std::cout << "Min index: " << min_index << std::endl;
      //  std::cout << (m_basis(row) < min_index) << std::endl << std::endl;
      //  std::cout << "After condition" << std::endl;
      if (m_basis[row] < min_index) {
        // std::cout << "Inside if" << std::endl;
        min_index = m_basis[row];
        leaving_row = row;
      }
    }
    // std::cout << "Returning from finding pivot row" << std::endl;
    return leaving_row;
  }
};

PYBIND11_MODULE(linear, handle) {
  handle.doc() = "This is the doc";
  py::class_<LP>(handle, "LinearProgram")
      .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, bool,
                    std::vector<std::string>>())
      .def("simplexSolver", &LP::simplexSolver);
}
