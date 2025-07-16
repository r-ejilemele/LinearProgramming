#include "main.h"

#define TOLERANCE 1e-170

using HighPrecision = mpfr::mpreal;
using MatrixXmp = Eigen::Matrix<HighPrecision, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXmp = Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, 1>;
using ArrayXmp = Eigen::Array<mpfr::mpreal, Eigen::Dynamic, 1>;
using Eigen::placeholders::last;
namespace py = pybind11;

float some_fn(float arg1, float arg2) { return arg1 + arg2; }

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
  os << "[";
  for (size_t i = 0; i < v.size(); ++i) {
    os << v[i];
    if (i != v.size() - 1) {
      os << ", ";
    }
  }
  os << "]";
  return os;
}

// Convert 2D numpy array of doubles to Eigen::Matrix<mpreal, ...>
MatrixXmp numpy_to_mpreal_matrix(py::array_t<double> input) {
  auto buf = input.unchecked<2>(); // 2D access
  MatrixXmp result(buf.shape(0), buf.shape(1));
  for (pybind11::ssize_t i = 0; i < buf.shape(0); i++) {
    for (pybind11::ssize_t j = 0; j < buf.shape(1); j++) {
      result(i, j) = buf(i, j);
    }
  }
  return result;
}

// Convert 1D numpy array of doubles to Eigen::Matrix<mpreal, Dynamic, 1>
VectorXmp numpy_to_mpreal_vector(py::array_t<double> input) {
  auto buf = input.unchecked<1>(); // 1D access
  VectorXmp result(buf.shape(0));
  for (pybind11::ssize_t i = 0; i < buf.shape(0); i++) {
    result(i) = buf(i);
  }
  return result;
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
   * @param constraint Holds whether a constraint is greater than or less than
   */
  LP(MatrixXmp A, MatrixXmp B, MatrixXmp C, bool maxim,
     std::vector<std::string> constraint) {
    m_constraints = constraint;
    m_permutation_matrix = MatrixXmp::Identity(A.rows() + 1, A.rows() + 1);
    m_a = A;
    m_Ca = A;
    m_La.resize(A.rows(), A.cols());
    m_b = B;
    m_Cb = B;
    m_Cb.resize(B.rows(), B.cols());
    m_Lb.resize(B.rows(), B.cols());
    m_c = (maxim ? -1 : 1) * C;
    types = constraint;
    VectorXmp rowMax = m_a.rowwise().maxCoeff();
    rowMax = rowMax.unaryExpr([](const mpfr::mpreal &x) {
      if (x == 0)
        return mpfr::mpreal(1);
      if (x > 1e3 || x < 1e-3) {
        double log10_val = std::log10(x.toDouble());
        double rounded = std::round(log10_val);
        return mpfr::pow(mpfr::mpreal(10.0), rounded);
      }
      return mpfr::mpreal(1);
    });
    Eigen::DiagonalMatrix<mpfr::mpreal, Eigen::Dynamic> rowMaxDiag(rowMax);
    rowMaxDiag = rowMaxDiag.inverse();
    MatrixXmp a = rowMaxDiag * m_a;
    MatrixXmp b = rowMaxDiag * m_b;
    for (auto i = 0; i < A.rows(); ++i) {
      if (constraint[i] == "G") {
        m_Lb(i) = -b(i);
        // m_Cb(i) = -b(i);
        m_La.row(i) = -a.row(i);
        // m_Ca.row(i) = -a.row(i);
      } else {
        m_Lb(i) = b(i);
        m_La.row(i) = a.row(i);
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
  Eigen::VectorXd simplexSolverDouble() {
    VectorXmp result_mp =
        simplexSolver(); // original function returning VectorXmp

    // Create a double VectorXd of the same size
    Eigen::VectorXd result_double(result_mp.size());

    // Convert each element from mpfr::mpreal to double
    for (int i = 0; i < result_mp.size(); ++i) {
      result_double[i] =
          result_mp[i].toDouble(); // or static_cast<double>(result_mp[i])
    }
    return result_double;
  }

  /**
   * @brief  an Instance of a Linear Program
   *
   *
   * @returns the feasible solution this Linear Program using the Simplex
   * algorithm
   */
  VectorXmp simplexSolver() {
    phaseOneSimplexSolver();
    // std::cout << "Basis: " << std::endl << m_basis << std::endl;
    // std::cout << "Table: \n" << m_table << std::endl;

    bool exists_negative_in_last_row =
        (m_table(m_table.rows() - 1, Eigen::seq(0, last - 1)).array() <
         -TOLERANCE)
            .any();
    int counter = 0;
    std::cout << "Table: \n" << m_table << std::endl;
    while (exists_negative_in_last_row) {
      std::cout << std::endl
                << "Phase 2: "
                << m_table(m_table.rows() - 1, m_table.cols() - 1) << std::endl;
      //   std::cout << "Table: \n" << m_table << std::endl;
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
    // std::cout << "Final Table: \n" << m_table << std::endl;

    std::cout << std::endl
              << std::endl
              << m_table(m_table.rows() - 1, m_table.cols() - 1) << std::endl
              << std::endl;
    m_table = m_permutation_matrix.transpose() * m_table;
    m_La = m_permutation_matrix(Eigen::seq(0, Eigen::placeholders::last - 1),
                                Eigen::seq(0, Eigen::placeholders::last - 1))
               .transpose() *
           m_La;
    m_Lb = m_permutation_matrix(Eigen::seq(0, Eigen::placeholders::last - 1),
                                Eigen::seq(0, Eigen::placeholders::last - 1))
               .transpose() *
           m_Lb;
    VectorXmp output(m_num_variables, 1);
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
    // std::cout << "permutation shape: (" << m_permutation_matrix.rows() << ",
    // "
    //           << m_permutation_matrix.cols() << ")" << std::endl;
    // std::cout << "output shape: (" << output.rows() << ", " << output.cols()
    //           << ")" << std::endl;
    // std::cout << "c shape: (" << m_c.rows() << ", " << m_c.cols() << ")"
    //           << std::endl;
    std::cout << "FINAL RESULT: " << m_c.transpose() * output << std::endl;
    // refine_solution();
    return output;
  }

private:
  std::vector<std::string> m_constraints;
  MatrixXmp m_a;
  MatrixXmp m_La;
  MatrixXmp m_Ca;
  VectorXmp m_b;
  VectorXmp m_Lb;
  VectorXmp m_Cb;
  MatrixXmp m_c;
  MatrixXmp m_table;
  MatrixXmp m_phase_one_table;
  MatrixXmp m_permutation_matrix;
  std::vector<int> m_basis;
  VectorXmp m_objective;
  int64_t m_num_artificial;
  std::vector<std::string> types;
  int m_num_variables;

  void m_build_phase_one_table() {
    // Extend rhs vector by one and add zero

    m_push_back(m_b, 0);

    // Stack phase 1 objective function below constraint vector
    MatrixXmp stacked;
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
    MatrixXmp stacked2 = stacked;
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
      VectorXmp col_copy = m_phase_one_table.col(col);

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
    VectorXmp rhs =
        m_phase_one_table(Eigen::seq(0, Eigen::placeholders::last - 1),
                          Eigen::placeholders::last);
    MatrixXmp temp = m_phase_one_table(
        Eigen::seq(0, Eigen::placeholders::last - 1),
        Eigen::seq(0, Eigen::placeholders::last - (m_num_artificial + 1)));
    auto vec = m_make_vector_with_one_non_zero_elem(0, 0, temp.rows());
    MatrixXmp temp2;
    m_h_stack(temp, vec, temp2);
    // std::cout << "stacked the vector of zeros" << std::endl;

    m_h_stack(temp2, static_cast<MatrixXmp>(rhs), temp);
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
      VectorXmp col_copy = m_table.col(col);

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
  void refine_solution() {
    int size = m_a.rows() + m_a.cols();
    MatrixXmp temp = m_Ca;
    MatrixXmp Identity = MatrixXmp::Identity(m_Ca.rows(), m_Ca.rows());
    MatrixXmp A = m_Ca;
    m_h_stack(temp, Identity, A);
    // std::cout << "m_La shape: (" << m_La.rows() << ", " << m_La.cols() << ")"
    //           << std::endl;
    // std::cout << "A rows: " << A.rows() << std::endl;

    VectorXmp b = m_Cb;
    for (int i = 0; i < m_constraints.size(); i++) {
      if (m_constraints[i] == "G") {
        A.col(m_Ca.cols() + i) = -A.col(m_Ca.cols() + i);
      }
    }
    VectorXmp z(size, 1);
    for (int i = 0; i < size; i++) {
      auto [hot, index] = m_is_one_hot(m_table.col(i));

      if (hot) {
        z(i) = m_table(index, m_table.cols() - 1);
      } else {
        z(i) = 0;
      }
    }
    // VectorXmp x = z.head(m_num_variables);
    // VectorXmp s = z.tail(m_a.rows());

    std::cout << "b shape: (" << b.rows() << ", " << b.cols() << ")"
              << std::endl;
    std::cout << "A shape: (" << A.rows() << ", " << A.cols() << ")"
              << std::endl;
    std::cout << "z shape: (" << z.rows() << ", " << z.cols() << ")"
              << std::endl;
    // std::cout << "s shape: (" << s.rows() << ", " << s.cols() << ")"
    //           << std::endl;
    MatrixXmp residual = b - A * z;

    std::cout << "Residual norm: " << residual.norm() << std::endl;
    std::cout << "Residual norm: \n" << residual << std::endl;
    while (residual.norm() > 1e1) {
      std::cout << "Residual norm: " << residual.norm() << std::endl;
      MatrixXmp c = A.colPivHouseholderQr().solve(residual);
      z += c;
      // z.cwiseMax(-TOLERANCE);
      residual = b - A * z;
      std::cout << "Residual norm: " << residual.norm() << std::endl;
    }
    std::cout << "REFINED OUTPUT:" << m_c.transpose() * z.head(m_num_variables)
              << std::endl;
  }
  VectorXmp m_make_vector_with_one_non_zero_elem(Eigen::Index position,
                                                 double value, int64_t size) {
    VectorXmp vec = VectorXmp::Zero(size);
    // std::cout << "Before: " << position << ", " << size << std::endl;
    vec(position) = value;
    // std::cout << "After here" << std::endl;
    return vec;
  }

  bool m_check_assignment_is_valid(Eigen::Ref<MatrixXmp> table) {
    VectorXmp output(m_num_variables, 1);
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
               Eigen::Ref<MatrixXmp> table) {
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
                     Eigen::Ref<MatrixXmp> table) {
    for (int i = 0; i < table.rows(); i++) {
      if (i != pivot_row) {

        mpfr::mpreal factor = table(i, pivot_column);
        if (factor != 0)
          table.row(i) -= factor * table.row(pivot_row);
      }
    }
    return;
  }

  std::pair<bool, int> m_is_one_hot(const Eigen::Ref<VectorXmp> v) {

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
  void m_push_back(VectorXmp &vec, double value) {
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
  void m_v_stack(MatrixXmp matrix1, MatrixXmp matrix2, MatrixXmp &out) {
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
  void m_h_stack(MatrixXmp matrix1, MatrixXmp matrix2, MatrixXmp &out) {
    // std::cout << "HELLO" << std::endl;
    // std::cout << "HELLO2" << std::endl;
    out.resize(matrix1.rows(), matrix1.cols() + matrix2.cols());
    out << matrix1, matrix2;
    return;
  }

  int m_argmin(const VectorXmp v) {

    return std::distance(v.data(),
                         std::min_element(v.data(), v.data() + v.size()));
  }
  int m_find_pivot_column(const VectorXmp v) {
    for (int j = 0; j < v.size() - 1; ++j) {
      if (v(j) < -TOLERANCE) {
        return j;
      }
    }

    // std::cout << "Returning from finding pivot column" << std::endl;
    return -1;
  }
  int m_find_pivot_row(const Eigen::Index pivot_column,
                       Eigen::Ref<MatrixXmp> table) {
    std::cout << "Entering from finding pivot row" << std::endl;

    // std::cout << "Inside Pivot Row: " << std::endl;
    auto rhs = table.col(table.cols() - 1).array();
    // std::cout << "Inside Pivot Row2: " << std::endl;
    // std::cout << "PIVOT COLUMN: " << std::endl
    //           << pivot_column << std::endl
    //           << std::endl;
    // std::cout << "m_table shape: (" << m_table.rows() << ", " <<
    // m_table.cols()
    //           << ")" << std::endl;
    ArrayXmp col = table.col(pivot_column).array();
    // std::cout << "Inside Pivot Row3: " << std::endl;
    // std::cout << "RHS: " << std::endl << rhs << std::endl << std::endl;
    Eigen::Index n = table.rows() - 1;
    ArrayXmp ratio =
        (col > -TOLERANCE)
            .select(rhs / col, std::numeric_limits<double>::infinity());

    // std::cout << "After finding the ratios" << std::endl;
    VectorXmp ratio_segment = ratio.segment(0, n);
    // std::cout << "Ratio Segment1: \n" << ratio_segment << std::endl;
    ratio_segment = (ratio_segment.array() > -TOLERANCE)
                        .select(ratio_segment,
                                std::numeric_limits<mpfr::mpreal>::infinity());
    mpfr::mpreal min_ratio = ratio_segment.minCoeff();
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
      if ((mpfr::abs(ratio_segment(i)) - min_ratio) < TOLERANCE) {
        candidate_rows.push_back(i);
      }
    }
    if (candidate_rows.size() == 1)
      return candidate_rows[0];
    // std::cout << "After finding candidate : " << candidate_rows.size()
    //           << std::endl;
    // std::cout << "Basis: " << std::endl << m_basis << std::endl;
    mpfr::mpreal min_index = std::numeric_limits<mpfr::mpreal>::max();
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
    std::cout << "Returning from finding pivot row" << std::endl;
    int best_idx = leaving_row;
    col = col.abs();
    mpfr::mpreal best_val = col(leaving_row);
    for (int i = leaving_row; i < ratio_segment.size(); ++i) {
      if (ratio_segment(i) > -TOLERANCE && col(i) > best_val) {
        best_val = col(i);
        best_idx = i;
      }
    }
    std::swap(m_basis[leaving_row], m_basis[best_idx]);
    table.row(leaving_row).swap(table.row(best_idx));
    m_La.row(leaving_row).swap(m_La.row(best_idx));
    m_Lb.row(leaving_row).swap(m_Lb.row(best_idx));
    m_permutation_matrix.row(leaving_row)
        .swap(m_permutation_matrix.row(best_idx));
    return best_idx;
    // return leaving_row;
  }
};

PYBIND11_MODULE(linear, handle) {
  mpfr::mpreal::set_default_prec(128);
  handle.doc() = "This is the doc";
  py::class_<LP>(handle, "LinearProgram")
      .def(py::init([](py::array_t<double> A_np, py::array_t<double> b_np,
                       py::array_t<double> C_np, bool flag,
                       std::vector<std::string> names) {
        MatrixXmp A = numpy_to_mpreal_matrix(A_np);
        VectorXmp b = numpy_to_mpreal_matrix(b_np);
        MatrixXmp C = numpy_to_mpreal_matrix(C_np);

        return std::make_unique<LP>(A, b, C, flag, names);
      }))
      .def("simplexSolver", &LP::simplexSolverDouble);
}
