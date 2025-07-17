#include "main.h"

#define TOLERANCE 1e-100

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
        m_La.row(i) = -a.row(i);
      } else {
        m_Lb(i) = b(i);
        m_La.row(i) = a.row(i);
      }
    }

    m_build_phase_one_table();
    m_num_variables = m_c.size();
  }

  Eigen::VectorXd simplexSolverDouble() {
    VectorXmp result_mp = simplexSolver();

    // Create a double VectorXd of the same size
    Eigen::VectorXd result_double(result_mp.size());

    // Convert each element from mpfr::mpreal to double
    for (int i = 0; i < result_mp.size(); ++i) {
      result_double[i] = result_mp[i].toDouble();
    }
    return result_double;
  }

  /**
   * @brief Phase two of the simplex algorithm to find the optimal solution
   *
   * @returns the feasible solution this Linear Program using the Simplex
   * algorithm
   */
  VectorXmp simplexSolver() {
    phaseOneSimplexSolver();

    // Stopping condition: No negative elements in last row
    bool exists_negative_in_last_row =
        (m_table(m_table.rows() - 1, Eigen::seq(0, last - 1)).array() <
         -TOLERANCE)
            .any();
    int counter = 0;
    while (exists_negative_in_last_row) {
      Eigen::Index pivot_column =
          m_find_pivot_column(m_table.row(m_table.rows() - 1));

      int pivot_row = m_find_pivot_row(pivot_column, m_table);

      m_pivot(pivot_row, pivot_column, m_table);

      exists_negative_in_last_row =
          (m_table(m_table.rows() - 1, Eigen::seq(0, last - 1)).array() <
           -TOLERANCE)
              .any();
      counter++;
    }

    // Undo any row swaps that were done during pivoting
    m_table = m_permutation_matrix.transpose() * m_table;
    m_La = m_permutation_matrix(Eigen::seq(0, Eigen::placeholders::last - 1),
                                Eigen::seq(0, Eigen::placeholders::last - 1))
               .transpose() *
           m_La;
    m_Lb = m_permutation_matrix(Eigen::seq(0, Eigen::placeholders::last - 1),
                                Eigen::seq(0, Eigen::placeholders::last - 1))
               .transpose() *
           m_Lb;

    // Grab assignments out of the tableau
    VectorXmp output(m_num_variables, 1);
    for (int i = 0; i < m_num_variables; i++) {
      auto [hot, index] = m_is_one_hot(m_table.col(i));

      if (hot) {
        output(i) = m_table(index, m_table.cols() - 1);
      } else {
        output(i) = 0;
      }
    }

    std::cout << "FINAL RESULT: " << m_c.transpose() * output << std::endl;
    std::cout << "FINAL ASSIGNMENT: \n" << output << std::endl;
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

  /**
   * Builds the initial tableau for phase one of the simplex algorithm
   */
  void m_build_phase_one_table() {
    // Extend rhs vector by one and add zero

    m_push_back(m_b, 0);

    // Stack phase 1 objective function below constraint vector
    MatrixXmp stacked;
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
    }

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

    // Makes sure that every column that has a basic variable is actually basic
    m_h_stack(stacked2, m_b, m_phase_one_table);
    for (auto i = 0; i < m_basis.size(); ++i) {
      auto col = m_basis[i];
      VectorXmp col_copy = m_phase_one_table.col(col);

      col_copy(i) = 0;
      if ((col_copy.array() != 0).any()) {
        m_elimination(i, col, m_phase_one_table);
      }
    }
  }

  /**
   * @brief performs phase one of the simplex algorithm to find a feasible
   * solution
   */
  void phaseOneSimplexSolver() {

    // Stopping condition: No negative elements in last row or feasible solution
    // found
    bool exists_negative_in_last_row =
        (m_phase_one_table(m_phase_one_table.rows() - 1,
                           Eigen::seq(0, Eigen::placeholders::last - 1))
             .array() < -TOLERANCE)
            .any();

    while (exists_negative_in_last_row) {
      bool is_feasible = m_check_assignment_is_valid(m_phase_one_table);
      if (is_feasible)
        break;

      Eigen::Index pivot_column = m_find_pivot_column(
          m_phase_one_table.row(m_phase_one_table.rows() - 1));

      int pivot_row = m_find_pivot_row(pivot_column, m_phase_one_table);

      m_pivot(pivot_row, pivot_column, m_phase_one_table);

      exists_negative_in_last_row =
          (m_phase_one_table(m_phase_one_table.rows() - 1,
                             Eigen::seq(0, last - 1))
               .array() < -TOLERANCE)
              .any();
    }

    auto numColsInTable = m_phase_one_table.cols();
    // Removes artificial variables from basis if they are still in basis
    for (int i = m_basis.size() - 1; i >= 0; i--) {
      if (m_basis[i] >= m_phase_one_table.cols() - (m_num_artificial + 1)) {
        auto row = m_phase_one_table.row(i);
        for (int j = 0; j < row.size(); ++j) {
          if (row(j) != 0 &&
              (std::find(m_basis.begin(), m_basis.end(), row(j)) ==
               m_basis.end()) &&
              j <= (numColsInTable - (m_num_artificial + 2))) {
            m_pivot(i, j, m_phase_one_table);
            break;
          }
        }
      }
    }

    // Rebuild phase two table
    VectorXmp rhs =
        m_phase_one_table(Eigen::seq(0, Eigen::placeholders::last - 1),
                          Eigen::placeholders::last);
    MatrixXmp temp = m_phase_one_table(
        Eigen::seq(0, Eigen::placeholders::last - 1),
        Eigen::seq(0, Eigen::placeholders::last - (m_num_artificial + 1)));
    auto vec = m_make_vector_with_one_non_zero_elem(0, 0, temp.rows());
    MatrixXmp temp2;
    m_h_stack(temp, vec, temp2);

    m_h_stack(temp2, static_cast<MatrixXmp>(rhs), temp);
    m_v_stack(temp, m_objective.transpose(), m_table);

    for (auto i = 0; i < m_basis.size(); ++i) {
      auto col = m_basis[i];
      VectorXmp col_copy = m_table.col(col);

      col_copy(i) = 0;
      if ((col_copy.array() != 0).any()) {
        m_elimination(i, col, m_table);
      }
    }
  }

  /**
   * @brief refines the solution found by the simplex algorithm to ensure it is
   * feasible and optimal
   *
   */
  void refine_solution() {
    int size = m_a.rows() + m_a.cols();
    MatrixXmp temp = m_Ca;
    MatrixXmp Identity = MatrixXmp::Identity(m_Ca.rows(), m_Ca.rows());
    MatrixXmp A = m_Ca;
    m_h_stack(temp, Identity, A);

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

    MatrixXmp residual = b - A * z;

    while (residual.norm() > 1e1) {
      MatrixXmp c = A.colPivHouseholderQr().solve(residual);
      z += c;
      residual = b - A * z;
    }
  }
  VectorXmp m_make_vector_with_one_non_zero_elem(Eigen::Index position,
                                                 double value, int64_t size) {
    VectorXmp vec = VectorXmp::Zero(size);
    vec(position) = value;
    return vec;
  }

  /**
   * @brief checks if a feasible assignment has been found in the tableau
   *
   * @param table the tableau to check
   * @returns true if the assignment is feasible, false otherwise
   */
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
    return ((m_La * output).array() <= m_Lb.array() + 1e-8).all();
  }

  /**
   * @brief performs a pivot operation on the tableau
   * @param pivot_row the row that will be used as the pivot
   * @param pivot_column the column that will be used as the pivot
   * @param table the tableau that will be modified
   */
  void m_pivot(Eigen::Index pivot_row, Eigen::Index pivot_column,
               Eigen::Ref<MatrixXmp> table) {

    auto pivot_elem = table(pivot_row, pivot_column);
    table.row(pivot_row) = table.row(pivot_row) / pivot_elem;
    m_elimination(pivot_row, pivot_column, table);
    m_basis[pivot_row] = pivot_column;
  }

  /**
   * @brief performs m elimination on the tableau
   *
   * @param pivot_row the row that will be used to eliminate other rows
   * @param pivot_column the column that will be used to eliminate other rows
   * @param table the tableau that will be modified
   */
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

  /**
   * @brief checks if a vector is one-hot encoded
   * @param v the vector to check
   * @returns a pair where the first element is true if the vector is one-hot
   * encoded, and the second element is the index of the non-zero element (or -1
   * if not one-hot)
   */
  std::pair<bool, int> m_is_one_hot(const Eigen::Ref<VectorXmp> v) {

    bool found = false;
    int index = -1;
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
   */
  void m_push_back(VectorXmp &vec, double value) {
    auto n = vec.size();
    vec.conservativeResize(vec.size() + 1);
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
    out.resize(matrix1.rows(), matrix1.cols() + matrix2.cols());
    out << matrix1, matrix2;
    return;
  }

  /**
   * @brief finds the index of the minimum element in a vector
   * @param v the vector to search
   * @returns the index of the minimum element
   */
  int m_argmin(const VectorXmp v) {

    return std::distance(v.data(),
                         std::min_element(v.data(), v.data() + v.size()));
  }

  /**
   * @brief given a tableau, finds the pivot column
   * @param v the last row of the tableau that will be used to find the pivot
   * column
   * @returns the index of the pivot column, or -1 if there is no pivot
   */
  int m_find_pivot_column(const VectorXmp v) {
    for (int j = 0; j < v.size() - 1; ++j) {
      if (v(j) < -TOLERANCE) {
        return j;
      }
    }

    return -1;
  }

  /**
   * @brief given a pivot column and a tableau, finds the pivot row using
   *  partial pivoting
   *
   *@param pivot_column column in which the pivot will be found
   *@param table the tableau that will be used to find the pivot row
   *@returns the index of the pivot row
   */
  int m_find_pivot_row(const Eigen::Index pivot_column,
                       Eigen::Ref<MatrixXmp> table) {

    auto rhs = table.col(table.cols() - 1).array();
    ArrayXmp col = table.col(pivot_column).array();
    Eigen::Index n = table.rows() - 1;

    // Replace negative elements in pivot column with infinity
    ArrayXmp ratio =
        (col > -TOLERANCE)
            .select(rhs / col, std::numeric_limits<double>::infinity());

    // Replace negative ratios with infinity
    VectorXmp ratio_segment = ratio.segment(0, n);
    ratio_segment = (ratio_segment.array() > -TOLERANCE)
                        .select(ratio_segment,
                                std::numeric_limits<mpfr::mpreal>::infinity());

    mpfr::mpreal min_ratio = ratio_segment.minCoeff();

    // Candidate rows for Bland's rule of selecting the pivot row.
    std::vector<int> candidate_rows;

    for (int i = 0; i < ratio_segment.size(); ++i) {
      if ((mpfr::abs(ratio_segment(i)) - min_ratio) < TOLERANCE) {
        candidate_rows.push_back(i);
      }
    }

    // Selects candidate row with smallest index in basis
    mpfr::mpreal min_index = std::numeric_limits<mpfr::mpreal>::max();
    int leaving_row = -1;
    for (int row : candidate_rows) {
      if (m_basis[row] < min_index) {
        min_index = m_basis[row];
        leaving_row = row;
      }
    }

    // If there is a row with a larger pivot element below the pivot row, then
    // swap rows and use that as the leaving row instead
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
