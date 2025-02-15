
#include <iostream>
#include <iomanip>
#include <chrono>

// Use long double for precision and to avoid overflow
using Real = long double;

Real calculate(long long iterations, long long param1, long long param2) {
  Real result = 1.0L; // Initialize with long double literal
  for (long long i = 1; i <= iterations; ++i) {
    //Direct calculation of inverse for efficiency
    result -= 1.0L / (i * param1 - param2);
    result += 1.0L / (i * param1 + param2);
  }
  return result;
}

int main() {
  auto start = std::chrono::high_resolution_clock::now();
  Real result = calculate(100000000, 4, 1) * 4.0L; // Use long double literal for final multiplication
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << std::fixed << std::setprecision(12) << "Result: " << result << std::endl;
  std::cout << "Execution Time: " << duration.count() / 1000.0 << " seconds" << std::endl;
  return 0;
}

