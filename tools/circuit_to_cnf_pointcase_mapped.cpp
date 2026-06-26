#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Gate = std::array<int, 3>;

static int val(char c) {
  const std::string chars =
      "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?";
  auto p = chars.find(c);
  if (p == std::string::npos) return -1;
  return static_cast<int>(p);
}

static std::vector<Gate> parse_circuit(const std::string &path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("failed to open " + path);
  std::vector<Gate> gates;
  std::vector<int> w;
  int overflow = 0;
  char ch;
  while (in.get(ch)) {
    if (ch == ';') {
      if (!w.empty()) {
        if (w.size() != 3) throw std::runtime_error("bad gate");
        gates.push_back({w[0], w[1], w[2]});
        w.clear();
      }
      overflow = 0;
    } else if (ch == '~') {
      overflow++;
    } else if (ch == '\n' || ch == '\r' || ch == '\t' || ch == ' ') {
      continue;
    } else {
      int b = val(ch);
      if (b < 0) throw std::runtime_error("bad char");
      w.push_back(b + 83 * overflow);
      overflow = 0;
    }
  }
  if (!w.empty()) throw std::runtime_error("unterminated gate");
  return gates;
}

static std::vector<int> parse_map(const std::string &s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) out.push_back(std::stoi(item));
  }
  return out;
}

struct CnfWriter {
  std::ofstream out;
  uint64_t clauses = 0;

  explicit CnfWriter(const std::string &path) : out(path) {
    if (!out) throw std::runtime_error("failed to create " + path);
    out << "p cnf 0000000000 0000000000\n";
  }

  void clause(std::initializer_list<int> lits) {
    for (int lit : lits) out << lit << ' ';
    out << "0\n";
    clauses++;
  }
};

int main(int argc, char **argv) {
  if (argc != 8) {
    std::cerr << "usage: circuit_to_cnf_pointcase_mapped circuit out.cnf "
                 "low_bits total_wires flip_bit input_bit output_map_csv\n";
    return 2;
  }

  const int low_bits = std::stoi(argv[3]);
  const int total_wires = std::stoi(argv[4]);
  const int flip = std::stoi(argv[5]);
  const int bit = std::stoi(argv[6]);
  auto output_map = parse_map(argv[7]);
  if (low_bits <= 0 || total_wires < low_bits || flip < 0 || flip >= low_bits ||
      (bit != 0 && bit != 1) || static_cast<int>(output_map.size()) < low_bits) {
    throw std::runtime_error("bad arguments");
  }

  auto gates = parse_circuit(argv[1]);
  std::cerr << "parsed gates " << gates.size() << "\n";

  std::vector<int> cur(total_wires);
  int vars = 0;
  for (int w = 0; w < total_wires; w++) cur[w] = ++vars;

  CnfWriter cnf(argv[2]);

  for (int w = low_bits; w < total_wires; w++) cnf.clause({-cur[w]});

  for (const auto &g : gates) {
    int a0 = cur[g[0]];
    int b = cur[g[1]];
    int c = cur[g[2]];
    int z = ++vars;
    // z = a0 XOR (b OR !c)
    cnf.clause({b, -c, -a0, z});
    cnf.clause({b, -c, a0, -z});
    cnf.clause({-b, -a0, -z});
    cnf.clause({-b, a0, z});
    cnf.clause({c, -a0, -z});
    cnf.clause({c, a0, z});
    cur[g[0]] = z;
  }

  for (int w = 0; w < low_bits; w++) {
    int x = w + 1;
    int y = cur[output_map[w]];
    if (w == flip) {
      cnf.clause({bit ? x : -x});
      cnf.clause({bit ? -y : y});
    } else {
      int d = ++vars;
      cnf.clause({-x, -y, -d});
      cnf.clause({-x, y, d});
      cnf.clause({x, -y, d});
      cnf.clause({x, y, -d});
      cnf.clause({-d});
    }
  }

  cnf.out.flush();
  cnf.out.close();

  std::fstream patch(argv[2], std::ios::in | std::ios::out);
  std::ostringstream header;
  header << "p cnf " << vars << ' ' << cnf.clauses;
  std::string h = header.str();
  if (h.size() > 27) throw std::runtime_error("header too long");
  h.resize(27, ' ');
  h.push_back('\n');
  patch.seekp(0);
  patch << h;

  std::cerr << "vars " << vars << " clauses " << cnf.clauses << "\n";
  return 0;
}
