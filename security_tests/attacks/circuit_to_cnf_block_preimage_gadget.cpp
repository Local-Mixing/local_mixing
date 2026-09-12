#include <array>
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

static int hex_val(char c) {
  if ('0' <= c && c <= '9') return c - '0';
  if ('a' <= c && c <= 'f') return 10 + c - 'a';
  if ('A' <= c && c <= 'F') return 10 + c - 'A';
  return -1;
}

static std::vector<Gate> parse(const std::string &path) {
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

static int parse_int(const std::string &s, const char *name) {
  int out = 0;
  std::stringstream ss(s);
  ss >> out;
  if (!ss || out < 0) throw std::runtime_error(std::string("bad ") + name);
  return out;
}

static std::vector<int> parse_hex_bits(std::string s, int bits) {
  if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0) s = s.substr(2);
  if (s.empty()) throw std::runtime_error("empty hex value");
  std::vector<int> out(bits, 0);
  for (int bit = 0; bit < bits; bit++) {
    int hex_pos = static_cast<int>(s.size()) - 1 - bit / 4;
    if (hex_pos < 0) break;
    int hv = hex_val(s[hex_pos]);
    if (hv < 0) throw std::runtime_error("bad hex digit");
    out[bit] = (hv >> (bit % 4)) & 1;
  }
  for (int bit = bits; bit < static_cast<int>(s.size()) * 4; bit++) {
    int hex_pos = static_cast<int>(s.size()) - 1 - bit / 4;
    int hv = hex_val(s[hex_pos]);
    if (hv < 0) throw std::runtime_error("bad hex digit");
    if (((hv >> (bit % 4)) & 1) != 0) {
      throw std::runtime_error("hex value has non-zero bits above requested width");
    }
  }
  return out;
}

static void write_update_clauses(std::ofstream &out, int old_a, int b, int c,
                                 int new_a) {
  out << -old_a << ' ' << -b << ' ' << -new_a << " 0\n";
  out << -old_a << ' ' << c << ' ' << -new_a << " 0\n";
  out << old_a << ' ' << -b << ' ' << new_a << " 0\n";
  out << old_a << ' ' << c << ' ' << new_a << " 0\n";
  out << -old_a << ' ' << b << ' ' << -c << ' ' << new_a << " 0\n";
  out << old_a << ' ' << b << ' ' << -c << ' ' << -new_a << " 0\n";
}

int main(int argc, char **argv) {
  if (argc != 11 && argc != 12) {
    std::cerr
        << "usage: circuit_to_cnf_block_preimage_gadget F.txt out.cnf "
           "total_wires original_wires fixed_input_start fixed_input_bits "
           "fixed_input_hex output_start output_bits target_hex [aux_zero]\n";
    return 2;
  }

  auto gates = parse(argv[1]);
  int total_wires = parse_int(argv[3], "total wire count");
  int original_wires = parse_int(argv[4], "original wire count");
  int fixed_start = parse_int(argv[5], "fixed input start");
  int fixed_bits = parse_int(argv[6], "fixed input bit count");
  int output_start = parse_int(argv[8], "output start");
  int output_bits = parse_int(argv[9], "output bit count");
  bool aux_zero = false;
  if (argc == 12) {
    std::string flag = argv[11];
    aux_zero = flag == "1" || flag == "true" || flag == "--aux-zero";
  }

  if (total_wires <= 0 || original_wires <= 0 || original_wires > total_wires) {
    throw std::runtime_error("bad wire dimensions");
  }
  if (fixed_start + fixed_bits > total_wires || output_start + output_bits > total_wires) {
    throw std::runtime_error("fixed or output block outside wire range");
  }

  auto fixed = parse_hex_bits(argv[7], fixed_bits);
  auto target = parse_hex_bits(argv[10], output_bits);

  for (auto [a, b, c] : gates) {
    if (a < 0 || a >= total_wires || b < 0 || b >= total_wires || c < 0 ||
        c >= total_wires) {
      throw std::runtime_error("wire out of range");
    }
  }

  std::vector<int> state(total_wires);
  for (int i = 0; i < total_wires; i++) state[i] = i + 1;
  int next_var = total_wires + 1;
  long long aux_clauses = aux_zero ? (total_wires - original_wires) : 0;
  long long clause_count =
      6LL * gates.size() + fixed_bits + output_bits + aux_clauses;

  std::ofstream out(argv[2]);
  if (!out) throw std::runtime_error("failed to create output cnf");
  out << "p cnf " << (total_wires + static_cast<int>(gates.size())) << ' '
      << clause_count << "\n";

  for (int i = 0; i < fixed_bits; i++) {
    int lit = state[fixed_start + i];
    out << (fixed[i] ? lit : -lit) << " 0\n";
  }
  if (aux_zero) {
    for (int i = original_wires; i < total_wires; i++) out << -state[i] << " 0\n";
  }

  for (auto [a, b, c] : gates) {
    int old_a = state[a];
    int new_a = next_var++;
    write_update_clauses(out, old_a, state[b], state[c], new_a);
    state[a] = new_a;
  }

  for (int i = 0; i < output_bits; i++) {
    int lit = state[output_start + i];
    out << (target[i] ? lit : -lit) << " 0\n";
  }

  std::cerr << "gates " << gates.size() << "\n";
  std::cerr << "total_wires " << total_wires << "\n";
  std::cerr << "original_wires " << original_wires << "\n";
  std::cerr << "fixed_input_start " << fixed_start << "\n";
  std::cerr << "fixed_input_bits " << fixed_bits << "\n";
  std::cerr << "output_start " << output_start << "\n";
  std::cerr << "output_bits " << output_bits << "\n";
  std::cerr << "aux_zero " << (aux_zero ? "yes" : "no") << "\n";
  std::cerr << "vars " << (next_var - 1) << "\n";
  std::cerr << "clauses " << clause_count << "\n";
  return 0;
}
