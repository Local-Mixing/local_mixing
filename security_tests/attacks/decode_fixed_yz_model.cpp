#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
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
  bool comment = false;
  bool line_start = true;
  char ch;
  while (in.get(ch)) {
    if (comment) {
      if (ch == '\n' || ch == '\r') {
        comment = false;
        line_start = true;
      }
      continue;
    }
    if (line_start && ch == '#') {
      comment = true;
    } else if (ch == ';') {
      if (!w.empty()) {
        if (w.size() != 3) throw std::runtime_error("bad gate");
        gates.push_back({w[0], w[1], w[2]});
        w.clear();
      }
      overflow = 0;
    } else if (ch == '~') {
      overflow++;
      line_start = false;
    } else if (ch == '\n' || ch == '\r' || ch == '\t' || ch == ' ') {
      if (ch == '\n' || ch == '\r') line_start = true;
      continue;
    } else {
      int b = val(ch);
      if (b < 0) throw std::runtime_error("bad char in circuit");
      w.push_back(b + 83 * overflow);
      overflow = 0;
      line_start = false;
    }
  }
  if (!w.empty()) throw std::runtime_error("unterminated gate");
  return gates;
}

static unsigned char hex_nibble(char c) {
  if ('0' <= c && c <= '9') return c - '0';
  if ('a' <= c && c <= 'f') return 10 + c - 'a';
  if ('A' <= c && c <= 'F') return 10 + c - 'A';
  throw std::runtime_error("bad hex digit");
}

static std::array<int, 128> parse_hex128(std::string s) {
  if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0) s = s.substr(2);
  if (s.size() > 32) throw std::runtime_error("value too wide");
  while (s.size() < 32) s = "0" + s;
  std::array<int, 128> bits{};
  for (int hex_pos = 0; hex_pos < 32; hex_pos++) {
    unsigned char nib = hex_nibble(s[31 - hex_pos]);
    for (int j = 0; j < 4; j++) bits[4 * hex_pos + j] = (nib >> j) & 1;
  }
  return bits;
}

static std::string hex128(const std::array<int, 384> &bits, int offset) {
  std::ostringstream out;
  out << "0x" << std::hex << std::setfill('0');
  for (int chunk = 3; chunk >= 0; chunk--) {
    uint32_t v = 0;
    for (int i = 0; i < 32; i++) {
      if (bits[offset + chunk * 32 + i]) v |= 1u << i;
    }
    out << std::setw(8) << v;
  }
  return out.str();
}

static void eval(std::array<int, 384> &bits, const std::vector<Gate> &gates) {
  for (auto [a, b, c] : gates) bits[a] ^= bits[b] | !bits[c];
}

int main(int argc, char **argv) {
  if (argc != 6) {
    std::cerr << "usage: decode_fixed_yz_model circuit solver.out y128 z128 target128\n";
    return 2;
  }

  auto gates = parse_circuit(argv[1]);
  auto y_value = parse_hex128(argv[3]);
  auto z_value = parse_hex128(argv[4]);
  auto target = parse_hex128(argv[5]);
  std::ifstream in(argv[2]);
  if (!in) throw std::runtime_error("failed to open solver output");

  std::array<int, 384> input{};
  for (int i = 0; i < 128; i++) {
    input[128 + i] = y_value[i];
    input[256 + i] = z_value[i];
  }

  std::string line;
  bool sat = false;
  while (std::getline(in, line)) {
    if (line.rfind("s SATISFIABLE", 0) == 0) sat = true;
    if (line.empty() || line[0] != 'v') continue;
    std::istringstream ss(line.substr(1));
    long long lit = 0;
    while (ss >> lit) {
      if (lit == 0) break;
      long long v = lit < 0 ? -lit : lit;
      if (v >= 1 && v <= 128) input[static_cast<size_t>(v - 1)] = lit > 0;
    }
  }

  if (!sat) {
    std::cout << "sat no\n";
    return 1;
  }

  auto output = input;
  eval(output, gates);
  bool middle_ok = true;
  for (int i = 0; i < 128; i++) {
    if (output[128 + i] != target[i]) middle_ok = false;
  }

  std::cout << "sat yes\n";
  std::cout << "X " << hex128(input, 0) << "\n";
  std::cout << "Y " << hex128(input, 128) << "\n";
  std::cout << "Z " << hex128(input, 256) << "\n";
  std::cout << "out_left " << hex128(output, 0) << "\n";
  std::cout << "out_middle " << hex128(output, 128) << "\n";
  std::cout << "out_right " << hex128(output, 256) << "\n";
  std::cout << "target_middle " << argv[5] << "\n";
  std::cout << "middle_verified " << (middle_ok ? "yes" : "no") << "\n";
  return middle_ok ? 0 : 1;
}
