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

static uint64_t parse_u64(const std::string &s) {
  uint64_t out = 0;
  std::stringstream ss(s);
  if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0)
    ss >> std::hex >> out;
  else
    ss >> out;
  if (!ss) throw std::runtime_error("bad target");
  return out;
}

static std::array<int, 128> apply_forward(std::array<int, 128> bits,
                                          const std::vector<Gate> &gates) {
  for (auto [a, b, c] : gates) bits[a] ^= bits[b] | !bits[c];
  return bits;
}

static uint64_t low64(const std::array<int, 128> &bits) {
  uint64_t out = 0;
  for (int i = 0; i < 64; i++) {
    if (bits[i]) out |= 1ULL << i;
  }
  return out;
}

static void print_hex128(const char *label, const std::array<int, 128> &bits) {
  std::cout << label << " 0x" << std::hex << std::setfill('0');
  for (int chunk = 3; chunk >= 0; chunk--) {
    uint32_t v = 0;
    for (int i = 0; i < 32; i++) {
      if (bits[chunk * 32 + i]) v |= 1u << i;
    }
    std::cout << std::setw(8) << v;
  }
  std::cout << std::dec << "\n";
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: decode_forward_lowtarget_model F.txt kissat.out target64\n";
    return 2;
  }

  auto gates = parse(argv[1]);
  uint64_t target = parse_u64(argv[3]);
  std::vector<int> assign(129, -1);
  std::ifstream in(argv[2]);
  if (!in) throw std::runtime_error("failed to open model");
  std::string tok;
  while (in >> tok) {
    if (tok == "v") continue;
    if (tok == "s" || tok == "c") {
      std::string rest;
      std::getline(in, rest);
      continue;
    }
    int lit = 0;
    try {
      lit = std::stoi(tok);
    } catch (...) {
      continue;
    }
    if (lit == 0) continue;
    int var = lit > 0 ? lit : -lit;
    if (var >= 1 && var <= 128) assign[var] = lit > 0 ? 1 : 0;
  }

  std::array<int, 128> xbits{};
  for (int i = 0; i < 128; i++) {
    if (assign[i + 1] < 0) throw std::runtime_error("missing input assignment");
    xbits[i] = assign[i + 1];
  }
  auto ybits = apply_forward(xbits, gates);
  int bad_leading = 0;
  for (int i = 78; i < 128; i++) bad_leading += xbits[i] != 0;
  print_hex128("x", xbits);
  print_hex128("Fx", ybits);
  std::cout << "low64_Fx 0x" << std::hex << std::setw(16) << std::setfill('0')
            << low64(ybits) << std::dec << "\n";
  std::cout << "bad_leading_input_bits_127_78 " << bad_leading << "\n";
  std::cout << "verified " << ((low64(ybits) == target && bad_leading == 0) ? "yes" : "no")
            << "\n";
  return 0;
}
