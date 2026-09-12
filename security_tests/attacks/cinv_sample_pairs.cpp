#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Gate = std::array<int, 3>;
using State = std::array<uint64_t, 6>;

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
  std::vector<int> wires;
  int overflow = 0;
  char ch;
  while (in.get(ch)) {
    if (ch == ';') {
      if (!wires.empty()) {
        if (wires.size() != 3) throw std::runtime_error("bad gate");
        gates.push_back({wires[0], wires[1], wires[2]});
        wires.clear();
      }
      overflow = 0;
    } else if (ch == '~') {
      overflow++;
    } else if (ch == '\n' || ch == '\r' || ch == '\t' || ch == ' ') {
      continue;
    } else {
      int b = val(ch);
      if (b < 0) throw std::runtime_error("bad char");
      wires.push_back(b + 83 * overflow);
      overflow = 0;
    }
  }
  if (!wires.empty()) throw std::runtime_error("unterminated gate");
  return gates;
}

static unsigned char hex_nibble(char c) {
  if ('0' <= c && c <= '9') return c - '0';
  if ('a' <= c && c <= 'f') return 10 + c - 'a';
  if ('A' <= c && c <= 'F') return 10 + c - 'A';
  throw std::runtime_error("bad hex digit");
}

static std::array<uint64_t, 2> parse_hex128(std::string s) {
  if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0) s = s.substr(2);
  if (s.size() > 32) throw std::runtime_error("value too wide");
  while (s.size() < 32) s = "0" + s;
  std::array<uint64_t, 2> out{0, 0};
  for (int hex_pos = 0; hex_pos < 32; hex_pos++) {
    unsigned char nib = hex_nibble(s[31 - hex_pos]);
    out[(4 * hex_pos) / 64] |= uint64_t(nib) << ((4 * hex_pos) % 64);
  }
  return out;
}

static std::string hex128(uint64_t lo, uint64_t hi) {
  std::ostringstream out;
  out << "0x" << std::hex << std::setfill('0') << std::setw(16) << hi
      << std::setw(16) << lo;
  return out.str();
}

static inline bool bit(const State &s, int idx) {
  return (s[idx / 64] >> (idx % 64)) & 1ull;
}

static inline void flip(State &s, int idx) {
  s[idx / 64] ^= 1ull << (idx % 64);
}

static void eval(State &s, const std::vector<Gate> &gates) {
  for (auto [a, b, c] : gates) {
    if (bit(s, b) || !bit(s, c)) flip(s, a);
  }
}

int main(int argc, char **argv) {
  if (argc != 7) {
    std::cerr << "usage: cinv_sample_pairs circuit y128 z128 count seed out.tsv\n";
    return 2;
  }

  auto gates = parse(argv[1]);
  auto y = parse_hex128(argv[2]);
  auto z = parse_hex128(argv[3]);
  uint64_t count = std::strtoull(argv[4], nullptr, 0);
  uint64_t seed = std::strtoull(argv[5], nullptr, 0);
  std::ofstream out(argv[6]);
  if (!out) throw std::runtime_error("failed to create output");

  std::mt19937_64 rng(seed);
  for (uint64_t i = 0; i < count; i++) {
    State s{rng(), rng(), y[0], y[1], z[0], z[1]};
    const uint64_t x0 = s[0], x1 = s[1];
    eval(s, gates);
    const uint64_t u0 = s[2] ^ y[0];
    const uint64_t u1 = s[3] ^ y[1];
    out << hex128(u0, u1) << '\t' << hex128(x0, x1) << '\n';
  }

  std::cerr << "gates " << gates.size() << "\n";
  std::cerr << "samples " << count << "\n";
  return 0;
}
