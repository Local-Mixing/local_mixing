#include <charconv>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

struct Header {
  int wires = 0;
  std::int64_t gates = 0;
};

struct Gate {
  int target = 0;
  int complemented = 0;
  std::vector<int> controls;
  std::vector<int> polarities;
};

static int hex_value(char c) {
  if ('0' <= c && c <= '9') return c - '0';
  if ('a' <= c && c <= 'f') return 10 + c - 'a';
  if ('A' <= c && c <= 'F') return 10 + c - 'A';
  return -1;
}

static std::vector<int> parse_hex_bits(std::string value, int bits) {
  if (value.rfind("0x", 0) == 0 || value.rfind("0X", 0) == 0) {
    value.erase(0, 2);
  }
  if (value.empty()) throw std::runtime_error("empty target hex value");
  std::vector<int> result(bits, 0);
  for (int bit = 0; bit < bits; ++bit) {
    const int position = static_cast<int>(value.size()) - 1 - bit / 4;
    if (position < 0) break;
    const int nibble = hex_value(value[static_cast<std::size_t>(position)]);
    if (nibble < 0) throw std::runtime_error("invalid target hex digit");
    result[bit] = (nibble >> (bit % 4)) & 1;
  }
  for (int bit = bits; bit < static_cast<int>(value.size()) * 4; ++bit) {
    const int position = static_cast<int>(value.size()) - 1 - bit / 4;
    const int nibble = hex_value(value[static_cast<std::size_t>(position)]);
    if (nibble < 0) throw std::runtime_error("invalid target hex digit");
    if (((nibble >> (bit % 4)) & 1) != 0) {
      throw std::runtime_error("target has nonzero bits above requested width");
    }
  }
  return result;
}

static std::vector<int> parse_integer_line(const std::string &line) {
  std::vector<int> fields;
  const char *cursor = line.data();
  const char *end = cursor + line.size();
  while (cursor != end) {
    while (cursor != end && (*cursor == ' ' || *cursor == '\t' || *cursor == '\r')) {
      ++cursor;
    }
    if (cursor == end) break;
    int value = 0;
    const auto parsed = std::from_chars(cursor, end, value);
    if (parsed.ec != std::errc() || parsed.ptr == cursor) {
      throw std::runtime_error("invalid integer in gate line");
    }
    fields.push_back(value);
    cursor = parsed.ptr;
  }
  return fields;
}

static Header read_header(std::ifstream &input) {
  std::string marker;
  Header header;
  if (!(input >> marker >> header.wires >> header.gates) || marker != "mpmct1") {
    throw std::runtime_error("invalid mpmct1 header");
  }
  if (header.wires <= 0 || header.gates < 0 ||
      header.gates > std::numeric_limits<int>::max() - header.wires) {
    throw std::runtime_error("unsupported mpmct1 dimensions");
  }
  input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  return header;
}

static Gate parse_gate(const std::string &line, int wires) {
  const std::vector<int> fields = parse_integer_line(line);
  if (fields.size() < 3) throw std::runtime_error("short mpmct1 gate");
  Gate gate;
  gate.target = fields[0];
  gate.complemented = fields[1];
  const int width = fields[2];
  if (gate.target < 0 || gate.target >= wires ||
      (gate.complemented != 0 && gate.complemented != 1) || width < 0 ||
      fields.size() != static_cast<std::size_t>(3 + 2 * width)) {
    throw std::runtime_error("malformed mpmct1 gate");
  }
  gate.controls.reserve(static_cast<std::size_t>(width));
  gate.polarities.reserve(static_cast<std::size_t>(width));
  std::vector<unsigned char> seen(static_cast<std::size_t>(wires), 0);
  for (int index = 0; index < width; ++index) {
    const int control = fields[static_cast<std::size_t>(3 + 2 * index)];
    const int polarity = fields[static_cast<std::size_t>(4 + 2 * index)];
    if (control < 0 || control >= wires || control == gate.target ||
        (polarity != 0 && polarity != 1) || seen[static_cast<std::size_t>(control)]) {
      throw std::runtime_error("invalid or duplicate mpmct1 control");
    }
    seen[static_cast<std::size_t>(control)] = 1;
    gate.controls.push_back(control);
    gate.polarities.push_back(polarity);
  }
  return gate;
}

template <class Callback>
static Header for_each_gate(const std::string &path, Callback callback) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("failed to open circuit: " + path);
  const Header header = read_header(input);
  std::string line;
  std::int64_t count = 0;
  while (std::getline(input, line)) {
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    Gate gate = parse_gate(line, header.wires);
    callback(gate, count);
    ++count;
  }
  if (count != header.gates) {
    throw std::runtime_error("gate count does not match mpmct1 header");
  }
  return header;
}

static int parse_positive_int(const char *text, const char *name) {
  int value = 0;
  const std::string_view view(text);
  const auto result = std::from_chars(view.data(), view.data() + view.size(), value);
  if (result.ec != std::errc() || result.ptr != view.data() + view.size() || value <= 0) {
    throw std::runtime_error(std::string("invalid ") + name);
  }
  return value;
}

int main(int argc, char **argv) {
  try {
    if (argc != 5 && argc != 6) {
      std::cerr << "usage: mpmct1_zero_slice_to_cnf FINAL.mpmct1 OUT.cnf N "
                   "TARGET_HEX [--analyze-only]\n";
      return 2;
    }
    const std::string circuit_path = argv[1];
    const std::string output_path = argv[2];
    const int n = parse_positive_int(argv[3], "logical width n");
    const std::vector<int> target = parse_hex_bits(argv[4], n);
    const bool analyze_only = argc == 6 && std::string(argv[5]) == "--analyze-only";
    if (argc == 6 && !analyze_only) throw std::runtime_error("unknown optional argument");

    std::int64_t total_controls = 0;
    const Header first = for_each_gate(
        circuit_path, [&](const Gate &gate, std::int64_t index) {
          total_controls += static_cast<std::int64_t>(gate.controls.size());
          if ((index + 1) % 1000000 == 0) {
            std::cerr << "[cnf] validated " << (index + 1) << " gates\n";
          }
        });
    if (first.wires != 4 * n) {
      throw std::runtime_error("expected final circuit to have exactly 4n wires");
    }

    const std::int64_t variables = first.wires + first.gates;
    const std::int64_t clauses =
        2 * total_controls + 2 * first.gates + (first.wires - n) + n;
    std::cerr << "[cnf] analysis wires=" << first.wires << " gates=" << first.gates
              << " controls=" << total_controls << " vars=" << variables
              << " clauses=" << clauses << "\n";
    if (analyze_only) return 0;

    std::ofstream output(output_path);
    if (!output) throw std::runtime_error("failed to create CNF: " + output_path);
    std::vector<char> output_buffer(8 * 1024 * 1024);
    output.rdbuf()->pubsetbuf(output_buffer.data(),
                              static_cast<std::streamsize>(output_buffer.size()));
    output << "p cnf " << variables << ' ' << clauses << "\n";
    output << "c zero-slice preimage: inputs " << n << ".." << (first.wires - 1)
           << " fixed to zero; outputs " << n << ".." << (2 * n - 1)
           << " fixed to the public target\n";

    std::vector<int> state(static_cast<std::size_t>(first.wires));
    for (int wire = 0; wire < first.wires; ++wire) state[static_cast<std::size_t>(wire)] = wire + 1;
    for (int wire = n; wire < first.wires; ++wire) output << -(wire + 1) << " 0\n";

    int next_variable = first.wires + 1;
    const Header second = for_each_gate(
        circuit_path, [&](const Gate &gate, std::int64_t index) {
          const int old_target = state[static_cast<std::size_t>(gate.target)];
          const int new_target = next_variable++;
          const int result_literal = gate.complemented ? -new_target : new_target;
          std::vector<int> literals;
          literals.reserve(gate.controls.size());
          for (std::size_t i = 0; i < gate.controls.size(); ++i) {
            const int value = state[static_cast<std::size_t>(gate.controls[i])];
            literals.push_back(gate.polarities[i] ? value : -value);
          }

          // result_literal = old_target XOR AND(literals).  This direct
          // encoding uses one state variable per circuit gate and 2k+2
          // clauses for a width-k gate, including the k=0 case.
          for (const int literal : literals) {
            output << literal << ' ' << -old_target << ' ' << result_literal << " 0\n";
            output << literal << ' ' << old_target << ' ' << -result_literal << " 0\n";
          }
          for (const int literal : literals) output << -literal << ' ';
          output << old_target << ' ' << result_literal << " 0\n";
          for (const int literal : literals) output << -literal << ' ';
          output << -old_target << ' ' << -result_literal << " 0\n";

          state[static_cast<std::size_t>(gate.target)] = new_target;
          if ((index + 1) % 500000 == 0) {
            std::cerr << "[cnf] emitted " << (index + 1) << " gates\n";
          }
        });
    if (second.wires != first.wires || second.gates != first.gates ||
        next_variable - 1 != variables) {
      throw std::runtime_error("circuit changed between CNF passes");
    }
    for (int bit = 0; bit < n; ++bit) {
      const int variable = state[static_cast<std::size_t>(n + bit)];
      output << (target[static_cast<std::size_t>(bit)] ? variable : -variable) << " 0\n";
    }
    output.close();
    if (!output) throw std::runtime_error("failed while writing CNF");

    std::cerr << "[cnf] wires=" << first.wires << " gates=" << first.gates
              << " controls=" << total_controls << " vars=" << variables
              << " clauses=" << clauses << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}
