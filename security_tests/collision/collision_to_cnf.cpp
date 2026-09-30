// Encode a collision for H(x) = C(0^{pad} || x)_{out} as DIMACS CNF.
//
// Two independent evaluations of the same mpmct1 tape share no variables.
// Free inputs occupy wires [0, in_bits) of each copy; wires [in_bits, width)
// are fixed to zero. The low out_bits of each final state are forced equal,
// and the two free inputs are forced to differ in at least one bit.
//
// usage: collision_to_cnf CIRCUIT.mpmct1 OUT.cnf
//            [--in-bits 64] [--pad 32] [--out-bits 32] [--analyze-only]

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

static int parse_nonnegative_int(const char *text, const char *name) {
  int value = 0;
  const std::string_view view(text);
  const auto result = std::from_chars(view.data(), view.data() + view.size(), value);
  if (result.ec != std::errc() || result.ptr != view.data() + view.size() || value < 0) {
    throw std::runtime_error(std::string("invalid ") + name);
  }
  return value;
}

// Emit one evaluation copy. `var_base` is the first DIMACS variable for this
// copy's initial wires (1-based). Returns the next free variable index and
// writes final wire→variable map into `final_state`.
static int emit_copy(std::ostream &output, const std::string &circuit_path, int wires,
                     int in_bits, int var_base, int next_variable,
                     std::vector<int> &final_state) {
  std::vector<int> state(static_cast<std::size_t>(wires));
  for (int wire = 0; wire < wires; ++wire) {
    state[static_cast<std::size_t>(wire)] = var_base + wire;
  }
  for (int wire = in_bits; wire < wires; ++wire) {
    output << -(var_base + wire) << " 0\n";
  }

  for_each_gate(circuit_path, [&](const Gate &gate, std::int64_t) {
    const int old_target = state[static_cast<std::size_t>(gate.target)];
    const int new_target = next_variable++;
    const int result_literal = gate.complemented ? -new_target : new_target;
    std::vector<int> literals;
    literals.reserve(gate.controls.size());
    for (std::size_t i = 0; i < gate.controls.size(); ++i) {
      const int value = state[static_cast<std::size_t>(gate.controls[i])];
      literals.push_back(gate.polarities[i] ? value : -value);
    }
    for (const int literal : literals) {
      output << literal << ' ' << -old_target << ' ' << result_literal << " 0\n";
      output << literal << ' ' << old_target << ' ' << -result_literal << " 0\n";
    }
    for (const int literal : literals) output << -literal << ' ';
    output << old_target << ' ' << result_literal << " 0\n";
    for (const int literal : literals) output << -literal << ' ';
    output << -old_target << ' ' << -result_literal << " 0\n";
    state[static_cast<std::size_t>(gate.target)] = new_target;
  });
  final_state = state;
  return next_variable;
}

int main(int argc, char **argv) {
  try {
    if (argc < 3 || (argc == 2 && std::string(argv[1]) == "--help")) {
      const bool help = argc == 2 && std::string(argv[1]) == "--help";
      auto &stream = help ? std::cout : std::cerr;
      stream << "usage: collision_to_cnf CIRCUIT.mpmct1 OUT.cnf "
                "[--in-bits 64] [--pad 32] [--out-bits 32] [--analyze-only]\n"
                "Find x1 != x2 on wires [0,in_bits) with pad high wires zero "
                "such that the low out_bits of C agree.\n";
      return help ? 0 : 2;
    }
    const std::string circuit_path = argv[1];
    const std::string output_path = argv[2];
    if (circuit_path == output_path) {
      throw std::runtime_error("CNF output must differ from circuit input");
    }
    int in_bits = 64;
    int pad = 32;
    int out_bits = 32;
    bool analyze_only = false;
    for (int index = 3; index < argc; ++index) {
      const std::string option = argv[index];
      if (option == "--analyze-only") {
        analyze_only = true;
      } else if (option == "--in-bits" && index + 1 < argc) {
        in_bits = parse_nonnegative_int(argv[++index], "in-bits");
      } else if (option == "--pad" && index + 1 < argc) {
        pad = parse_nonnegative_int(argv[++index], "pad");
      } else if (option == "--out-bits" && index + 1 < argc) {
        out_bits = parse_nonnegative_int(argv[++index], "out-bits");
      } else {
        throw std::runtime_error("unknown or incomplete option: " + option);
      }
    }
    if (in_bits <= 0 || out_bits <= 0) {
      throw std::runtime_error("in-bits and out-bits must be positive");
    }

    std::int64_t total_controls = 0;
    const Header first = for_each_gate(
        circuit_path, [&](const Gate &gate, std::int64_t) {
          total_controls += static_cast<std::int64_t>(gate.controls.size());
        });
    const int width = in_bits + pad;
    if (first.wires < width) {
      throw std::runtime_error("circuit narrower than in-bits + pad");
    }
    if (out_bits > first.wires) {
      throw std::runtime_error("out-bits exceeds circuit width");
    }

    // Per copy: wires initial vars + one var per gate; zero-pad units;
    // gate clauses 2k+2. Shared: out_bits equality (2 clauses each via xor
    // helper? we force equal with 2 clauses per bit: (a\/~b)/\(~a\/b));
    // differ: one big OR of (x1_i XOR x2_i) using in_bits aux vars.
    const std::int64_t vars_per_copy = first.wires + first.gates;
    const std::int64_t differ_aux = in_bits;  // d_i <=> x1_i XOR x2_i
    const std::int64_t variables = 2 * vars_per_copy + differ_aux;
    const std::int64_t gate_clauses_per_copy = 2 * total_controls + 2 * first.gates;
    const std::int64_t zero_clauses_per_copy = first.wires - in_bits;
    // equality: 2 clauses per out bit
    const std::int64_t equal_clauses = 2 * static_cast<std::int64_t>(out_bits);
    // d_i XOR defs: 4 clauses each; plus 1 clause OR_i d_i
    const std::int64_t differ_clauses = 4 * static_cast<std::int64_t>(in_bits) + 1;
    const std::int64_t clauses = 2 * (gate_clauses_per_copy + zero_clauses_per_copy) +
                                equal_clauses + differ_clauses;

    std::cerr << "[cnf] wires=" << first.wires << " gates=" << first.gates
              << " controls=" << total_controls << " vars=" << variables
              << " clauses=" << clauses << " in_bits=" << in_bits
              << " pad=" << pad << " out_bits=" << out_bits << "\n";
    if (analyze_only) return 0;

    std::ofstream output(output_path);
    if (!output) throw std::runtime_error("failed to create CNF: " + output_path);
    std::vector<char> output_buffer(8 * 1024 * 1024);
    output.rdbuf()->pubsetbuf(output_buffer.data(),
                              static_cast<std::streamsize>(output_buffer.size()));
    output << "p cnf " << variables << ' ' << clauses << "\n";
    output << "c collision: two evaluations; inputs 1.." << in_bits
           << " and " << (vars_per_copy + 1) << ".." << (vars_per_copy + in_bits)
           << "; low " << out_bits << " outputs equal; inputs differ\n";

    std::vector<int> final1;
    std::vector<int> final2;
    int next = static_cast<int>(vars_per_copy) + 1;
    // Copy 1 uses variables 1..wires for initials, then gate vars.
    // After copy 1, next should be wires+gates+1 = vars_per_copy+1.
    next = emit_copy(output, circuit_path, first.wires, in_bits, /*var_base=*/1,
                     /*next_variable=*/first.wires + 1, final1);
    if (next != static_cast<int>(vars_per_copy) + 1) {
      throw std::runtime_error("copy-1 variable allocation mismatch");
    }
    const int copy2_base = next;
    next = emit_copy(output, circuit_path, first.wires, in_bits, copy2_base,
                     copy2_base + first.wires, final2);
    if (next != static_cast<int>(2 * vars_per_copy) + 1) {
      throw std::runtime_error("copy-2 variable allocation mismatch");
    }

    for (int bit = 0; bit < out_bits; ++bit) {
      const int a = final1[static_cast<std::size_t>(bit)];
      const int b = final2[static_cast<std::size_t>(bit)];
      output << a << ' ' << -b << " 0\n";
      output << -a << ' ' << b << " 0\n";
    }

    // d_i = x1_i XOR x2_i; require OR d_i
    const int differ_base = next;  // first differ aux
    for (int bit = 0; bit < in_bits; ++bit) {
      const int x1 = 1 + bit;
      const int x2 = copy2_base + bit;
      const int d = differ_base + bit;
      // d <=> x1 XOR x2:
      // (~x1 \/  x2 \/ d) /\ ( x1 \/ ~x2 \/ d) /\ (~x1 \/ ~x2 \/ ~d) /\ (x1 \/ x2 \/ ~d)
      output << -x1 << ' ' << x2 << ' ' << d << " 0\n";
      output << x1 << ' ' << -x2 << ' ' << d << " 0\n";
      output << -x1 << ' ' << -x2 << ' ' << -d << " 0\n";
      output << x1 << ' ' << x2 << ' ' << -d << " 0\n";
    }
    for (int bit = 0; bit < in_bits; ++bit) {
      output << (differ_base + bit);
      if (bit + 1 < in_bits) output << ' ';
    }
    output << " 0\n";

    if (differ_base + in_bits - 1 != variables) {
      throw std::runtime_error("final variable count mismatch");
    }
    output.close();
    if (!output) throw std::runtime_error("failed while writing CNF");
    std::cerr << "[cnf] wrote " << output_path << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}
