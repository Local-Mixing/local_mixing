import argparse

BASE = 83

# Convert string to circuit
def char_to_wire(c):
    if '0' <= c <= '9':
        return ord(c) - ord('0')
    if 'a' <= c <= 'z':
        return ord(c) - ord('a') + 10
    if 'A' <= c <= 'Z':
        return ord(c) - ord('A') + 36

    special = {
        '!':62,'@':63,'#':64,'$':65,'%':66,'^':67,'&':68,'*':69,
        '(':70,')':71,'-':72,'_':73,'=':74,'+':75,'[':76,']':77,
        '{':78,'}':79,'<':80,'>':81,'?':82
    }

    if c in special:
        return special[c]

    raise ValueError(f"Invalid wire char: {c}")

def parse_circuit(text):
    gates = []

    for part in text.strip().split(';'):
        if not part:
            continue

        i = 0
        wires = []

        while i < len(part):
            overflow = 0
            while i < len(part) and part[i] == '~':
                overflow += 1
                i += 1

            c = part[i]
            i += 1

            wire = char_to_wire(c) + overflow * BASE
            wires.append(wire)

        if len(wires) != 3:
            raise ValueError(f"Gate must have 3 wires: {part}")

        gates.append(wires)

    return gates

def draw_svg(gates, n_wires, outfile, min_gate=0, max_gate=None):
    if max_gate is None:
        max_gate = len(gates)

    gate_spacing = 10
    wire_spacing = 30
    radius = 20

    left_margin = 50 

    gates_to_draw = gates[min_gate:max_gate]

    width = len(gates_to_draw) * gate_spacing + left_margin + 50
    height = n_wires * wire_spacing + 50

    with open(outfile, "w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{width}" height="{height}" '
                f'viewBox="0 0 {width} {height}">\n')

        # draw wires and numbers
        for w in range(n_wires):
            y = w * wire_spacing + 25
            f.write(f'<line x1="{left_margin}" y1="{y}" x2="{width}" y2="{y}" '
                    f'stroke="black" stroke-width="1"/>\n')

            # right-align ones digit
            s = str(w)
            ones_x = left_margin - 5
            f.write(f'<text x="{ones_x - len(s)*12}" y="{y + 5}" font-size="30" '
                    f'font-family="monospace">{s}</text>\n')

        # draw gates
        for i, (a, b, c) in enumerate(gates_to_draw):
            x = i * gate_spacing + 40 + left_margin

            wires = [a, b, c]
            ys = [w * wire_spacing + 25 for w in wires if w < n_wires]
            if ys:
                f.write(
                    f'<line x1="{x}" y1="{min(ys)}" x2="{x}" y2="{max(ys)}" '
                    f'stroke="black" stroke-width="2"/>\n'
                )

            # ⊕ circ plus 
            if a < n_wires:
                y = a * wire_spacing + 25
                dark_red = "#CF0000"
                f.write(f'<circle cx="{x}" cy="{y}" r="{radius}" '
                        f'stroke="{dark_red}" fill="none" stroke-width="5"/>\n')
                f.write(f'<line x1="{x-radius}" y1="{y}" x2="{x+radius}" y2="{y}" '
                        f'stroke="{dark_red}" stroke-width="5"/>\n')
                f.write(f'<line x1="{x}" y1="{y-radius}" x2="{x}" y2="{y+radius}" '
                        f'stroke="{dark_red}" stroke-width="5"/>\n')

            # ○ white circle
            if b < n_wires:
                y = b * wire_spacing + 25
                f.write(f'<circle cx="{x}" cy="{y}" r="{radius - 4}" stroke="black" '
                        f'fill="white" stroke-width="2"/>\n')
                f.write(f'<line x1="{x-radius+4}" y1="{y}" x2="{x+radius-4}" y2="{y}" '
                        f'stroke="black" stroke-width="2"/>\n')
            # ● black circle
            if c < n_wires:
                y = c * wire_spacing + 25
                f.write(f'<circle cx="{x}" cy="{y}" r="{radius-4}" fill="black"/>\n')

        f.write('</svg>\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to the circuit file")
    parser.add_argument("-n", type=int, required=True, help="Number of wires")
    parser.add_argument("-p", "--path", default="circuit.svg", help="Output SVG file")
    parser.add_argument("--min", type=int, default=0, help="Index of first gate to draw")
    parser.add_argument("--max", type=int, help="Index of last gate to draw (exclusive)")

    args = parser.parse_args()

    with open(args.file) as f:
        text = f.read()

    gates = parse_circuit(text)

    draw_svg(gates, args.n, args.path, min_gate=args.min, max_gate=args.max)
    print(f"Saved to {args.path}")

if __name__ == "__main__":
    main()