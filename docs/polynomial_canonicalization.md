# Polynomial Canonicalization

Suppose two circuits have different gates, but compute the same function. How
do we recognize this without evaluating every possible input? This is the
problem our database lookups need to solve. We also want circuits that differ
only by the names of their wires to share an entry.

We do this in two parts. First, we write the function of each output wire as a
polynomial in the input wires. Then we choose a canonical labeling of those
wires. The resulting polynomial list is what we hash for the
[frozen database](FROZEN_DATABASE.md).

## Writing a circuit as polynomials

Every wire starts with its own input variable, so initially $P_i=x_i$. We then
apply the gates in order, updating the target polynomial at each gate. The
arithmetic is over $GF(2)$, with Boolean variables:

$$x+x=0,\qquad x^2=x,\qquad \neg x=1+x.$$

Thus, a polynomial is just the set of monomials that occur an odd number of
times. For instance,

$$x_0+x_0+x_1x_1+x_1x_2=x_1+x_1x_2.$$

This is the algebraic normal form, or ANF. In
[`polynomial.rs`](../src/canonicalization/polynomial.rs), a monomial is a
`u64` bitmask: bit $i$ says whether $x_i$ is present. Mask `0` means the
constant $1$, and an empty polynomial means $0$. Multiplying monomials takes
the union of their variables, which is simply bitwise OR. Polynomials are
sorted vectors of masks; addition merges them and cancels shared terms.

For our r57 gate, written `g57` in the code, the update is

$$P_a\leftarrow P_a+(1+P_b)P_c+1.$$

This matches the executor's $a\mathrel{\oplus}=b\vee\neg c$ convention.
[`window.rs`](../src/canonicalization/window.rs) composes these three-wire
gates. The current mixing tape also contains wider gates with positive and
negative controls, so [`xgate.rs`](../src/canonicalization/xgate.rs) uses

$$P_t\leftarrow P_t+\prod_{(w,s)\in\mathrm{controls}}L_{w,s}
+\mathrm{comp},\qquad
L_{w,+}=P_w,\quad L_{w,-}=1+P_w.$$

Both routes produce the same ANF for the same function. We can therefore look
up a heterogeneous window and replace it with a stored g57 circuit.

The polynomial list retains all outputs. We do not subtract the original
input from each wire: an unchanged wire has polynomial $x_i$, not $0$.

## Choosing the wire labels

Normalizing each polynomial already removes differences in how a function was
computed. There is still the question of wire names. A gate on wires
`[12, 40, 97]` should be recognizable as a relabeling of the same gate on
`[0, 1, 2]`.

We first collect the wires touched by the selected window and map them to
`0..n`. These are local variables for the inputs at the start of the window;
we do not compose the entire circuit prefix. We then run
`canonicalize_polys_4` in
[`canonicalize.rs`](../src/canonicalization/canonicalize.rs):

1. **Build degree classes.** For each output polynomial, count its monomials
   at each degree, highest degree first. Group outputs with equal profiles.
   For each group, sum its polynomials with coefficients in $\mathbb{N}$.
   Here repeated monomials are counted, rather than cancelled, because we
   want to know how often they occur across the group.
2. **Refine the wire ranking.** Begin with all wires tied. Scan these class
   polynomials, considering higher-degree monomials first, then the current
   ranks of their variables, then their coefficients. Within each tied
   monomial level, count how often each wire appears. Different frequencies
   split a tied wire group. After a split, restart the scan because the new
   ranks may distinguish monomials that were previously tied.
3. **Try the two tiebreaks.** First compare the output polynomials of tied
   wires using the current variable ranks. If this does not split anything,
   build class polynomials from the current rank groups and repeat the
   frequency refinement. Any split returns us to the first refinement.
4. **Use Rule L if ties remain.** Take the first tied group, temporarily
   promote each candidate wire, and continue recursively. Keep the branch
   with the lexicographically smallest completed polynomial representation.
5. **Apply the resulting order.** Reorder the output polynomials and rename
   every variable using that same wire order. Finally, trim trailing
   independent identity wires.

We note that the degree classes in step 1 do not give the wires their initial
ranks. Every wire starts tied; the classes only determine which polynomial
information the refinement examines first.

Rule L is the search part of the algorithm. If two branches reach the same
form, their labelings reveal a symmetry of the polynomial system. We remember
this symmetry and skip later candidates already covered by it, provided it
preserves the current rank groups. This is the same general idea as
refinement and search in graph canonicalization, but the production path
works directly on the polynomials. The separate experimental graph
canonicalizer is not the database-key algorithm.

The same order must be applied to both sides of a wire: its input variable
and its output polynomial. We are relabeling a circuit's wires, not choosing
unrelated input and output permutations. For instance, an identity function
and a wire swap remain different functions.

## Identity wires and returning to the circuit

Suppose the canonical list ends with $P_i=x_i$, and $x_i$ occurs in no other
output polynomial. That wire does nothing and affects nothing, so we can
remove it from the key. We continue backwards until the first wire that fails
either condition. An unchanged control wire that still appears in another
output must stay.

This lets circuits with extra, independent identity wires share a key. We do
not need a separate table for every total wire count. The returned permutation
still covers the full window, even when its polynomial list was trimmed.

For a lookup, `CanonicalXPolys` keeps the two maps needed to return a stored
replacement to physical wires:

```text
canonical wire c
    -> order.data[c]              (wire in the dense window)
    -> used_wires[order.data[c]]  (wire in the original circuit)
```

[`friend_to_xgates`](../src/stages/db_mixing/replacement.rs) applies this map.
If a stored circuit uses additional identity workspace, it assigns available
wires for that workspace as well. The representation identifies the function;
the maps tell us where to put it back.

## From a canonical form to a database key

[`polys_repr_blob`](../src/canonicalization/keys.rs) writes the sorted
monomial masks of each polynomial as little-endian `u64`s, followed by a
`u64::MAX` separator. The lookup key is

```rust
xxh3_128(&polys_repr_blob(&canonical_polys)).to_le_bytes()
```

This is a 16-byte key. The polynomial representation is exact; its hash is
the compact database index. When we need an exact equivalence check on a
replacement, `polys_equivalent` composes both circuits over their combined
physical support and compares their ANFs directly.

Reversal is handled separately from wire canonicalization. Since each valid
gate is an involution, reversing the gate sequence computes the inverse
function. The regular g57 store chooses the smaller of the forward and
reverse canonical forms. Normal regular lookup computes both directions and
probes the smaller one; a reverse-direction replacement is reversed again
before insertion. Compatibility modes can probe both keys. Curated lookup
uses the forward form, with regular fallback following the regular rule.

We do not automatically identify a function with arbitrary input negations,
output negations, affine changes of basis, or internal shuffles. The legacy
`canonicalize_polys_single_neg` helper explicitly substitutes
$x_w\leftarrow x_w+1$ for specified pending input NOTs. That is a requested
change to the function being keyed, rather than a search over every possible
negation. Likewise, the shuffles in
[gadgetization](GADGETIZATION.md) are actual circuit operations, not labels
that canonicalization can simply discard.

## Keeping local lookups affordable

ANF can still grow exponentially. A local window is useful because it often
has a small polynomial representation even when the whole circuit does not.
There are several limits on this work:

- A lookup window may touch at most 64 distinct wires, because monomials are
  `u64` masks. Physical wire numbers can be larger; only the dense window
  needs to fit.
- XGate composition has separate budgets for raw multiplication terms,
  reduced terms per polynomial, and reduced terms across all live wires.
  The g57 composition path has its own monomial cap.
- The DB move can reject a window by its touched-wire span. After composing
  the ANF, it can also reject a direction whose exact degree exceeds the
  configured database degree limit, before doing canonical wire ordering.
- Rule L has an optional candidate budget shared across the entire recursive
  call. It charges the candidates at each search node before symmetry
  pruning. Exceeding the budget abandons the canonicalization.

A budget failure means we skip this lookup; it does not supply a partial key
or establish that no equivalent circuit exists. In particular, intermediate
polynomials can grow before later gates cancel them, and a touched-wire span
can be larger than the function's final support.

The [GSS driver](../src/gss/runner.rs) currently pins the Rule L budget to
512 and the legacy g57 monomial cap to 200,000. XGate composition uses its
separate term budgets. Explicit library APIs accept
`CanonicalizationOptions`, `G57CanonicalizationOptions`, or
`XGateCanonicalizationOptions`; these calls use their supplied options and
bypass the legacy process-wide canonicalization caches.

In the GSS compatibility path, repeated dense windows can reuse cached
canonicalizations. The g57 cache stores successful forms, wire orders, and
hashes. The XGate cache stores successful forms and failures, with its
composition budgets and degree limit included in the cache key. GSS assigns
these caches approximately 256 MiB and 1024 MiB respectively. On exceeding
their configured capacity, they clear the map. These are separate from the
frozen database's lookup cache, which remembers database hits and misses.

For the surrounding flow, see the [six GSS steps](GSS_PIPELINE.md). For the
longer mathematical discussion and examples, see
[Local Mixing Documentation](Local_Mixing_Documentation.pdf).
