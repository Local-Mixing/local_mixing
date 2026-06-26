# Local Mixing Documentation

## 1. Overview
<!-- TODO: Insert Links -->

Local Mixing is attempting to achieve indistinguishability obfuscation on general-purpose programs. We reduce this problem to general circuits, and then to reversible circuits, and finally even further to circuits with only the eca57 gate. This document serves to explain each of the components of our mixing. 

- High-level pipeline:
  1. Gadgetize
  2. Expansion 
  3. Hidden SAMFs
  4. Additional SAMFs
  5. Undo accumulated shuffle & negation state
  6. Compress
  7. Repeat 2-6

## 2. The ECA57 Gate

Local Mixing represents a reversible circuit as a sequence of ECA57 gates.
ECA57 is sufficient for this purpose because it is
[universal for reversible computation](https://arxiv.org/pdf/1809.08050).

A gate is written as

```text
[active, positive_control, negative_control]
```

or, more briefly, `[a, b, c]`. The three pins must refer to distinct wires. If
their current values are $x_a$, $x_b$, and $x_c$, the gate computes

$$
x_a \leftarrow x_a \oplus (x_b \lor \neg x_c),
$$

while leaving every other wire unchanged. Equivalently, the active wire is
flipped unless the positive control is `0` and the negative control is `1`.

| Positive control \(x_b\) | Negative control \(x_c\) | Flip \(x_a\)? |
|---:|---:|:---|
| `0` | `0` | Yes |
| `0` | `1` | No |
| `1` | `0` | Yes |
| `1` | `1` | Yes |

The gate is self-inverse because its controls are unchanged. Applying it twice
XORs the same value into the active wire twice:

$$
x_a \oplus f(x_b,x_c) \oplus f(x_b,x_c) = x_a.
$$
Thus, two identical adjacent gates cancel and may be removed from a circuit.

### 2.1 Collisions and Commutation

Two gates `[a, b, c]` and `[a', b', c']` **collide** when the active wire of
either gate is a control wire of the other:

$$
a \in \{b',c'\} \quad\text{or}\quad a' \in \{b,c\}.
$$

In a collision, one gate may modify a value that determines whether the other
gate flips its active wire. Their order can therefore change the circuit's
function. Gates that do not collide **commute** and may be reordered without
changing the function.

Sharing only control wires does not constitute a collision. Sharing the same
active wire also does not constitute a collision: both gates XOR a
control-dependent value into that wire, and XOR is commutative.

> **Invariant:** Reordering non-colliding ECA57 gates or removing adjacent
> identical gates preserves the circuit's functionality.

### 2.2 Gate and Text Encoding

Internally, each gate is stored as three wire indices:

| Position | Name | Meaning |
|---|---|---|
| `gate[0]` | Active wire | The wire that may be flipped |
| `gate[1]` | Positive control | Causes a flip when it is `1` |
| `gate[2]` | Negative control | Causes a flip when it is `0` |

`CircuitSeq::repr()` converts a circuit into its compact text format. Each wire
index is encoded as a token, the three tokens are concatenated in
active-positive-negative order, and `;` terminates the gate.

| Wire indices | Encoding |
|---|---|
| `0`-`9` | `0`-`9` |
| `10`-`35` | `a`-`z` |
| `36`-`61` | `A`-`Z` |
| `62`-`71` | `! @ # $ % ^ & * ( )` |
| `72`-`82` | `- _ = + [ ] { } < > ?` |
| `83` and above | Add one `~` prefix for each complete block of 83, then encode the remainder using the table above |

For example:

| Internal gate | Text representation |
|---|---|
| `[0, 1, 2]` | `012;` |
| `[10, 36, 62]` | `aA!;` |
| `[83, 84, 165]` | `~0~1~?;` |

Therefore, the circuit `[[0, 1, 2], [10, 36, 62]]` is represented as:

```text
012;aA!;
```

The inverse operation is `CircuitSeq::from_string()`. The implementation is in
[`CircuitSeq::repr()` and
`CircuitSeq::from_string()`](../src/circuit/circuit.rs).

## 3. Circuit Equivalence and Canonicalization

When making circuit replacements with equal functionality, some replacements are not *true* replacements. For instance, consider the circuit with 2 gates `[0,1,2][3,4,5]`. The two gates commute and so an equivalent circuit is `[3,4,5][0,1,2]`. When trying to make a replacement for `[0,1,2][3,4,5]`, `[3,4,5][0,1,2]` would not be a meaninful replacement, as it is simply an equivalent circuit up to commuting gates. Thus, rather than allowing arbitrary replacements of equal functionality, we will use a rainbow table that holds **structurally different** circuits based on our definition of canonicalization. We will have two definitions for canonicalization: the above discussed commuting-gate canonicalization, but also a wire-relabeling canonicalization via polynomials.  
### 3.1 Commuting-Gate Canonicalization

In order to take a canonical ordering of commuting gates, we simply choose the ordering that has the lexicographical minimum based on the wire indices of each gate. We say `[a,b,c]` is lexicographically smaller than `[a',b',c']` if $a < a'$, if $a = a'$ then if $b < b'$, and if $a = b'$ and $b = b'$ then if $c < c'$. 

### 3.2 Polynomial Canonicalization

The rainbow table should not store separate entries merely because two
circuits use different names for their wires. For example, a circuit using
wires `0` through `5` may have exactly the same structure and functionality as
one using wires `6` through `11`. A consistent renaming of wires should
therefore produce the same database key.

The direct solution would be to try all $n!$ permutations of the wires,
rewrite the circuit under each permutation, and choose the smallest result.
That becomes too expensive because canonicalization is performed for many
candidate subcircuits during expansion and compression. `canonicalize_polys_4`
instead constructs the ordering incrementally. It distinguishes variables by
their structural roles in the circuit's output polynomials, resorting to
backtracking only when genuine symmetries remain.

#### Algebraic representation

The functionality of an $n$-wire circuit is represented by a vector of
Boolean polynomials

$$
(P_0,P_1,\ldots,P_{n-1}),
$$

where $P_i(x_0,\ldots,x_{n-1})$ is the final value of output wire $i$.
These are algebraic-normal-form polynomials over $GF(2)$:

- addition is XOR;
- multiplication is AND;
- $x_i^2=x_i$, so every monomial is square-free;

For example,

$$
P_2 = 1+x_0+x_1x_3
$$

means that output wire `2` contains the XOR of `1`, input wire `0`, and the
product of input wires `1` and `3`.

In the implementation, a monomial is a `u64` bit mask. Bit $i$ is set when
the monomial contains $x_i$. Thus, $x_1x_3$ is represented by the mask
`0b1010`. A polynomial is a set of these masks. Set membership is sufficient
because equal terms cancel in pairs over $GF(2)$. Our actual mixed circuits will use more than $64$, but as we are only making *local* replacements, we will never actually need to canonicalize anything with $64$ wires.

The circuit is evaluated symbolically from left to right. Initially wire $i$
contains $x_i$. Each ECA57 gate updates the polynomial on its active wire,
using the current polynomials on its controls, while the other output
polynomials remain unchanged.

Before this process, the surrounding canonicalization code removes gaps in the
physical wire names. If a subcircuit uses wires `[3, 7, 11]`, those wires are
temporarily renamed `[0, 1, 2]`. The original list is retained so a database
replacement can later be mapped back to the physical circuit. Since our canonicalization attempts to make equal relabelings together, it doesn't depend on wire indices, which is what allows us to map our indices down to the minimal ones before canonicalization. 

#### What must be canonicalized

We will find a wire relabeling, or *permutation*, that puts a circuit in its canonical form. A wire permutation renames two things simultaneously:

1. the output position $P_i$; and
2. every occurrence of $x_i$ inside every polynomial.

Canonicalization must therefore order the output polynomials and rename all
their variables using the same permutation. Looking only at the output
polynomials individually is not enough: a variable's ranking also depends on
where and how it appears throughout the entire polynomial system.

The algorithm maintains a rank for every variable. Variables with the same
rank are still indistinguishable; variables with different ranks have already
been placed into different ordered classes. Initially all variables have rank
zero, meaning that every wire name is treated as interchangeable.

#### Step 1: degree profiles

For each output polynomial, the algorithm records how many monomials it has of
each degree, ordered from highest degree to lowest. For a circuit with $n$ wires, the maximum degree is $n-1$, which allows us to store the profile for each polynomial by starting at $n-1$. For example, the profile

```text
[0 degree-4 terms, 2 degree-3 terms, 1 degree-2 term,
 3 degree-1 terms, 1 constant term]
```

describes the shape of a polynomial without referring to any variable names.
Polynomials with equal profiles are placed in the same class $C_i$, and the
classes are ordered by profile, with larger high-degree profiles first.

For every class $C_i$, the algorithm forms a class polynomial with coefficients in $\mathbb{N}$, rather than in $GF(2)$. 

$$
P_{C_i}=\sum_{j\in C_i} P_j.
$$

Keeping track of coefficients lets the algorithm observe the shared structure of variables that are still otherwise indistinguishable.

#### Step 2: refine ranks using class polynomials

Each class polynomial is divided into ordered monomial levels. A monomial's
rank is determined by:

1. higher degree first
2. the sorted list of the current ranks of its variables
3. higher coefficient in the class polynomial.

In particular, $M$ is higher ranked than $M'$ if it is of higher degree. If they have equal degree, then we look at the current ranking of the variables $\sigma$ and look at the current highest variable(s) (there may be a tie amongst variables in the current ranking) in $M$ and $M'$. If $M$ contains a variable of a higher rank that is not in $M'$, then we say that $M$ ranks higher. If the previous two checks fail, then we finally say $M$ is higher if its coefficient is greater than that of $M'$. 

At the beginning, all variables have the same rank, so the first distinction
is mostly by degree and multiplicity. As ranks become more specific, the rank
pattern of a monomial carries more information. For example, `x_0x_1` and `x_0x_2` are tied if we have partial ranking $0 \to (1,2)$, which means that $0$ is ranked the highest and $1,2$ are tied. However, if we find out later that $1 \to 2$, then we actually have that $x_0x_1$ is the higher ranked monomial.  

In step 2, we look at scan each monomial, starting from the highest ranked monomial, and the highest ranked class polynomial. If there are multiple monomials of the same rank, then we consider all of them at once. Thus, the initial ranking is created by first looking at the list of monomials of highest degree and coefficient in the first class polynomial. Suppose we have `x_0x_1` and `x_0x_2` as the list of monomials of highest rank in the first class monomial. As we start with $0,1,2$ all being tied, these two monomials are also equally ranked. However, since $x_0$ occurs more than both $x_1$ and $x_2$, but $x_1$ and $x_2$ are tied, we get the initial ranking $0 \to (1,2)$. Any future ranking now attempts to split the remaining tie(s). We then look to the next highest ranked monomial(s) to try to break this tie, until we exhaust all class polynomials. 

Whenever a tie is successfully broken, he algorithm then restarts the
entire refinement process because the new ranks may cause previously identical
monomials, levels, and variables to become distinguishable. Of course, the partial ranking of variables $\sigma$ carries through as we continue our attempts at breaking the remaining tie(s).

#### Step 3: compare individual output polynomials

If the class-polynomial scan cannot split any remaining tie, the algorithm
examines the tied variables' own output polynomials.

Each polynomial is converted to a temporary key by replacing every variable
with its current rank. Recall that each individual polynomial lives in $GF(2)$ and so we don't need to worry about coefficients here. The ranks within each monomial are sorted, and the
monomials are sorted by lexicographically (by our earlier definition of monomial ranking). This key ignores the
original wire numbers but preserves everything currently known about the
polynomial's structure. This ranking allows us to attempt to break ties between wires. 

Tied variables whose output-polynomial keys differ are separated. As before,
the algorithm applies one split and immediately restarts refinement from the beginning (step 2), carrying the current wire ranking $\sigma$ with it. 

#### Step 4: dynamic rank-class polynomials

The original classes $C_i$ were based only on degree profiles. Later
refinement may discover more meaningful groups. The algorithm therefore builds
a second family of aggregate polynomials $P_{D_i}$, one for each current rank
class $D_i$, where tied wires are put into the same class.

The algorithm scans these dynamic class polynomials with the same monomial-level and
variable-frequency procedure used in step 2 on the class polynomials. 

Once again, on tie-break, we immediately start over from step 2, carrying $\sigma$ over. 

Steps 2 through 4 repeat until every variable has a unique rank or no deterministic refinement rule can make progress.

#### Step 5: Rule L for unresolved symmetries

Sometimes several variables remain structurally tied. This can happen because
the polynomial system has a real automorphism: swapping those variables leaves
all currently visible structure unchanged. Choosing the lowest original wire
number would be fast, but it would not be canonical because a prior wire
renaming could change that choice. 

Rule L resolves the first remaining tied class by trying every member as the
next variable in the order. For each candidate, the algorithm:

1. promotes that candidate ahead of the other tied variables;
2. recursively reruns all refinement rules;
3. fully renames the resulting polynomial vector;
4. serializes the result into a lexicographically comparable form.

The branch producing the lexicographically smallest complete polynomial system
is selected. In other words, this is a *backtracking* step. Most variables are normally separated by the
earlier refinement rules, so Rule L searches only the residual symmetric groups rather than all $n!$ wire permutations.

#### Step 6: construct the canonical form

Once every variable has a final position, the algorithm applies that single
ordering everywhere:

- output polynomials are placed in canonical wire order;
- variables inside every monomial are renamed to their canonical positions;
- monomial sets remain interpreted over $\mathbb F_2$.

The returned permutation is `final_order`, where

```text
final_order[canonical_position] = original_dense_wire
```

This direction is important when a canonical database circuit is later
rewired onto the wires of the subcircuit being replaced.

Finally, `trim_canonicalized` removes trailing identity outputs of the form
$P_i=x_i$, but only when $x_i$ is not used by any other output polynomial.
These wires carry no information about the nontrivial part of the function and
do not need to appear in the database key. This allows us to have canonical forms of circuits that are equivalent in functionality, yet differ in the number of wires. 

As none of our rules used information about wire labelings, the result is independent of the input wire labels. Relabeling circuits that are simply relabelings of each under under their canonicalizations, will thus yield the same canonical circuit. 

The implementation is in
[`canonicalize_polys_4()`](../src/circuit/circuit.rs), with the main refinement
loop in `canon4_run()`.

### 3.3 Reversal Storage

Every ECA57 gate is self-inverse. If a circuit applies the gates

```text
g1, g2, ..., gm
```

then its inverse applies the same gates in the opposite order:

```text
gm, ..., g2, g1
```

The forward circuit and its inverse usually have different polynomial
representations, but they contain the same replacement information in opposite
directions. The regular rainbow table can therefore normalize the pair by
canonicalizing both gate orders and storing circuits under only one of the two
forms. The chosen form is the one with the lexicographically smaller canonical
polynomial vector. If those vectors are equal, the canonical gate sequences
provide the tie-breaker.

Our rainbow tables will only store one direction of the canonical circuit. During replacement lookup, the code first queries the forward canonical form.
If it misses, it canonicalizes the reversed gate sequence and queries the
inverse form. A hit on the reversed form returns a circuit implementing the
inverse of the desired subcircuit, so the replacement's gate sequence is
reversed once more before rewiring and insertion. Because every gate is
self-inverse, this second reversal restores the required forward
functionality.

```text
subcircuit
    |
    +-- forward lookup succeeds --> use replacement directly
    |
    +-- reverse lookup succeeds --> reverse replacement gates, then use it
```

This forward/inverse implementation does not always need to be used, as we will see in other parts of our database. 

The selection between forward and inverse canonical forms is implemented by
[`CircuitSeq::canonicalize_polys()`](../src/circuit/circuit.rs). Runtime
forward-then-reverse lookup is implemented in
[`compress_lmdb()`](../src/replace/replace.rs) and the replacement helpers in
[`src/replace/pairs.rs`](../src/replace/pairs.rs).

### 3.4 Rewiring Replacements

Circuits in the database are stored using canonical wire labels, but the
subcircuit being replaced may use different physical wire numbers. A database
replacement must therefore be translated out of canonical coordinates before
it can be inserted into the full circuit.

There are three wire-labeling spaces involved:

1. **Physical wires:** the actual wire numbers used by the full circuit.
2. **Dense wires:** the wires used by the selected subcircuit, temporarily
   renamed to `0, 1, ..., k-1`.
3. **Canonical wires:** the ordering chosen by polynomial canonicalization and
   used by the rainbow table.

For example, suppose a subcircuit uses physical wires `[3, 7, 11]`. Before
canonicalization, they are densely renamed:

```text
dense 0 -> physical 3
dense 1 -> physical 7
dense 2 -> physical 11
```

Polynomial canonicalization may then determine that the canonical order is
`[dense 2, dense 0, dense 1]`. A replacement returned by the database is
written in that canonical order. Rewiring first maps each canonical wire back
to its dense wire, and then maps each dense wire back to the corresponding
physical wire:

```text
canonical wire -> dense wire -> physical wire
```

The returned wire permutation from our polynomial canonicalzation performs the first mapping.
The saved mapping we perform before polynomial canonicalization is saved and allows us to perform the second:

A replacement can use more distinct wires than the original subcircuit. When
this happens, the existing dense wires retain their original physical
assignments, and each additional replacement wire is assigned a randomly
chosen physical wire that the original subcircuit did not use. The completed
mapping is then applied to every gate in the replacement before it is spliced
into the circuit.

The relevant implementation is in
[`CircuitSeq::canonicalize_polys_single()` and
`CircuitSeq::unrewire_subcircuit()`](../src/circuit/circuit.rs), with the
replacement-side mapping performed by `compress_lmdb()` in
[`src/replace/replace.rs`](../src/replace/replace.rs).

## 4. Rainbow Tables

In order to make *structurally* different replacements while maintaining
functionality, we use a rainbow table rather than attempting to generate a
truly random circuit of equal functionality at replacement time. The rainbow
table groups together circuits that have the same canonical polynomial form.
Once we know the canonical form of a subcircuit, we can look up other circuits
that compute the same function and choose one with the properties we want.

We store these circuits in LMDB. LMDB is a B-tree-style key-value database,
which fits our use case because the database is constructed ahead of time and
mixing mainly performs read-only lookups. Building a B-tree containing hundreds
of millions or billions of keys is expensive, so we split each rainbow table
into 256 independent shards. Since one lookup concerns only one canonical
polynomial, we never need to compare keys across shards.

The key is produced as follows:

```text
Subcircuit
    |
    | polynomial canonicalization
    v
Canonical polynomial vector
    |
    | deterministic serialization
    v
Polynomial byte representation
    |
    | XXH3-128
    v
128-bit database key
```

XXH3 is not a cryptographically secure hash. We use it because it is extremely
fast and provides sufficient collision resistance for this database. The
first byte of the hash chooses one of the 256 shards, and the complete hash is
used as the LMDB key inside that shard. The regular shards are named `00`
through `ff`; the curated shards are named `curated_00` through
`curated_ff`.

The value associated with a key is a list of canonical circuits that share
that polynomial form. These circuits may use different numbers of gates and
different numbers of wires. They are stored consecutively in
length-prefixed blob form:

```text
[blob length][circuit blob][blob length][circuit blob]...
```

Each gate in a circuit blob uses three bytes, one byte for each wire index.
The length prefix therefore gives the number of following bytes, not the
number of gates. During lookup, we read one length, reconstruct the circuit
from the following blob, and continue until the entire value has been read.

We use two related rainbow tables during mixing: the regular database and the
*curated* database.

### 4.1 Regular Database

<!-- TODO: add DB code back -->

In order to create the general-purpose database, we enumerate possible
circuits of a given gate length. We do not need to consider every circuit on
every possible set of wire labels. For instance, in the one-gate case, it is
enough to consider `[0,1,2]` and the meaningful permutations of those three
pins. A gate such as `[3,4,5]` is only a rewiring of the same structure and
will have the same canonical polynomial form.

Once we have the canonical circuits of length $n$, we can generate candidates
of length $n+1$ by prepending or appending one possible gate. We can also
generate length-$n$ candidates by concatenating known canonical circuits of lengths $i$
and $j$, where $i+j=n$. Both methods create a large overcount: many candidates
are related by commuting-gate order, wire relabeling, or even reversal. We canonicalize the candidates again and merge those that
produce the same canonical polynomial into the running list of canonical circuits under that polynomial. 

At the moment, we have generated all canonical circuits through six gates,
seven-gate circuits with a minimum of 15 wires, and eight-gate circuits with a
minimum of 18 wires.

We primarily use the regular database during compression. The compression
procedure:

1. samples a subcircuit;
2. computes its canonical polynomial form;
3. hashes that form and queries the corresponding shard;
4. keeps only candidates shorter or of equal gate-length than the sampled subcircuit;
5. chooses a candidate with the minimum gate count;
6. chooses randomly when multiple minimum-length candidates remain;
7. rewires the chosen circuit back onto the physical wires.

The regular database is also used as a fallback during expansion when the
curated database, discussed below, does not contain the requested canonical
form. Expansion performs the same lookup but keeps candidates that are longer
than the input subcircuit instead of shorter.

### 4.2 Curated Database

We already know that not all replacements are equally useful. A replacement
that differs only slightly from the original circuit may preserve too much of
its local structure. The curated database attempts to find replacements whose
equivalence is spread across a larger, locally incompressible structure.

Let $P$ be the set of canonical polynomial forms, and let $C_p$ be the set of
canonical circuits associated with $p\in P$. If $c_a,c_b\in C_p$, then the two
circuits compute the same function. Therefore,

$$
c_a \mathbin{\|} c_b^{-1}
$$

is an identity circuit, where $\|$ denotes concatenation and $c_b^{-1}$ is
obtained by reversing the gate order of $c_b$.

We call an identity **minimal** when the entire circuit computes the identity,
but every proper contiguous subcircuit is incompressible using the regular
database. Intuitively, the cancellation is distributed across the whole
identity rather than appearing as an obvious shorter identity inside it.

Suppose a minimal identity is

$$
I=i_1i_2\cdots i_m.
$$

Splitting it after gate $i_k$ gives a prefix

$$
A=i_1\cdots i_k
$$

and a suffix

$$
B=i_{k+1}\cdots i_m.
$$

Since $AB$ is the identity, $A=B^{-1}$. Because ECA 57 gates are self-inverse,

$$
B^{-1}=i_mi_{m-1}\cdots i_{k+1}.
$$

The prefix and reversed suffix are therefore equivalent circuits. We
canonicalize the prefix, apply the same wire ordering to both circuits, and
store both under the prefix's hashed canonical polynomial form. Repeating this
for every split point of every minimal identity produces the curated database.

For example, from

$$
i_1 i_2 i_3 ... i_m
$$

we obtain the equivalent pair

$$
i_1 i_2 \text{    and     }  i_m ... i_3
$$

The importance of this construction is that the curated database can contain
circuits longer than those originally enumerated in the regular database. A
minimal identity built from two six-gate circuits may have twelve gates, and a
split of that identity may produce equivalent circuits of lengths two and ten.
This makes the curated database especially useful for meaningful
*expansions*: a short circuit can be replaced by a much longer equivalent
circuit whose internal pieces are not immediately compressible on their own.

Our curated database contains every one-gate and two-gate circuit from the
regular database. It misses one three-gate circuit and many circuits at larger
sizes. Thus, when a curated expansion lookup misses, we can fall back to the
regular database. Curated lookup itself uses the forward canonical form; the
regular fallback may also try the reversed form described in Section 3.3.

The database opening and shard naming are implemented in
[`src/replace/main_mix.rs`](../src/replace/main_mix.rs). The regular
compression and expansion lookups are in
[`src/replace/replace.rs`](../src/replace/replace.rs), while curated lookup and
candidate selection are in
[`src/replace/pairs.rs`](../src/replace/pairs.rs).

## 5. Gadgetization

Circuits that have low algebraic degree are prone to differential attacks. For
more information on these types of attacks, see
[Algebraic Cryptanalysis](https://link.springer.com/book/10.1007/978-0-387-88757-9).
Thus, our later overall mixing algorithm can be said to work best on circuits
whose internal regions have high algebraic degree. Since we want our mixer to
work on general circuits, this *gadgetization* step serves to integrate a
random, high-degree auxiliary computation into the middle of the original
circuit.

A circuit has **high** algebraic degree when the polynomials describing its
internal wires contain terms whose degree is close to the maximum permitted by
the number of wires. Of course, because we are maintaining the original
functionality, gadgetization cannot simply change the degree of the final
function computed on the output wires. What is important is that the
intermediate states in the center of the circuit have high degree, even though
the original function is recovered at the end.

To achieve this, we utilize an extra $n$ auxiliary wires. The gadgetized
circuit thus lies on $2n$ wires. The first $n$ outputs will have the
functionality of the original circuit, while the latter $n$ outputs are
allowed to contain random values. The goal is therefore to integrate the
auxiliary computation into the original computation without changing what is
eventually returned on the first $n$ wires.

Suppose $r_j$ is an auxiliary value with high algebraic degree and $w_i$ is a
*computation* value from our original circuit. One way to imbue the physical
representation of $w_i$ with high algebraic degree is to form
$w_i\oplus r_j$. Of course, if we replace $w_i$ by this value directly, we
have lost the ability to read $w_i$ without later removing $r_j$. Simply
removing the same mask at every use would also expose the original computation
in the middle of the circuit.

Instead, we represent each original computation value $w_i$ using a secret
value $s_i$ and an auxiliary value $r_i$:

$$
w_i=s_i\oplus r_i.
$$

Here, $w_i$ is the value used by the original circuit $C$, while $s_i$ and
$r_i$ are the two physical values used by the gadgetized circuit. The
gadgetizer only needs to remember which physical wires currently hold $s_i$
and $r_i$.

If our original circuit is $C=g_1g_2\cdots g_m$, we would like to replace each
gate $g_k$ with a gadget $F_k$ such that applying $F_k$ to the physical values
$s_i$ and $r_i$ has the same effect as applying $g_k$ to the original values
$w_i$. We do not want to recover $w_i$, apply the original gate, and then
encode it again. Instead, the gadget should *homomorphically* compute the
ECA57 gate while the original values remain encoded.

As each computation value $w_i$ is represented by a secret value $s_i$ and an
auxiliary value $r_i$, we also need gadgets that can randomize these physical
values and change their locations without changing $w_i$. These are the SG and
RG gadgets described below.

Before gadgetization begins, we also shoot the original gates left or right
past non-colliding gates. This changes the visible gate order while preserving
functionality, so two gadgetizations of the same input do not need to begin
from the same ordering.

### 5.1 Left Bookend

The left bookend converts the ordinary $n$-wire input into the secret-wire
representation used by the middle of the gadgetized circuit. It consists of
two pieces, which we call $Z$ and $M_p$.

First, $Z$ applies a balanced sequence of random ECA57 gates. This serves to create an auxiliary circuit with high algebraic degree. The active wire
of every $Z$ gate lies in the auxiliary half, while its controls may come from
any of the $2n$ wires. Thus, $Z$ changes only the auxiliary outputs while
allowing their values to become complicated functions of both the original
and auxiliary inputs. The number of gates used in each $Z$ bookend is

$$
\max(2n\lfloor\ln n\rfloor,64).
$$

After $Z$, the $M_p$ transformation encodes every original computation value
$w_i$ as $s_i\oplus r_i$. Initially, $w_i$ is on wire $i$, and its
corresponding auxiliary value is on wire $n+i$. For each value, we randomly
choose two distinct destination wires for $s_i$ and $r_i$. An 11-gate $W_i$
gadget
then transforms

```text
(w_i, auxiliary_i, old(s destination), old(r destination))
```

into

```text
(old(s destination), old(r destination),
 w_i XOR auxiliary_i, auxiliary_i).
```

Writing the four input wires as

```text
q0 = computation
q1 = auxiliary
q2 = destination secret wire
q3 = destination auxiliary wire,
```

the actual $W_i$ gadget is

```text
[q0, q3, q2]
[q3, q2, q1]
[q1, q3, q2]
[q2, q0, q1]
[q2, q1, q0]
[q0, q1, q2]
[q0, q2, q1]
[q1, q0, q3]
[q3, q0, q1]
[q3, q1, q0]
[q1, q3, q0]
```

The $W_i$ gadget essentially does two things at once: it encodes $w_i$ as
$s_i\oplus r_i$, and it randomizes which physical wires hold $s_i$ and $r_i$
amongst all $2n$ wires. One simpler way to achieve this would be to permute all
$2n$ wires and then separately create the secret-auxiliary bindings. This
"4 way dance" between $q_0,q_1,q_2,q_3$ is simply one way to achieve both
encoding and randomizing the wire roles.

As each ECA57 gate is self-inverse, $W_i^{-1}$ is obtained by applying these
same 11 gates in the opposite order.

The original computation value is unchanged, but it is now represented by two
physical values that depend on the randomized auxiliary state. Namely,

$$
w_i=s_i\oplus r_i.
$$

The $W_i$ gadgets also move the values that were previously stored at the two
destination wires. Because of this, we cannot choose all secret-wire locations in
advance and assume that they remain valid. The implementation keeps a live
record of:

- where each unencoded computation value currently lives;
- where each auxiliary value currently lives; and
- which physical wires currently hold every $s_i$ and $r_i$.

At the end of the left bookend, the gadgetizer stores the state as

```text
pairs[i] = (physical wire holding s_i, physical wire holding r_i).
```

For example, the mapping

| Computation value | Secret wire | Auxiliary wire | Decoding relation |
|---:|---:|---:|---|
| $w_0$ | $s_0$ on `7` | $r_0$ on `2` | $w_0=\text{wire}[7]\oplus\text{wire}[2]$ |
| $w_1$ | $s_1$ on `5` | $r_1$ on `0` | $w_1=\text{wire}[5]\oplus\text{wire}[0]$ |
| $w_2$ | $s_2$ on `1` | $r_2$ on `4` | $w_2=\text{wire}[1]\oplus\text{wire}[4]$ |

would be stored as

```text
pairs = [(7, 2), (5, 0), (1, 4)]
```

The implementation calls this mapping `pairs`, but conceptually each entry
records the physical locations of $s_i$ and $r_i$. This mapping describes the
encoded circuit on which the SG gadgets, which allow us to do our
*homomorphic computation*, operate.
The actual wire numbers are chosen randomly and may later change when an RG
gadget randomizes the secret values, auxiliary values, or their locations.

### 5.2 SG Gadgets

An SG, or *secret gadget*, replaces one gate of the original circuit and computes it homomorphically.
Suppose the original gate is

```text
[a, b, c],
```

so that it updates computation value $a$ using computation controls $b$ and
$c$. At the time this gate is processed, the gadgetizer looks up the three
current encodings:

$$
w_a=s_a\oplus r_a,\qquad
w_b=s_b\oplus r_b,\qquad
w_c=s_c\oplus r_c.
$$

The SG consists of six ECA57 gates acting on

```text
s_a, s_b, r_b, s_c, r_c.
```

In `[active, positive_control, negative_control]` notation, the SG is

```text
[r_c, s_b, r_b]
[s_a, r_c, r_b]
[s_a, s_b, r_c]
[r_c, s_b, r_b]
[s_a, r_b, s_c]
[s_a, s_c, s_b]
```

The auxiliary wire $r_a$ of the active computation value does not occur in
the gadget and is left unchanged.

After those six gates, the updated secret value $s_a'$ and the unchanged
auxiliary value $r_a$ decode to the result of the original ECA57 gate:

$$
w_a'=s_a'\oplus r_a
  =w_a\oplus(w_b\lor\neg w_c).
$$

The two control encodings continue to represent $w_b$ and $w_c$. Thus, the
six physical gates perform exactly one original gate on the encoded state
without first exposing $w_a$, $w_b$, or $w_c$ on an individual wire.

Every original ECA57 gate is replaced by one SG. We would be able to form the original functionality of our original circuit solely using SG gadgets. However, this could potentially leave a trail on the heavily structured gadgetized circuit that we will attempt to blur, even if we can't quite fully break it. 

### 5.3 RG Gadgets

The RG, or *rerandomization gadget*, changes the physical representation of
the secret-wire state while preserving every represented value. We currently use three
types of RG.

**RG1** acts on two secret wires. It transforms their four physical wires so
that the two represented values exchange secret wires. After the six-gate
gadget, the two physical wires previously associated with computation value
$i$ represent computation value $j$, and vice versa. The gadgetizer then swaps
the two entries in its internal `pairs` mapping so later SGs continue to find
the correct values.

Using the local wire order, where $w_i = s_i \oplus r_i$ and $w_j = s_j \oplus r_j$

```text
q0 = s_i
q1 = r_i
q2 = s_j
q3 = w_j,
```

RG1 is

```text
[q1, q2, q3]
[q0, q3, q2]
[q3, q1, q0]
[q2, q0, q1]
[q0, q3, q2]
[q1, q2, q3]
```

If the original
secret wires are

```text
s_i = w_i XOR r_i
s_j = w_j XOR r_j,
```

then after the six-gate RG1 they are tracked as

```text
s_i = w_j XOR r_j
s_j = w_i XOR r_i.
```

In other words, we maintain our pairings, but swap the secret values between two computation wires. 

**RG2** also acts on two secret wires, but instead of merely exchanging them,
it breaks the old bindings and forms two new secret wires. If the original
secret wires are

```text
s_i = w_i XOR r_i
s_j = w_j XOR r_j,
```

then after the six-gate RG2 they are tracked as

```text
s_i = w_j XOR r_i
s_j = w_i XOR r_j.
```

Using the same local wire order as RG1, RG2 is

```text
[q0, q3, q2]
[q1, q0, q2]
[q2, q0, q3]
[q2, q3, q0]
[q1, q3, q2]
[q3, q0, q2]
```

The XOR of the two components of each new secret wire is still the correct
computation value. However, the physical relationship between the computation
and auxiliary wires has changed, which makes the secret-wire structure less
stable across the circuit.

**RG3** rerandomizes a single secret wire. It applies the same ECA57 update to
its computation and auxiliary wires, using two other physical wires as
controls. If that common update is $f$, then

$$
(w_i\oplus f)\oplus(r_i\oplus f)=w_i\oplus r_i=s_i.
$$

The represented value is therefore unchanged, even though both of its
physical components have changed.

If $u$ and $v$ are the two randomly selected control wires, RG3 is simply

```text
[w_i, u, v]
[r_i, u, v]
```

From our earlier notation, we have $f = u \lor \neg v$.

After every `--rg-frequency` SG gadgets, the implementation chooses one of
RG1, RG2, or RG3 at random. The default frequency is `2`, meaning that two
original gates are simulated and then one rerandomization gadget is inserted.
For RG1 and RG2, groups of two secret wires are shuffled into a queue before
use. For RG3, individual secret wires are shuffled similarly. This
distributes rerandomization across the secret-wire state rather than
repeatedly selecting the same few values.

The important point is that the gadgetizer's internal `pairs` mapping changes
after RG1 and RG2, but the represented computation does not. The next SG reads
the updated mapping and continues the original computation on the newly
represented secret wires.

Of course, this bookkeeping of computation, auxiliary, and secret wires is
only used to facilitate the construction of our gadgets. Once gadgetization is
over, none of this mapping is saved in the output. The only output is the
resulting sequence of ECA57 gates, so an adversary cannot simply look up which
physical wires formed each secret wire during gadgetization.

### 5.4 Right Bookend

After all original gates have been replaced by SGs and mixed with RGs, each
final computation output $w_i$ still exists as the XOR of its secret value
$s_i$ and auxiliary value $r_i$. These values may now lie anywhere among the
$2n$ physical wires. The right bookend decodes $w_i$ onto physical wire $i$.

If either $s_i$ or $r_i$ is already located on output wire $i$, we XOR the
other value into it. Since $w_i=s_i\oplus r_i$, this places the decoded
computation value $w_i$ on output wire $i$. Otherwise, we use the inverse
$W_i$ gadget to place $s_i\oplus r_i$ onto wire $i$. The old contents of any
displaced wires are relocated rather than discarded.

Outputs are decoded in increasing order. Once wire $i$ has been finalized, the
live-location tracking prevents later decoding operations from overwriting it.
After all $n$ values have been decoded, the lower half satisfies

$$
(\text{output}_0,\ldots,\text{output}_{n-1})=C(x),
$$

where $C$ is the original circuit.

Finally, another random $Z$ circuit is applied. As in the left bookend, its
active wires are restricted to the upper $n$ auxiliary wires, so it cannot
change the recovered outputs in the lower half. It further randomizes the
auxiliary outputs and removes the need for them to return to their original
values.

The complete transformation can be summarized as

```text
Original n-wire circuit
        |
        | random commuting-gate shooting
        v
Reordered n-wire circuit + n auxiliary wires
        |
        | initial Z on auxiliary wires
        v
Randomized auxiliary state
        |
        | M_p sharing transformation
        v
2n-wire encoded state: s_i = w_i XOR r_i
        |
        | SG simulation + RG rerandomization
        v
Encoded final computation values
        |
        | inverse sharing / live decoding
        v
Original outputs on wires 0,...,n-1
        |
        | final Z on auxiliary wires
        v
Gadgetized 2n-wire circuit
```

Our gadgetized circuit retains functionality on the original $n$ wires, has random output on the latter $n$ wires, and has high algebraic degree throughout the middle of the circuit. 

The implementation is in
[`gadgetize()` and the SG/RG helpers](../src/replace/gadgets.rs).

## 6. SAMFs

We have many hard-coded circuits that **swap and maybe flip** (SAMF) two wires, which
we abbreviate as SAMFs. A SAMF acts on two selected physical positions
$\ell<h$. Its net operation is to first swap the two wire values and then
optionally negate either of the two resulting positions.

There are four SAMF types:

| Type | Operation after the swap |
|---:|---|
| `0` | Negate neither position |
| `1` | Negate position $\ell$ |
| `2` | Negate position $h$ |
| `3` | Negate both positions |

If the original values at positions $\ell$ and $h$ are $x_\ell$ and $x_h$,
then the four types compute

| Type | New value at $\ell$ | New value at $h$ |
|---:|---|---|
| `0` | $x_h$ | $x_\ell$ |
| `1` | $\neg x_h$ | $x_\ell$ |
| `2` | $x_h$ | $\neg x_\ell$ |
| `3` | $\neg x_h$ | $\neg x_\ell$ |

The negation convention is always stated in the **post-swap wire space**. For
example, type `1` means that position $\ell$ is negated after the values have
been exchanged. This convention matters when many SAMFs are composed.

Each logical SAMF can be implemented by several different hard-coded ECA57
circuits. The implementation randomly chooses a minimum-depth circuit from
either a three-wire or four-wire pool, depending on the number of available
wires. The additional wires are temporary helpers; the net effect on the two
selected wires is still exactly the swap-and-maybe-flip operation above.

The purpose of inserting a SAMF is not to preserve the value on each physical
wire at that point in the circuit. Instead, we allow the correspondence
between logical wires and physical positions to change, and then rewrite the
remainder of the circuit to follow that new correspondence. At the end, we
undo the accumulated wire permutation and negations.

### 6.1 Tracking Wire Positions

Whenever we insert a SAMF, every later gate must be adjusted to the new wire
labeling. Suppose the next original gate is

```text
[0, 1, 2]
```

and an earlier SAMF exchanged physical positions `0` and `5`. The logical
value that used to be found at position `0` is now found at position `5`, so
the rewritten gate is

```text
[5, 1, 2].
```

With many SAMFs, it is not sufficient to remember only the most recent swap.
We store the complete sequence of transpositions and evaluate a logical wire
through all of them to find its current physical position. If the accumulated
wire permutation is $T$, then an original gate `[a,b,c]` is emitted as

```text
[T(a), T(b), T(c)].
```

The `Transpositions` structure records this sequence. It can also convert the
sequence into one net permutation, compose the transpositions from multiple
passes, or reconstruct a swap sequence from a permutation.

### 6.2 Tracking Negations

In addition to the current wire position, we store one pending-negation bit
for every physical position. A value of `1` means that the value currently at
that position is the negation of the logical value expected there.

When a SAMF acts on positions $\ell$ and $h$, their pending-negation bits are
first swapped because the wire values themselves move. We then toggle the
bits selected by the SAMF type:

```text
swap(mask[ell], mask[h])

type 0: toggle neither
type 1: toggle mask[ell]
type 2: toggle mask[h]
type 3: toggle both
```

This gives the negation state in the current, post-SAMF wire space.

Before emitting an ECA57 gate, we inspect the pending-negation bits on its two
controls. Recall that `[a,b,c]` updates the active wire according to
$b\lor\neg c$. If either control is secretly negated, then the condition under
which the gate fires has changed. We therefore insert a hard-coded ECA57 NOT
gadget on that control and clear its pending-negation bit before emitting the
gate.

A pending negation on the active wire does not need to be corrected before the
gate. The ECA57 update has the form

$$
a\leftarrow a\oplus f(b,c).
$$

Negating $a$ before this update gives

$$
\neg a\oplus f(b,c)
  =\neg\left(a\oplus f(b,c)\right),
$$

so the active-wire negation may remain pending after the gate. Only pending
negations on the controls affect which update is performed.

For example, suppose the next rewritten gate is `[5,1,2]`, and the current
negation mask says that position `1` is negated. We first insert a NOT gadget
on position `1`, clear that mask bit, and then emit `[5,1,2]`. The gate now
sees the same logical control value as the original circuit.

### 6.3 Applying the Unsamf

At the end of the circuit, we must restore the original external wire
meaning. We call this final correction the **unsamf**.

The unsamf is constructed from two pieces of state:

1. the net wire permutation produced by all inserted SAMFs; and
2. the remaining pending-negation mask in the current wire space.

We first reduce the complete transposition list to its net permutation and
decompose that permutation back into a swap sequence. The pending negations
are then folded into the negation types of these inverse swaps whenever
possible. For example, if a wire participating in an inverse swap is still
negated, the inverse SAMF type is changed so that the swap also removes that
negation.

Some wires are fixed points of the net permutation and therefore do not occur
in the reconstructed swap sequence. If a fixed wire still has a pending
negation, we append an explicit NOT gadget for it.

Finally, the inverse swaps are emitted in reverse order, and each hard-coded
SAMF circuit is itself reversed. Since every ECA57 gate is self-inverse,
reversing the gate sequence computes the inverse SAMF. After all inverse SAMFs
and leftover NOTs have been applied, the output wires once again have their
original labels and logical values.

Before appending an inverse SAMF or leftover NOT verbatim, the implementation
tries to hide it using the rainbow tables. This is just one use of our rainbow tables. It combines the undo gadget with up
to three gates already at the end of the output and looks for an equivalent
replacement:

1. first, a shorter curated replacement;
2. then, any curated equivalent replacement;
3. finally, any equivalent replacement from the regular database.

If every lookup misses, the undo gadget is appended directly. Thus, the
rainbow-table lookup affects how visible the unsamf is, but never whether the
wire permutation and negations are correctly undone.

Overall, this gives us an equivalentally functional circuit, but with values slightly randomized with linear transformations in the middle. 

The SAMF generation, tracking, and undo logic is implemented in
[`src/replace/transpositions.rs`](../src/replace/transpositions.rs), primarily
through `Transpositions` and `apply_unsamf()`.

## 7. The Collision Game

Outside of simple linear transformations such as SAMFs, we would like to make
small, local, functionally equivalent circuit replacements as part of our
mixing. In order to do so, we grab small subcircuits and attempt to expand them
through our rainbow tables. In addition, we attempt to hide a SAMF inside some
of these expansions, as opposed to simply inserting the complete SAMF
verbatim. This makes the SAMF less visible and helps *lock* the new wire
labeling into the surrounding nonlinear circuit.

We call this process the ***collision game***. Before beginning, we randomly
choose either the forward or reversed direction. In the forward direction, we
use the circuit as given. In the reversed direction, we reverse the entire
gate sequence before playing the game and reverse it once more after the
game, plain SAMF insertion, and unsamf are complete.

The collision game itself is identical in either direction. We begin with the
first gate in the remaining circuit, which we call $g_1$, and shoot it to the
right. Whenever the next gate commutes with $g_1$, we swap their order and
continue moving $g_1$. The gates passed by $g_1$ are emitted in their original
relative order. We stop when $g_1$ reaches the end of the circuit or meets the
first gate with which it collides.

For example,

```text
g1  g2  g3  gi  gi+1 ...
 \_______/
  commuting

        g1 collides with gi
              |
              v
g2  g3  [g1  gi]  gi+1 ...
```

At this point, $g_2$ and $g_3$ have already been emitted. We attempt to replace
a window beginning with the newly adjacent pair $[g_1,g_i]$. The window may
also include gates immediately after $g_i$. By default, we use a window of $3$, and so we would use the curated database to
replace $g_1 g_i g_{i+1}$ with a longer equivalent circuit. We then use the end of
that expansion as context for hiding one SAMF.

If no expansion is found, we emit $g_1$ immediately before $g_i$. The next
iteration then uses $g_i$ as the new shot gate and continues through the
remaining suffix. If a replacement is made, we consume the collider and any
additional gates included in the replacement window. We emit every gate of
the expansion except its final gate, and use that final gate as the next gate
to shoot through the untouched suffix.

### 7.1 Curated Expansion

Let `gates_ahead_expand` be the largest expansion-window size. At a collision,
we first try the largest available window beginning with the shot gate and its
first collider:

```text
[g_1, g_i, g_(i+1), ..., g_(i+k-2)]
```

where

```text
k = min(gates_ahead_expand, shot gate + gates remaining in the suffix).
```

The window is always at least the two-gate colliding pair. If
`gates_ahead_expand` is less than `2`, it is treated as `2`. By default, our window will have $3$ gates. 

Before looking up the window, we translate every gate through the accumulated
SAMF wire permutation. We also require every control wire in the window to be
**clean**, meaning that it has no pending negation. A pending negation on a
control changes the function of an ECA57 gate, so looking up the uncorrected
window would not necessarily preserve functionality. Active-wire negations do
not prevent expansion for the same reason discussed in Section 6.2.

If the largest window is not clean or does not have a database expansion, we
remove its final gate and try again:

```text
k, k-1, k-2, ..., 2.
```

Thus, a miss on a four-gate window does not discard the collision. We still
try the corresponding three-gate window and finally the shot/collider pair
itself. If every size misses, no expansion is made, the shot gate is emitted,
and the collider remains at the beginning of the unprocessed suffix.

The curated lookup keeps only candidates longer than the input window. Among
them, it favors circuits that are both long and wide, scoring each candidate
by

```text
number of gates + number of distinct wires
```

and choosing randomly among the highest-scoring candidates. If the curated
database does not contain the canonical form, the lookup may fall back to the
regular database.

Once an expansion is found, it is computed only once. Even if the later SAMF
hiding attempt fails, we keep the expanded circuit rather than returning to
the original short window.

### 7.2 Hiding a SAMF

After obtaining a curated expansion, we choose two distinct random wires that
the hidden SAMF will swap. These two wires remain fixed for this collision,
while we may try several of the four SAMF negation types.

We do not attempt to hide the entire SAMF at once. Instead, we take the first
three gates of one randomly selected hard-coded SAMF and attach them to a
small context ending at the tail of the expansion:

```text
[context expansion gates] [SAMF gate 0] [SAMF gate 1] [SAMF gate 2]
```

The context contains up to `gates_ahead_samf` gates. By default, this is $3$ gates. If the expansion is shorter than the requested context, we reach backward into gates that have already been emitted
immediately before the expansion. For example, with a context size of `3` and
a two-gate expansion, the hiding window uses

```text
[1 preceding output gate] [2 expansion gates] [3 SAMF gates].
```

We canonicalize this complete window and ask the curated database for an
equal-or-shorter equivalent replacement. A successful replacement must
actually absorb the SAMF prefix. We reject a candidate if the same three SAMF
gates still appear consecutively inside it, since this would merely move the
prefix without hiding it.

If the lookup succeeds, we replace the context and SAMF prefix with the
database circuit. Any gates remaining after the first three gates of the SAMF
are then emitted normally:

```text
context + SAMF[0..3]
          |
          | curated compression
          v
hidden replacement + SAMF[3..]
```

Although only the prefix was absorbed, the complete logical SAMF has now been
inserted. We add its transposition to the accumulated wire permutation, update
the pending-negation mask, and rewrite all future gates into the new wire
space exactly as described in Section 6. In this case, the final gate of the
inserted SAMF becomes the next gate to shoot, rather than the final gate of the
expansion. If the complete SAMF suffix was absorbed, we instead continue from
the final gate of the hidden replacement.

If no SAMF type can be hidden, we keep the curated expansion verbatim and do
not insert a SAMF at that collision. The expansion is still a useful
structural replacement, while the failed hide does not change the wire or
negation state.

The overall collision path is therefore:

```text
First remaining gate
        |
        | move right across commuting gates
        v
First collision
        |
        | curated expansion lookup
        v
Longer equivalent circuit
        |
        | append and compress a SAMF prefix
        v
Hidden SAMF succeeds? ---- no ----> keep expansion only
        |
       yes
        |
        v
Emit hidden replacement and SAMF suffix
        |
        v
Update permutation and negation state
        |
        v
Shoot the final SAMF gate through the untouched suffix

If no SAMF was hidden, shoot the final expansion gate instead.
```

### 7.3 Collision-Game Parameters

The collision game is controlled by four parameters:

| Parameter | Default | Meaning |
|---|---:|---|
| `--gates_ahead_expand` | `2` | Maximum number of input gates in the expansion window, beginning with the shot gate and its first collider. On a miss, the window shrinks down to those two gates. |
| `--gates_ahead_samf` | `3` | Number of context gates immediately before the three-gate SAMF prefix. Context may come from both the expansion tail and already-emitted output. |
| `--type_attempts` | `1` | Number of distinct SAMF negation types to try at each collision. Types are sampled without replacement. |
| `--shooting_times` | `1` | Number of complete collision-game passes to run before the later plain SAMF insertion. |

For each collision, `type_attempts` changes only the SAMF negation type. The
two swapped wires are selected once and reused across those attempts. Each
type still chooses one random hard-coded SAMF circuit from the appropriate
pool. The first type that can be hidden wins.

When `shooting_times` is greater than one, the output of one complete pass
becomes the input to the next. The permutations and pending negations from all
passes are composed, so the final unsamf can restore the original external
wire meanings after the repeated game.

The random direction choice is made in `main_mix`, outside of the collision
game itself. This allows us to start at either the beginning and shoot to the right, or to start at the end and shoot to the left. This is important because the collision game leaves its hidden
SAMF state to be combined with the later plain SAMFs and undone by one merged
unsamf. For a reversed game, we therefore reverse back only after that merged
unsamf is finished. 

The collision game is implemented by `shuffled_shooting_game_core()` and
`shuffled_shooting_game_repeated_core()` in
[`src/replace/transpositions.rs`](../src/replace/transpositions.rs).

## 8. Plain SAMF Insertion

Other than hidden SAMFs, we may want to insert more SAMFs to further randomize
our circuits. We thus insert *plain SAMFs* throughout the output of the
collision game. The plain insertion procedure inserts $m$ randomly generated
SAMFs every $x$ gates. More precisely, before every gate whose index is
divisible by $x$, including gate $0$, we generate $m$ random transpositions.
For each transposition, we choose:

- two distinct wire positions;
- one of the four negation types; and
- one compatible hard-coded ECA57 implementation.

We then emit the SAMF circuits, update the accumulated permutation and
negation mask, rewrite the next gate into the current wire space, and
flush any pending negations on its controls.

The collision game and this plain insertion are treated as one combined
shuffle. We compose the wire permutation and negation state produced by both,
then undo them with one merged unsamf. It is during this unsamf that we again
attempt to hide the SAMF circuits, as described in Section 6.

With `--single-end`, the SAMF state from the collision game and its plain
insertion is accumulated across rounds and undone only after the final round.

## 9. Compression

After mixing, we attempt to shrink the circuit back down using the regular
rainbow table. Unlike the curated database used during the collision game, the
regular database is used here to find strictly shorter circuits that compute
the same function.

Compression is randomized and iterative. We do not scan every possible
subcircuit. Instead, we repeatedly select small convex subcircuits, look up
their canonical polynomial form, and replace them whenever the database
contains a shorter implementation.

### 9.1 Selecting a Subcircuit

A subcircuit does not need to begin as one contiguous interval in the circuit.
We first choose a random gate and grow a **convex subcircuit** around it. In
this context, convexity means that the selected gates can be brought together
using only swaps of commuting gates, without moving one selected gate across
a gate with which it collides. This definition of convexity can more clearly be defined by turning a circuit into its graph representation and then using the notion of convexity for graphs. 

<!-- TODO add this -->

We rotate between three selection strategies:

1. prefer gates that introduce new wires, producing wide subcircuits;
2. use a simpler greedy growth procedure; and
3. prefer gates that reuse the existing wires, producing subcircuits with more
   gates on fewer wires.

Once a convex set has been selected, we move every unselected gate between its
first and last gates out of the interval. Each such gate is moved left or
right only when it commutes with every gate that it crosses. The selected
subcircuit is then contiguous and can be replaced as one block.

### 9.2 Rainbow-Table Lookup

Within the selected block, we make several random compression attempts. For
each attempt, we choose a smaller contiguous subcircuit and canonicalize its
polynomial vector as described in Section 3. We take smaller contiguous subcircuits because canonicalizing circuits on many wires can be slow, and replacements on larger subcircuits are rare. We serialize the canonicalized vector of polynomials and
compute its 128-bit XXH3 hash, which determines both the LMDB key and the database shard.

We first look up the ordinary forward canonical form. If that key is not in
the database, we reverse the selected gate sequence, canonicalize the reversed
function, and try that key instead. Since every ECA57 gate is self-inverse,
reversing a circuit gives the inverse permutation. A circuit stored for that
inverse can therefore be reversed again to obtain a replacement for the
original forward computation.

The reversed lookup is only a fallback after a forward miss. We do not look up
both directions and compare their candidates when the forward key succeeds.

The database value may contain several circuits with the same canonical
function. We discard every candidate that is not strictly shorter than the
selected subcircuit. Among the remaining candidates, we keep those with the
fewest gates and randomly choose one when several shortest candidates remain.
If neither direction has a database entry, or if its entry contains no shorter
candidate, the circuit is left unchanged for that attempt.

### 9.3 Rewiring the Replacement

The database canonical circuit is written in minimal and canonical wire labels, not the physical
wire numbers used by the current circuit. We therefore undo the wire
permutation recorded during canonicalization and map the replacement back onto
the wires used by the selected subcircuit.

Some database circuits use more wires than the selected gates themselves. In
that case, we randomly assign their additional canonical wires to currently
unused physical wires. This is why the compressor is allowed to use the full
wire set even when the chosen subcircuit touches only a few wires.

If the replacement came from the reversed lookup, we reverse its gates before
rewiring it. We then splice the shorter replacement into the contiguous
interval occupied by the old subcircuit. 

The complete local replacement is therefore:

```text
random convex subcircuit
        |
        | commute outside gates away
        v
contiguous subcircuit
        |
        | canonicalize and hash
        v
canonical polynomial vector + XXH3 key
        |
        | look up and choose a shortest strictly smaller candidate
        v
canonical replacement circuit
        |
        | undo canonical/minimal wire labels
        v
replacement on physical circuit wires
        |
        | splice into the original interval
        v
shorter equivalent circuit
```

### 9.4 Cancellation and Repetition

ECA57 is self-inverse. Two adjacent identical gates therefore cancel:

```text
g g = identity.
```

We remove these pairs before and after local compression. After removing a
pair, we step backward and check again, since the removal may bring another
identical pair together.

For large circuits, `compress_loop()` divides the circuit into random
contiguous chunks. It uses approximately one chunk per 1,500 gates, up to four
times the available Rayon thread count, and compresses those chunks in
parallel. The chunks are joined back together after each iteration. New random
boundaries are chosen on the next iteration, allowing later passes to reach
subcircuits that crossed an earlier chunk boundary.

Each chunk performs samples `100` convex subcircuits, and each selected convex
block performs `10` smaller rainbow-table compression attempts. The selection
strategy rotates between the three modes described in section 9.1 on each complete iteration.

Compression continues until the circuit has reduced by fewer than `50` gates
over the previous six iterations. Thus, one iteration with no reduction does
not immediately stop the process; later random selections may still find a
replacement that earlier attempts missed. This is a loose indicator on when our compressor has *stabilized* our circuit. It is possible it can compress further, but unlikely to compress much further. 

If our compressor cannot reduce a mixed circuit back to its original,
pre-mixing size, then we call the circuit ***incompressible*** with respect to
our current rainbow table and compression procedure. This does not mean that
the circuit is theoretically impossible to compress. Our compressor is heavily
dependent on the contents of its rainbow table, and a significantly stronger,
or even theoretical, rainbow table may be able to reduce the circuit further.
However, the circuit could then also be remixed using that stronger rainbow
table.

The compression loop and replacement logic are implemented in
[`src/replace/replace.rs`](../src/replace/replace.rs), primarily through
`compress_loop()`, `compress_big_ancillas()`, and `compress_lmdb()`.

## 10. One Complete Mixing Round

Before beginning our rounds, we gadgetize the circuit once to give it
high algebraic degree. We then run the collision game, including its curated
expansions and hidden SAMFs, and insert our configured plain SAMFs. We undo the
combined wire permutation and negation state with one unsamf, compress the
resulting circuit, cancel adjacent identical gates, verify its functionality
against the original circuit, and write the round output.

We can repeat this process for several rounds, with each round operating on the
output of the previous one. We stop after the requested number of rounds, or
early if the compressed circuit length remains unchanged for several rounds.
We do not repeat gadgetization, since our later mixing steps do not bring the
circuit back down to low algebraic degree.

With `--single-end`, we carry the SAMF permutation and negation state across
rounds. We compose this state as each round is performed and undo it all
together after the final round, rather than separately after every round.

## 11. End-to-End Summary

```text
Original ECA57 circuit
    |
    v
Random commuting-gate shooting
    |
    v
2n-wire secret-shared gadget circuit
    |
    v
Collision game driven curated expansion
    |
    v
Hidden and plain SAMF insertion
    |
    v
Unsamf / wire-state restoration
    |
    v
Rainbow-table compression
    |
    v
Repeat rounds from collision-game to compression and verify functionality
    |
    v
Obfuscated equivalent circuit
