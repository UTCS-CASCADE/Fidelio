#!/usr/bin/env python3
'Compute CDF distributions for Intel PT disassembled traces.'

import re
import csv
from collections import Counter, defaultdict
from functools import lru_cache
import argparse
from pathlib import Path
import numpy as np
from typing import Tuple, Set, Dict, List, Optional

class Instruction:
    def __init__(self, process, pid, cpu, timestamp, address, function, binary, instruction):
        self.process = process
        self.pid = pid
        self.cpu = cpu
        self.timestamp = timestamp
        self.address = address           # hex string without 0x
        self.function = function
        self.binary = binary
        self.instruction = instruction

def write_percentile_table(prefix: str, data_dict: Dict[str, List[int]], output_dir: Path):
    percentiles = [0.5]

    for key, values in data_dict.items():
        if not values:
            continue

        arr = np.sort(np.asarray(values))
        n = len(arr)
        if n == 0:
            continue

        pct_vals = []
        for p in percentiles:
            idx = int(np.ceil(p * n)) - 1
            idx = max(0, min(idx, n - 1))
            pct_vals.append(arr[idx])

        path = output_dir / f"{prefix}_{key}_percentiles.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Percentile", "Value"])
            for p, v in zip(percentiles, pct_vals):
                w.writerow([p, v])

def parse_line(line) -> Optional[Instruction]:
    if "instruction trace error" in line:
        return None
    line = line.strip()
    if not line:
        return None

    pattern = (
        r"^(\S+)\s+"              # process
        r"(\d+)\s+"               # PID
        r"\[(\d+)\]\s+"           # CPU
        r"([\d\.]+):\s+"          # timestamp
        r"([0-9a-fA-F]+)\s+"      # address (hex, no 0x)
        r"(.+?)\s+\(([^)]+)\)\s+" # function and binary
        r"(.+)$"                  # instruction text
    )
    m = re.match(pattern, line)
    if not m:
        return None

    process, pid, cpu, timestamp, address, function, binary, instruction = m.groups()
    return Instruction(process, pid, cpu, timestamp, address, function.strip(), binary.strip(), instruction.strip())

@lru_cache(maxsize=8192)
def classify_instruction(instr: str) -> str:
    """
    Heuristic categorization into families:
      branch, load, store, move, arith, logic, mem, other
    """
    s = instr.strip()
    if not s:
        return "other"
    op = s.split()[0].lower()

    # Branchy
    if op.startswith("j") or op in {"call", "ret", "jmp"}:
        return "branch"

    # Memory touch detection (parentheses in AT&T syntax)
    touches_mem = "(" in s and ")" in s

    # Specific load/store for mov*
    if op.startswith("mov") and touches_mem:
        # Split on comma to get src,dst
        parts = [p.strip() for p in s.split(",", 1)]
        if len(parts) == 2:
            src, dst = parts
            if "(" in src and "(" not in dst:
                return "load"
            if "(" in dst and "(" not in src:
                return "store"
        # If ambiguous, treat as move
        return "move"

    # Other clear memory ops (simple heuristic)
    if op in {"push", "pop", "stos", "lods", "cmps", "scas"}:
        return "mem"

    if touches_mem:
        return "mem"

    # Moves
    if op.startswith("mov") or op == "lea":
        return "move"

    # Arithmetic
    if op.startswith(("add", "sub", "adc", "sbb", "mul", "imul", "div", "idiv", "inc", "dec", "neg")) or \
       op.startswith(("sar","sal","shl","shr","rol","ror")):
        return "arith"

    # Logic / compare
    if op.startswith(("and","or","xor","cmp","test","not")):
        return "logic"

    return "other"

def parse_trace(filename, *, count_families=True):
    instructions = []
    family_counts = Counter()
    with open(filename, "r") as f:
        for line in f:
            ins = parse_line(line)
            if not ins:
                continue
            instructions.append(ins)
            if count_families:
                family_counts[classify_instruction(ins.instruction)] += 1
    return instructions, family_counts


PAGE_SEQ_LIMIT = 8 

def group_blocks(instructions):
    """
    Groups sequential instructions into basic blocks by VA monotonic small deltas.
    Each block tuple: (start_va_hex, end_va_hex, block_len, lib_basename, pid)
    """
    if not instructions:
        return []

    blocks = []
    curr = instructions[0]
    start_va = curr.address
    end_va = curr.address
    block_len = 1
    curr_lib = curr.binary.split('/')[-1].rstrip(')')
    curr_pid = curr.pid

    for instr in instructions[1:]:
        next_va = int(instr.address, 16)
        curr_va = int(curr.address, 16)
        delta = next_va - curr_va
        # A basic block should only advance forward in the dynamic stream.
        same_flow = (0 < delta <= PAGE_SEQ_LIMIT)
        if same_flow and instr.pid == curr_pid and (instr.binary.split('/')[-1].rstrip(')') == curr_lib):
            # continue current block when pid/lib consistent
            end_va = instr.address
            block_len += 1
        else:
            # close current
            blocks.append((start_va, end_va, block_len, curr_lib, curr_pid))
            # start new block
            start_va = instr.address
            end_va = instr.address
            block_len = 1
            curr_lib = instr.binary.split('/')[-1].rstrip(')')
            curr_pid = instr.pid

        curr = instr

    blocks.append((start_va, end_va, block_len, curr_lib, curr_pid))
    return blocks

def is_conditional_branch(mnemonic: str) -> bool:
    """True if the instruction is a conditional branch (jz, jne, jge and not callq or retq)."""
    return mnemonic.startswith("j") and mnemonic not in {"jmp", "jmpq", "call", "callq", "ret", "retq"}

def classify_branches(instrs):
    """Return list of tuples (VA, taken_flag) for conditional branches."""
    branches = []
    n = len(instrs)
    if n < 2:
        return branches

    for i in range(n - 1):
        ins = instrs[i]
        next_ins = instrs[i + 1]

        # extract VA and mnemonic
        try:
            va = int(ins.address, 16)
        except:
            continue

        # mnemonic = first token of instruction string
        parts = ins.instruction.split()
        if not parts:
            continue
        mnemonic = parts[0].lower()

        if not is_conditional_branch(mnemonic):
            continue

        # next VA
        try:
            next_va = int(next_ins.address, 16)
        except:
            continue

        # heuristic for branch taken/not taken
        taken = abs(next_va - va) > 16
        branches.append((va, taken))

    return branches


def compute_runs(branches):
    """Compute consecutive run lengths for taken and not-taken."""
    if not branches:
        return [], []
    taken_runs, not_taken_runs = [], []
    current_taken = branches[0][1]
    length = 1
    for _, taken in branches[1:]:
        if taken == current_taken:
            length += 1
        else:
            if current_taken:
                taken_runs.append(length)
            else:
                not_taken_runs.append(length)
            current_taken = taken
            length = 1
    # finalize
    if current_taken:
        taken_runs.append(length)
    else:
        not_taken_runs.append(length)
    return taken_runs, not_taken_runs



def compute_jump_sizes_with_blocklens(blocks):
    """
    Only count jumps where next block shares same pid AND same library (avoid cross-process/lib jumps).
    Returns jump_sizes, block_lens, pairs[(block_len, jump_size)]
    """
    jump_sizes, block_lens, pairs = [], [], []
    for i in range(len(blocks) - 1):
        src = blocks[i]
        dst = blocks[i + 1]
        if src[3] != dst[3] or src[4] != dst[4]:
            continue
        src_len = src[2]
        src_end_va = int(src[1], 16)
        dst_start_va = int(dst[0], 16)
        diff = abs(dst_start_va - src_end_va)
        jump_sizes.append(diff)
        block_lens.append(src_len)
        pairs.append((src_len, diff))
    return jump_sizes, block_lens, pairs

def family_counts_per_block(instructions, blocks):
    'For the given instruction list and its blocks, return: - per_block_counts: list[Counter(family->count)] aligned with blocks - family_lists: dict[family] -> list of integer counts (same order as blocks)'
    # Build a fast index of instruction -> family
    fam_by_idx = []
    for ins in instructions:
        fam_by_idx.append(classify_instruction(ins.instruction))

    per_block_counts = []
    lengths = [b[2] for b in blocks]
    start = 0
    for blen in lengths:
        end = start + blen
        c = Counter()
        for i in range(start, end):
            c[fam_by_idx[i]] += 1
        per_block_counts.append(c)
        start = end

    family_lists = defaultdict(list)
    all_fams = sorted({f for c in per_block_counts for f in c})
    for c in per_block_counts:
        for f in all_fams:
            family_lists[f].append(c.get(f, 0))

    return per_block_counts, family_lists




def write_microbenchmark_distributions_csv(mb_label: str, datasets: Dict[str, List[float]], output_dir: Path) -> bool:
    'Write a single CSV with P50 values for one microbenchmark.'
    if not datasets:
        return False

    header_percentiles = ["P50"]
    percentile_values = [50]

    out_path = output_dir / f"{mb_label}_percentiles.csv"
    try:
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            # Header row
            w.writerow(["metric"] + header_percentiles)

            # Write each metric's percentiles
            for metric_name, values in sorted(datasets.items()):
                if not values:
                    freqs = [0.0]
                else:
                    arr = np.asarray(values)
                    try:
                        freqs = np.percentile(arr, percentile_values, method="nearest")
                    except TypeError:
                        # Older NumPy fallback.
                        freqs = np.percentile(arr, percentile_values, interpolation="nearest")

                # Write row: metric_name + P50 value with no decimal places
                w.writerow([metric_name] + [int(round(float(x))) for x in freqs])
        
        return True
    except Exception:
        return False


_alias_groups = [
    ("rax", ["rax","eax","ax","al","ah"]),
    ("rbx", ["rbx","ebx","bx","bl","bh"]),
    ("rcx", ["rcx","ecx","cx","cl","ch"]),
    ("rdx", ["rdx","edx","dx","dl","dh"]),
    ("rsi", ["rsi","esi","si","sil"]),
    ("rdi", ["rdi","edi","di","dil"]),
    ("rbp", ["rbp","ebp","bp","bpl"]),
    ("rsp", ["rsp","esp","sp","spl"]),
    ("rip", ["rip","eip","ip"]),
]
_alias_map: Dict[str,str] = {}
for base, names in _alias_groups:
    for n in names:
        _alias_map[n] = base
for i in range(8,16):
    base = f"r{i}"
    _alias_map[base] = base
    _alias_map[f"e{i}"] = base
    _alias_map[f"{base}d"] = base
    _alias_map[f"{base}w"] = base
    _alias_map[f"{base}b"] = base

def canonical_reg(reg: str) -> Optional[str]:
    'Return architectural register name for RAW tracking. Keeps xmm/ymm/zmm/k as-is (lowercased). Returns None for weird or segment/debug regs we choose to ignore.'
    if not reg:
        return None
    r = reg.lower().lstrip('%')
    # vector/scalar/special we keep:
    if r.startswith(("xmm","ymm","zmm","k")):
        return r
    # gp regs
    if r in _alias_map:
        return _alias_map[r]
    # common segment/base/index regs we generally ignore for RAW (fs/gs)
    if r in {"cs","ds","es","fs","gs","ss"}:
        return None
    # mmx
    if r.startswith("mm"):
        return r
    # mask CR/DR/Txx
    if r.startswith(("cr","dr","tr")):
        return None
    # default: return the thing (conservative)
    return r

_reg_re = re.compile(r"%[a-zA-Z][a-zA-Z0-9]*")

def extract_registers(text: str) -> List[str]:
    return [m.group(0) for m in _reg_re.finditer(text)]

def split_operands(instr_text: str) -> Tuple[str, List[str]]:
    """
    Return opcode and operand list (naive comma split; good enough for AT&T + perf script).
    """
    s = instr_text.strip()
    if not s:
        return "", []
    parts = s.split(None, 1)
    if len(parts) == 1:
        return parts[0].lower(), []
    op = parts[0].lower()
    ops = [o.strip() for o in parts[1].split(",")]
    return op, ops

def dest_is_register(dst: str) -> Optional[str]:
    """
    If the destination operand is a register (not memory), return its canonical name.
    """
    if "(" in dst and ")" in dst:
        return None
    regs = _reg_re.findall(dst)
    if not regs:
        return None
    # If multiple (rare), take the last token as the dest reg
    return canonical_reg(regs[-1])

def regs_in_operand_as_reads(opnd: str) -> Set[str]:
    """
    All registers appearing in an operand are *read* for addressing/source purposes.
    """
    regs = set()
    for r in extract_registers(opnd):
        cr = canonical_reg(r)
        if cr:
            regs.add(cr)
    return regs

def reads_writes_for_instruction(instr_text: str) -> Tuple[Set[str], Set[str]]:
    """
    Return (reads, writes) sets of canonical registers for one instruction in AT&T syntax.
    Heuristic but covers common cases.
    """
    op, ops = split_operands(instr_text)
    op = op.lower()
    reads: Set[str] = set()
    writes: Set[str] = set()

    # zero-operand
    if not ops:
        # treat as no reg read/write (e.g., nop, cpuid handled poorly; okay)
        return reads, writes

    # helpers
    def rw_binop():
        # typical two-operand: src, dst
        for o in ops:
            reads.update(regs_in_operand_as_reads(o))
        d = dest_is_register(ops[-1])
        if d:
            writes.add(d)

    def rw_unop_writes():
        # single-operand operations that read+write the operand
        reads.update(regs_in_operand_as_reads(ops[0]))
        d = dest_is_register(ops[0])
        if d:
            writes.add(d)

    # Categories
    if op.startswith("cmov"):
        # cmovcc src, dst : read src (+flags, ignored), write dst
        for o in ops:
            reads.update(regs_in_operand_as_reads(o))
        d = dest_is_register(ops[-1])
        if d:
            writes.add(d)
        return reads, writes

    if op in {"cmp","test"}:
        for o in ops:
            reads.update(regs_in_operand_as_reads(o))
        return reads, writes

    if op.startswith("mov"):
        # mov src, dst
        for o in ops:
            reads.update(regs_in_operand_as_reads(o))
        d = dest_is_register(ops[-1])
        if d:
            writes.add(d)
        return reads, writes

    if op == "lea":
        # lea src, dst : reads addressing regs in src, writes dst
        reads.update(regs_in_operand_as_reads(ops[0]))
        d = dest_is_register(ops[-1])
        if d:
            writes.add(d)
        return reads, writes

    if op in {"add","sub","adc","sbb","and","or","xor","xadd"} or \
       op in {"shl","shr","sal","sar","rol","ror"} or \
       op in {"imul"}:
        rw_binop()
        return reads, writes

    if op in {"inc","dec","neg","not"}:
        rw_unop_writes()
        return reads, writes

    if op == "push":
        # reads operand (+rsp), writes rsp
        reads.update(regs_in_operand_as_reads(ops[0]))
        reads.add("rsp")
        writes.add("rsp")
        return reads, writes

    if op == "pop":
        # reads rsp, writes dst (+rsp)
        reads.add("rsp")
        d = dest_is_register(ops[0])
        if d:
            writes.add(d)
        writes.add("rsp")
        return reads, writes

    if op in {"call","callq"}:
        # possible indirect register read; rsp read+write
        for o in ops:
            reads.update(regs_in_operand_as_reads(o))
        reads.add("rsp")
        writes.add("rsp")
        # rip write ignored
        return reads, writes

    if op in {"ret","retq"}:
        reads.add("rsp")
        writes.add("rsp")
        return reads, writes

    if op.startswith("j"):  # jmp/jcc
        # handle jmp *%reg as read
        for o in ops:
            reads.update(regs_in_operand_as_reads(o))
        return reads, writes

    # default: conservative two-operand attempt or read-only
    if len(ops) == 2:
        rw_binop()
    else:
        for o in ops:
            reads.update(regs_in_operand_as_reads(o))
    return reads, writes

def raw_dependency_distances(instructions: List[Instruction]) -> List[int]:
    'Compute RAW distances (in number of intervening dynamic instructions) per trace stream. Distance for a read of register R is i - last_write[R], if a previous write to R exists. Resets tracking across PID/binary changes.'
    dists: List[int] = []
    last_write: Dict[str, int] = {}
    prev_pid = None
    prev_lib = None

    decode = lru_cache(maxsize=8192)(reads_writes_for_instruction)
    for i, ins in enumerate(instructions):
        # Reset when context changes (avoid cross-PID/lib linking)
        curr_pid = ins.pid
        curr_lib = ins.binary
        if prev_pid is None:
            prev_pid, prev_lib = curr_pid, curr_lib
        elif curr_pid != prev_pid or curr_lib != prev_lib:
            last_write.clear()
            prev_pid, prev_lib = curr_pid, curr_lib

        reads, writes = decode(ins.instruction)

        # record distances for each read that has a prior write
        for r in reads:
            if r in last_write:
                d = i - last_write[r]
                if d > 0:
                    dists.append(d)

        # update last write indices
        for w in writes:
            last_write[w] = i

    return dists


def write_raw_csv(raw_dict: Dict[str, List[int]], output_dir: Path):
    for label, dists in raw_dict.items():
        path = output_dir / f"raw_dep_distances_{label}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Index", "RAW_Distance"])
            for i, d in enumerate(dists):
                w.writerow([i, d])

def write_cdf_percentiles_csv(jump_dict: Dict[str, List[int]], output_dir: Path):
    'For each application/trace, compute CDF P50 using the same values plotted in the CDF curves. Writes one CSV per app.'
    percentiles = [0.5]

    for label, jumps in jump_dict.items():
        if not jumps:
            continue

        data = np.sort(np.asarray(jumps))
        n = len(data)
        if n == 0:
            continue

        # compute percentile value (P50)
        pct_vals = []
        for p in percentiles:
            # percentile index using CDF convention (i/n)
            idx = int(np.ceil(p * n)) - 1
            idx = max(0, min(idx, n - 1))
            pct_vals.append(data[idx])

        out = output_dir / f"{label}_jump_cdf_percentiles.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Percentile", "Value"])
            for p, v in zip(percentiles, pct_vals):
                w.writerow([p, v])

def main():
    ap = argparse.ArgumentParser(description="Compute trace statistics and generate P50 CSV per microbenchmark")
    ap.add_argument("-i", "--inputPath", required=True, help="Parent directory containing binary subdirectories")
    ap.add_argument("-o", "--outputPath", required=True, help="Output directory for CSV files")
    args = ap.parse_args()

    input_dir = Path(args.inputPath)
    output_dir = Path(args.outputPath)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}")
        return

    # Get all binary subdirectories
    binary_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    
    if not binary_dirs:
        print(f"No subdirectories found in {input_dir}")
        return

    total_processed = 0
    total_success = 0
    
    # Process each binary directory
    for binary_dir in binary_dirs:
        # Find all disassembled .data files (those with _dis.data suffix)
        dis_files = sorted([f for f in binary_dir.glob("*_dis.data") if f.is_file()])
        
        if not dis_files:
            continue
        
        print(f"\nProcessing binary: {binary_dir.name}")
        
        for dis_file in dis_files:
            total_processed += 1
            
            filename_stem = dis_file.stem  # removes .data
            # filename_stem is like "container_hash_maps_bench_%Clear    unord_NonSSOString..._pt_dis"
            # We need to extract just the unique microbenchmark identifier
            
            # Simple approach: use the full stem as the label
            mb_label = filename_stem.replace("_pt_dis", "")
            
            # Parse the trace file
            try:
                instrs, _fam_counts_total = parse_trace(dis_file)
                blocks = group_blocks(instrs)
                per_block_counts, family_lists = family_counts_per_block(instrs, blocks)
            except Exception as e:
                print(f"  [FAILED] {dis_file.name}: {type(e).__name__}")
                continue
            
            if not instrs or not blocks:
                print(f"  [FAILED] {dis_file.name}: no instructions or blocks")
                continue
            
            # Compute all statistics
            jumps, blens, pairs = compute_jump_sizes_with_blocklens(blocks)
            branches = classify_branches(instrs)
            taken_runs, not_taken_runs = compute_runs(branches)
            raw_dists = raw_dependency_distances(instrs)
            
            # Build combined dataset for this microbenchmark
            mb_datasets: Dict[str, List[float]] = {}
            
            # Block sizes
            mb_datasets["block_sizes"] = [b[2] for b in blocks]
            
            # Jump sizes
            mb_datasets["jumps"] = jumps
            
            # Branch run lengths
            mb_datasets["taken_runs"] = taken_runs if taken_runs else [0]
            mb_datasets["not_taken_runs"] = not_taken_runs if not_taken_runs else [0]
            
            # RAW dependency distances
            mb_datasets["raw_dependencies"] = raw_dists if raw_dists else [0]
            
            # Instruction family counts per block
            for fam, counts in family_lists.items():
                mb_datasets[f"family::{fam}"] = counts
            
            # Write the combined CSV for this microbenchmark
            success = write_microbenchmark_distributions_csv(mb_label, mb_datasets, output_dir)
            
            if success:
                print(f"  [SUCCESS] {mb_label}: {len(instrs)} instrs, {len(blocks)} blocks, {len(jumps)} jumps, {len(raw_dists)} RAW deps")
                total_success += 1
            else:
                print(f"  [FAILED] {mb_label}: could not write CSV")
    
    print("\n=== Summary ===")
    print(f"Total processed: {total_processed}")
    print(f"Successful: {total_success}")
    print(f"Failed: {total_processed - total_success}")


if __name__ == "__main__":
    main()
