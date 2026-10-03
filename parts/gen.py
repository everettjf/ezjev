# ===== 代码生成的题：答案由程序算出来（对应 CRUXEval、GSM8K、BBH、CLadder、SATA；这些 benchmark 没有可用的 train split）=====
# 题型格式照抄评测请求，内容全部随机生成，不含任何评测题文本。
import datetime as _dt, math

GR = random.Random(SEED + 1)
NAMES = ["Alice", "Bob", "Claire", "Dave", "Eve", "Fred", "Gertrude", "Hana", "Ivan", "Jun", "Kofi", "Lena", "Mateo", "Nia",
         "Omar", "Priya", "Quinn", "Rosa", "Sven", "Tara", "Uma", "Viktor", "Wen", "Yara", "Zane"]

def opt_keys(opts, style=None):
    style = style or GR.choice(["paren", "option", "letter"])
    if style == "paren":
        return [f"({chr(65 + i)})" for i in range(len(opts))]
    if style == "option":
        return [f"option_{i}" for i in range(len(opts))]
    return [chr(65 + i) for i in range(len(opts))]

def bbh_mc(src, text, opts, gold_idx):
    """BBH 风格：题干末尾带 Options 块，key 是 (A)/(B)…，描述是选项原文。"""
    order = list(range(len(opts))); GR.shuffle(order)
    opts = [opts[i] for i in order]; g = order.index(gold_idx)
    keys = [f"({chr(65 + i)})" for i in range(len(opts))]
    text = text + "\nOptions:\n" + "\n".join(f"{k} {o}" for k, o in zip(keys, opts))
    return sample(src, text, choice("Which option is the correct answer?", dict(zip(keys, opts))), keys[g])

def bbh_bin(src, text, yes, labels=("Yes", "No")):
    return sample(src, text, choice("Which option is the correct answer?", {labels[0]: labels[0], labels[1]: labels[1]}),
                  labels[0] if yes else labels[1])

# ---------- BBH 风格 ----------
def g_boolean():
    def expr(d):
        if d == 0 or GR.random() < 0.3:
            v = GR.choice([True, False]); return str(v), v
        r = GR.random()
        if r < 0.25:
            s, v = expr(d - 1); return f"not {s}", not v
        a, va = expr(d - 1); b, vb = expr(d - 1)
        if r < 0.5:
            return f"( {a} )", va
        op = GR.choice(["and", "or"])
        return f"{a} {op} {b}", (va and vb) if op == "and" else (va or vb)
    s, _ = expr(3)
    v = eval(s)
    return bbh_bin("gen_bbh", f"{s} is", v, ("True", "False"))

def g_web_of_lies():
    ppl = GR.sample(NAMES, GR.randint(4, 6))
    truth = GR.choice([True, False])
    parts = [f"{ppl[0]} {'tells the truth' if truth else 'lies'}."]
    for p, q in zip(ppl[1:], ppl):
        says = GR.choice([True, False])
        parts.append(f"{p} says {q} {'tells the truth' if says else 'lies'}.")
        truth = (says == truth)
    return bbh_bin("gen_bbh", "Question: " + " ".join(parts) + f" Does {ppl[-1]} tell the truth?", truth)

def g_navigate():
    x = y = 0; steps = []
    for _ in range(GR.randint(3, 8)):
        d, k = GR.choice(["forward", "backward", "left", "right"]), GR.randint(1, 10)
        steps.append(f"Take {k} step{'s' if k > 1 else ''} {d}.")
        x += {"left": -k, "right": k}.get(d, 0); y += {"forward": k, "backward": -k}.get(d, 0)
    if GR.random() < 0.4:  # 让一部分题真的回到原点
        if x: steps.append(f"Take {abs(x)} step{'s' if abs(x) > 1 else ''} {'left' if x > 0 else 'right'}.")
        if y: steps.append(f"Take {abs(y)} step{'s' if abs(y) > 1 else ''} {'backward' if y > 0 else 'forward'}.")
        x = y = 0
    text = ("If you follow these instructions, do you return to the starting point? Always face forward. " + " ".join(steps)
            + "\nOptions:\n- Yes\n- No")
    return bbh_bin("gen_bbh", text, x == 0 and y == 0)

ITEMS = {"books": ["Ulysses", "Frankenstein", "Lolita", "Moby Dick", "Emma", "Dracula", "Hamlet"],
         "balls": ["red ball", "blue ball", "green ball", "yellow ball", "pink ball", "black ball", "white ball"],
         "positions": ["striker", "goalkeeper", "left winger", "right winger", "benchwarmer", "center midfielder", "cheerleader"],
         "partners": ["Patrick", "Sam", "Jamie", "Lola", "Melissa", "Ophelia", "Karl"]}

def g_tracking():
    k = GR.choice([3, 5, 7]); ppl = NAMES[:k]
    kind = GR.choice(list(ITEMS)); things = GR.sample(ITEMS[kind], k)
    hold = dict(zip(ppl, things))
    swaps = []
    for _ in range(k):
        a, b = GR.sample(ppl, 2); hold[a], hold[b] = hold[b], hold[a]; swaps.append(f"{a} and {b} swap")
    who = GR.choice(ppl)
    text = (f"{', '.join(ppl[:-1])}, and {ppl[-1]} each start with one item: "
            + ", ".join(f"{p} has the {t}" for p, t in zip(ppl, things)) + ". Then " + ". Then ".join(swaps)
            + f". At the end, {who} has the")
    return bbh_mc("gen_bbh", text, things, things.index(hold[who]))

ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]

def g_logical_deduction():
    k = GR.choice([3, 5, 7]); objs = GR.sample(NAMES, k); order = objs[:]; GR.shuffle(order)  # order[0] = 最左/第一
    facts = set()
    while True:  # 加约束直到排列唯一
        r = GR.random()
        if r < 0.4:
            i = GR.randrange(k); facts.add(f"{order[i]} finished {ORD[i]}.")
        else:
            i, j = sorted(GR.sample(range(k), 2)); facts.add(f"{order[i]} finished above {order[j]}.")
        sols = 0
        for p in itertools.permutations(objs):
            ok = all((f"{p[int(ORD.index(f.split()[-1][:-1]))]}" == f.split()[0]) if " finished " in f and "above" not in f
                     else p.index(f.split()[0]) < p.index(f.split()[-1][:-1]) for f in facts)
            sols += ok
            if sols > 1:
                break
        if sols == 1:
            break
    pos = GR.randrange(k)
    text = (f"The following paragraphs each describe a set of {k} objects arranged in a fixed order. In a race, there were {k} "
            f"runners: {', '.join(objs)}. " + " ".join(sorted(facts, key=lambda _: GR.random())))
    opts = [f"{o} finished {ORD[pos]}" for o in objs]
    return bbh_mc("gen_bbh", text, opts, objs.index(order[pos]))

def g_date():
    d = _dt.date(GR.randint(1900, 2030), GR.randint(1, 12), GR.randint(1, 28))
    shift, phrase = GR.choice([(1, "tomorrow"), (-1, "yesterday"), (7, "one week from today"), (-7, "one week ago"),
                               (10, "10 days from today"), (-30, "30 days ago"), (365, "one year from today")])
    ans = d + _dt.timedelta(days=shift)
    fmt = lambda x: x.strftime("%m/%d/%Y")
    wrong = {fmt(ans + _dt.timedelta(days=GR.choice([-3, -2, -1, 1, 2, 3, 30, -30]))) for _ in range(8)} - {fmt(ans)}
    opts = [fmt(ans)] + list(wrong)[:GR.randint(3, 5)]
    return bbh_mc("gen_bbh", f"Today is {fmt(d)}. What is the date {phrase} in MM/DD/YYYY?", opts, 0)

COLORS = ["red", "blue", "green", "grey", "purple", "mauve", "teal", "orange", "black", "gold"]
THINGS = ["notebook", "pen", "cat toy", "mug", "keychain", "sheet of paper", "stress ball", "fidget spinner"]
NUMW = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]

def g_colored():
    items = [(GR.randint(1, 3), GR.choice(COLORS), GR.choice(THINGS)) for _ in range(GR.randint(3, 6))]
    c = GR.choice([it[1] for it in items]); rm = GR.choice([it[2] for it in items])
    left = sum(n for n, col, t in items if col == c and t != rm)
    desc = ", ".join(f"{NUMW[n]} {col} {t}{'s' if n > 1 else ''}" for n, col, t in items)
    text = f"On the desk, there are {desc}. If I remove all the {rm}s from the desk, how many {c} things remain on it?"
    return bbh_mc("gen_bbh", text, NUMW[:13], left) if left < 13 else g_colored()

ADJ = [("opinion", ["lovely", "ugly", "nice", "terrible", "wonderful"]), ("size", ["big", "small", "tiny", "enormous", "midsize"]),
       ("age", ["old", "new", "ancient", "brand-new"]), ("shape", ["square", "circular", "triangular", "rectangular"]),
       ("color", ["red", "grey", "blue", "green", "white"]), ("origin", ["Brazilian", "Japanese", "German", "Mexican", "Indian"]),
       ("material", ["wool", "plastic", "glass", "leather", "steel"]), ("purpose", ["hiking", "drinking", "smoking", "exercise"])]

def g_hyperbaton():
    cats = sorted(GR.sample(range(len(ADJ)), GR.randint(3, 4)))
    words = [GR.choice(ADJ[c][1]) for c in cats]
    noun = GR.choice(["sweater", "knife", "ship", "shoe", "car", "box"])
    bad = words[:]
    while bad == words:
        GR.shuffle(bad)
    return bbh_mc("gen_bbh", "Which sentence has the correct adjective order:", [" ".join(words + [noun]), " ".join(bad + [noun])], 0)

def g_shape():
    k = GR.choice([3, 4, 5, 6, 7, 8])
    name = {3: "triangle", 4: "rectangle", 5: "pentagon", 6: "hexagon", 7: "heptagon", 8: "octagon"}[k]
    cx, cy, r = GR.uniform(30, 70), GR.uniform(30, 70), GR.uniform(10, 25)
    pts = []
    for i in range(k):
        a = 2 * math.pi * i / k + GR.uniform(-0.2, 0.2)
        rr = r if k != 4 else r
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    if k == 4:
        w, h = GR.uniform(10, 30), GR.uniform(10, 30); pts = [(cx, cy), (cx + w, cy), (cx + w, cy + h), (cx, cy + h)]
    d = f"M {pts[0][0]:.2f},{pts[0][1]:.2f} " + " ".join(f"L {x:.2f},{y:.2f}" for x, y in pts[1:]) + f" L {pts[0][0]:.2f},{pts[0][1]:.2f}"
    opts = ["circle", "heptagon", "hexagon", "kite", "line", "octagon", "pentagon", "rectangle", "sector", "triangle"]
    text = f'This SVG path element <path d="{d}"/> draws a'
    keys = [f"({chr(65 + i)})" for i in range(len(opts))]
    text += "\nOptions:\n" + "\n".join(f"{kk} {o}" for kk, o in zip(keys, opts))
    return sample("gen_bbh", text, choice("Which option is the correct answer?", dict(zip(keys, opts))), keys[opts.index(name)])

def g_temporal():
    who = GR.choice(NAMES); place = GR.choice(["the bakery", "the gym", "the library", "the museum", "the park"])
    hrs = sorted(GR.sample(range(6, 21), 5))
    free = GR.randrange(4)
    lines, slots = [f"{who} woke up at {hrs[0]}:00."], []
    for i in range(4):
        slot = f"{hrs[i]}:00 to {hrs[i + 1]}:00"; slots.append(slot)
        if i != free:
            lines.append(f"{GR.choice(NAMES)} saw {who} {GR.choice(['reading', 'shopping', 'jogging', 'eating lunch', 'working'])} from {slot}.")
    lines.append(f"{place.capitalize()} was closed after {hrs[4]}:00.")
    text = f"Today, {who} went to {place}. Between what times could they have gone?\nWe know that:\n" + "\n".join(lines)
    return bbh_mc("gen_bbh", text, slots, free)

def bbh_like(n):
    fs = [g_boolean, g_web_of_lies, g_navigate, g_tracking, g_logical_deduction, g_date, g_colored, g_hyperbaton, g_shape, g_temporal]
    return [GR.choice(fs)() for _ in range(n)]

# ---------- CRUXEval 风格：给代码和输入，选正确输出 ----------
CRUX = [
    "def f(s):\n    return s[::{a}]",
    "def f(nums):\n    return [x * {a} for x in nums if x % {b} == 0]",
    "def f(s):\n    return s.replace('{c1}', '{c2}').upper()",
    "def f(nums):\n    out = []\n    for n in nums:\n        out.append((nums.count(n), n))\n    out.sort(reverse=True)\n    return out",
    "def f(text):\n    return ''.join(ch for ch in text if ch not in '{c1}{c2}')",
    "def f(nums):\n    nums = nums[:]\n    nums.insert({a}, {b})\n    return nums",
    "def f(d):\n    return sorted(d.items(), key=lambda kv: kv[1])[:{a}]",
    "def f(s):\n    return s.count('{c1}') + len(s.split('{c2}'))",
    "def f(nums):\n    total = 0\n    for i, n in enumerate(nums):\n        if i % {a} == 0:\n            total += n\n    return total",
    "def f(s):\n    words = s.split()\n    return ' '.join(w[::-1] if len(w) > {a} else w for w in words)",
    "def f(nums):\n    return sorted(set(nums), reverse={bool})[{a}:]",
    "def f(s):\n    return s.center(len(s) + {a}, '{c1}')",
    "def f(nums):\n    while len(nums) > {a}:\n        nums = nums[1:] + nums[:1]\n        nums.pop()\n    return nums",
    "def f(s):\n    d = {{}}\n    for ch in s:\n        d[ch] = d.get(ch, 0) + 1\n    return max(d, key=d.get)",
    "def f(a, b):\n    return a[{a}:] + b[:{b}]",
    "def f(nums):\n    return [n for n in nums[::-1] if n > {b}]",
]

def rand_input(code):
    if "(a, b)" in code:
        return (GR.choice(["abcdef", "hello", "xyzzy", "python"]), GR.choice(["12345", "world", "qwerty"]))
    if "(d)" in code:
        return ({k: GR.randint(0, 9) for k in GR.sample("abcdefg", GR.randint(2, 5))},)
    if "nums" in code:
        return ([GR.randint(-3, 12) for _ in range(GR.randint(3, 8))],)
    return (GR.choice(["banana split", "hello world", "abracadabra", "mississippi", "the quick brown fox", "aabbccdd"]),)

def crux_like(n):
    out = []
    while len(out) < n:
        t = GR.choice(CRUX)
        code = t.format(a=GR.randint(1, 3), b=GR.randint(1, 4), c1=GR.choice("abeilos"), c2=GR.choice("xyz-"), bool=GR.choice([True, False]))
        args = rand_input(code)
        ns = {}
        try:
            exec(code, ns)
            val = ns["f"](*[a.copy() if hasattr(a, "copy") else a for a in args])
            res = repr(val)
        except Exception:
            continue
        wrong = set()
        for _ in range(12):  # 干扰项：同一模板换参数后的输出、或对正确输出做小改动
            alt = t.format(a=GR.randint(1, 3), b=GR.randint(1, 4), c1=GR.choice("abeilos"), c2=GR.choice("xyz-"), bool=GR.choice([True, False]))
            try:
                ns2 = {}; exec(alt, ns2); w = repr(ns2["f"](*[a.copy() if hasattr(a, "copy") else a for a in args]))
            except Exception:
                continue
            if w != res:
                wrong.add(w)
        if isinstance(val, (str, list)) and len(val) > 1:  # 对正确输出本身做小改动（不是对它的 repr 字符串）
            wrong |= {repr(v) for v in (val[::-1], val[1:], val[:-1], val + val[-1:])}
        elif isinstance(val, int) and not isinstance(val, bool):
            wrong |= {repr(val + d) for d in (-2, -1, 1, 2)}
        wrong = list(wrong - {res})
        if len(wrong) < 1:
            continue
        opts = [res] + GR.sample(wrong, min(len(wrong), GR.randint(1, 3)))
        state = {"code": code, "input": ", ".join(repr(a) for a in args), "task": "CRUXEval output selection"}
        out.append(mc("gen_crux", state, "Choose the correct output of f called with the supplied input arguments. "
                                         "Candidates are Python literal values.", opts, 0))
    return out

# ---------- GSM8K 风格：数值选择题 ----------
def num_opts(ans, k):
    fmt = lambda v: str(int(v)) if float(v).is_integer() else f"{v:.2f}".rstrip("0").rstrip(".")
    cands = [ans + d for d in (-10, -6, -5, -3, -2, -1, 1, 2, 3, 5, 6, 10)] + [ans * 2, ans / 2, ans * 10, ans + ans / 2]
    GR.shuffle(cands)
    opts, seen = [fmt(ans)], {fmt(ans)}
    for c in cands:
        if c >= 0 and fmt(c) not in seen and len(opts) < k:
            opts.append(fmt(c)); seen.add(fmt(c))
    return opts

def gsm_row(src, question, ans):
    k = GR.choice([4, 10])
    opts = num_opts(ans, k)
    if len(opts) < k:
        return None
    return mc(src, {"question": question, "task": f"GSM8K deterministic {k}-choice numeric selection"},
              "Choose the numeric answer to the problem in state. Do not provide reasoning. This is a named multiple-choice adaptation of GSM8K.",
              opts, 0)

WORD_PROBLEMS = [
    (lambda a, b, c: (f"{{n}} buys {a} boxes of pencils with {b} pencils in each box and gives away {c} pencils. How many pencils does {{n}} have left?", a * b - c)),
    (lambda a, b, c: (f"A shop sells apples for ${a} each and pears for ${b} each. {{n}} buys {c} apples and {c + 2} pears. How much does {{n}} spend in dollars?", a * c + b * (c + 2))),
    (lambda a, b, c: (f"{{n}} runs {a} km every weekday and {b} km on each weekend day. How many km does {{n}} run in {c} weeks?", (5 * a + 2 * b) * c)),
    (lambda a, b, c: (f"A tank holds {a * 10} liters. It is filled at {b} liters per minute and leaks {1 if b > 1 else 0} liter per minute. How many minutes until it is full, rounded down?", (a * 10) // max(1, b - (1 if b > 1 else 0)))),
    (lambda a, b, c: (f"{{n}} earns ${a * 5} per hour and works {b} hours a day for {c} days, then spends half of it. How many dollars are left?", a * 5 * b * c / 2)),
    (lambda a, b, c: (f"There are {a * 4} students. A quarter of them play chess and {b} of the rest play soccer. How many students play neither?", a * 4 - a - b) if a * 3 >= b else (f"There are {a + b} cats and {c} dogs. How many legs do they have in total?", 4 * (a + b + c))),
]

def gsm_like(n):
    out = []
    while len(out) < n:
        a, b, c = GR.randint(2, 12), GR.randint(2, 12), GR.randint(1, 9)
        q, ans = GR.choice(WORD_PROBLEMS)(a, b, c)
        r = gsm_row("gen_gsm", q.replace("{n}", GR.choice(NAMES)), float(ans))
        if r:
            out.append(r)
    return out

def gsm8k_fmt(n):
    """GSM8K train 按评测题的格式（state 里放题目、option_i 选项）。"""
    out = []
    for ex in stream("openai/gsm8k", "main", n=n):
        try:
            a = float(ex["answer"].split("####")[-1].strip().replace(",", ""))
        except ValueError:
            continue
        r = gsm_row("gsm8k_fmt", ex["question"], a)
        if r:
            out.append(r)
    return out

# ---------- CLadder 风格：因果概率 yes/no ----------
VARS = [("husband", "alarm set by husband", "ringing alarm"), ("smoking", "smoking", "lung cancer"), ("vaccination", "vaccination", "recovery"),
        ("tutoring", "tutoring", "passing the exam"), ("rain", "rain", "traffic jam"), ("xevo", "xevo", "gyzp"), ("zuph", "zuph", "rixq")]

def cladder_like(n):
    out = []
    for _ in range(n):
        xn, xdesc, y = GR.choice(VARS)
        p0, p1 = GR.randint(5, 95), GR.randint(5, 95)
        if p0 == p1:
            continue
        if GR.random() < 0.5:  # 无混杂：直接比较
            world = f"{xn.capitalize()} has a direct effect on {y}."
            info = (f"For those with no {xdesc}, the probability of {y} is {p0}%. For those with {xdesc}, the probability of {y} is {p1}%.")
            ate = p1 - p0
        else:  # 有混杂 Z：后门调整
            pz = GR.randint(10, 90); q = [GR.randint(5, 95) for _ in range(4)]  # P(Y|X=x,Z=z): x0z0, x0z1, x1z0, x1z1
            world = f"Confounder has a direct effect on {xn} and {y}. {xn.capitalize()} has a direct effect on {y}."
            info = (f"The overall probability of confounder is {pz}%. For those with no confounder and no {xdesc}, the probability of {y} is {q[0]}%. "
                    f"For those with confounder and no {xdesc}, {q[1]}%. For those with no confounder and {xdesc}, {q[2]}%. "
                    f"For those with confounder and {xdesc}, {q[3]}%.")
            ate = (q[2] * (100 - pz) + q[3] * pz) - (q[0] * (100 - pz) + q[1] * pz)
            if ate == 0:
                continue
        inc = GR.random() < 0.5
        ask = f"Will {xdesc} {'increase' if inc else 'decrease'} the chance of {y}?"
        text = ("Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors "
                f"or causal relationships: {world}\n\n{info}\n\n{ask}")
        yes = (ate > 0) == inc
        out.append(sample("gen_cladder", {}, choice(text, {"A": "yes", "B": "no"}), "A" if yes else "B"))
    return out

# ---------- SATA 风格：每个候选单独判断对不对 ----------
def sata_like(n):
    """SciQ train：带支撑段落的选择题拆成每个选项一个 yes/no；另加 "哪些数是质数" 这类多答案题。"""
    out = []
    for ex in stream("allenai/sciq", n=n):
        if not ex["support"]:
            continue
        cands = [ex["correct_answer"], ex["distractor1"], ex["distractor2"], ex["distractor3"]]; GR.shuffle(cands)
        qs, gold = {}, {}
        for i, c in enumerate(cands):
            qs[f"option_{i}"] = choice(f"Question: {ex['question']}\nDoes this candidate correctly answer the question?\nCandidate: {c}",
                                       {"no": "No", "yes": "Yes"})
            gold[f"option_{i}"] = "yes" if c == ex["correct_answer"] else "no"
        out.append(multi("sata_sciq", {"paragraph": ex["support"]}, qs, gold))
    tests = [("prime", lambda v: v > 1 and all(v % d for d in range(2, int(v ** 0.5) + 1))), ("even", lambda v: v % 2 == 0),
             ("a perfect square", lambda v: int(v ** 0.5) ** 2 == v), ("divisible by 3", lambda v: v % 3 == 0)]
    for _ in range(n // 3):
        name, fn = GR.choice(tests); nums = GR.sample(range(2, 100), GR.randint(5, 9))
        q = f"Which of the following numbers are {name}?"
        qs = {f"option_{i}": choice(f"Question: {q}\nDoes this candidate correctly answer the question?\nCandidate: {v}", {"no": "No", "yes": "Yes"})
              for i, v in enumerate(nums)}
        out.append(multi("sata_numbers", {"paragraph": f"Consider the numbers {', '.join(map(str, nums))}."}, qs,
                         {f"option_{i}": "yes" if fn(v) else "no" for i, v in enumerate(nums)}))
    return out

SOURCES.update({"gen_bbh": bbh_like, "gen_crux": crux_like, "gen_gsm": gsm_like, "gsm8k_fmt": gsm8k_fmt,
                "gen_cladder": cladder_like, "sata": sata_like})
