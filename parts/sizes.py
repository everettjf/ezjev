# ===== 每个数据源抽多少题（再乘以 DATA_SCALE）=====
SIZES = {
    # 语言理解 / NLI（对应 ANLI、ContractNLI、NLI4CT、VAST、RAGTruth）
    "mnli": 3000, "snli": 1000, "anli": 4500, "boolq": 3000,
    # 检索与分类（BANKING77、CLINC150、Amazon ESCI、PhishNChips……）
    "banking77": 3500, "clinc": 4000, "massive": 2500,
    "ag_news": 1000, "dbpedia": 1000, "yahoo": 1500, "emotion": 1500, "sst5": 1500, "irony": 1500,
    # 知识与推理（MMLU-Pro、GPQA、BBH、GSM8K、WinoGrande、HellaSwag……）
    "winogrande": 3000, "hellaswag": 3000, "mmlu_aux": 5000, "arc": 2000, "csqa": 1500, "obqa": 1000,
    "qasc": 1000, "sciq": 1500, "aqua": 1500, "gsm8k": 2500,
    # 多段落检索（BRIGHT、ToolRet、HoVer）
    "hotpot": 3000,
    # 工具调用（BFCL、API-Bank、When2Call）
    "tools": 3500,
    # 人类偏好（Arts & Human Judgment）
    "shp": 2000, "hh": 1000,
}

# ===== 第二版：补上 Decision Index 各 benchmark 的题型（parts/data_v2.py、parts/gen.py）=====
# 不在榜单里的分类数据（ag_news、yahoo、emotion、sst5 等）减量；GSM8K 换成评测题的格式。
SIZES_V2 = {
    **SIZES,
    "ag_news": 400, "dbpedia": 400, "yahoo": 500, "emotion": 500, "sst5": 500, "massive": 1500, "snli": 800, "mnli": 2000,
    "tools": 1000, "hotpot": 2000, "gsm8k": 0, "aqua": 800, "qasc": 600, "obqa": 600,
    # 语言理解：ContractNLI、VAST、NLI4CT、RAGTruth、iSarcasmEval、ACOS
    "contractnli": 450, "vast": 2500, "nli4ct": 1500, "ragtruth": 2500, "isarcasm": 2500, "acos": 1200, "sharc": 1200,
    # 检索与路由：ESCI、ToolRet/BRIGHT（相关性）、SGD
    "esci": 3000, "qnli": 1800, "sgd": 2500,
    # 工具：BFCL、API-Bank、ToolRet、When2Call
    "tool_rows": 4500, "when2call": 3000,
    # 人文：Humicroedit、New Yorker
    "humicroedit": 2000, "newyorker": 2000,
    # 代码生成的题：BBH、CRUXEval、GSM8K、CLadder、SATA
    "gen_bbh": 4000, "gen_crux": 2500, "gen_gsm": 1200, "gsm8k_fmt": 2500, "gen_cladder": 2000, "sata": 1200,
}
SIZES_V2 = {k: v for k, v in SIZES_V2.items() if v > 0}
