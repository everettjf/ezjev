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
