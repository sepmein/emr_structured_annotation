# 提示词词典：将 XML 标签值映射到 GLiNER 自然语言提示词
# 提示词越精确，模型召回率越高
# 格式：{ "XML标签值": ("提示词", "from_name") }
# from_name 对应 XML 里的 Labels name 属性

LABEL_PROMPTS: dict[str, tuple[str, str]] = {

    # ── 症状 (symptons_labels) ──────────────────────────────────────
    "发热": (
        "发热症状，包括体温升高、发烧、高热、低热、热退、体温异常等描述",
        "symptons_labels",
    ),
    "气促": (
        "呼吸急促、气促、呼吸增快、呼吸频率加快等描述",
        "symptons_labels",
    ),
    "鼻翼煽动": (
        "鼻翼煽动、鼻翼扇动，呼吸困难时鼻孔随呼吸扩张的体征",
        "symptons_labels",
    ),
    "三凹征": (
        "三凹征，吸气时胸骨上窝、锁骨上窝、肋间隙凹陷的呼吸困难体征",
        "symptons_labels",
    ),
    "喘鸣或喘息": (
        "喘鸣、喘息、哮鸣音、呼吸时有哨音或喘声等描述",
        "symptons_labels",
    ),
    "意识障碍": (
        "意识障碍、昏迷、嗜睡、谵妄、烦躁、定向力障碍、神志不清等描述",
        "symptons_labels",
    ),
    "惊厥": (
        "惊厥、抽搐、癫痫发作、抽风等描述",
        "symptons_labels",
    ),
    "拒食或喂养困难": (
        "拒食、拒奶、喂养困难、不吃奶、进食差等描述（多见于婴幼儿）",
        "symptons_labels",
    ),
    "呼吸衰竭": (
        "呼吸衰竭、I型呼吸衰竭、II型呼吸衰竭、呼衰、低氧血症合并呼吸功能不全",
        "symptons_labels",
    ),
    "休克": (
        "休克、感染性休克、脓毒性休克、血压下降伴循环衰竭等描述",
        "symptons_labels",
    ),
    "器官衰竭": (
        "器官衰竭、多器官功能障碍、MODS、肝衰竭、肾衰竭等描述",
        "symptons_labels",
    ),

    # ── 诊断 (diagnosis_labels) ────────────────────────────────────
    "肺实质异常影": (
        "肺实质异常影像，包括肺部阴影、渗出影、实变影、磨玻璃影等影像学描述",
        "diagnosis_labels",
    ),
    "肺组织炎症": (
        "肺组织炎症、肺炎、肺部炎性改变、肺间质炎症等描述",
        "diagnosis_labels",
    ),
    "肺炎诊断": (
        "肺炎诊断，包括社区获得性肺炎、医院获得性肺炎、CAP、HAP、肺炎确诊等",
        "diagnosis_labels",
    ),
    "肺部感染": (
        "肺部感染、下呼吸道感染、肺部细菌感染、肺部病毒感染等描述",
        "diagnosis_labels",
    ),

    # ── 治疗 (treatment_labels) ────────────────────────────────────
    "机械通气": (
        "机械通气、有创通气、无创通气、气管插管、呼吸机辅助通气、CPAP、BiPAP等",
        "treatment_labels",
    ),

    # ── 流行病学关联 (epidemics_labels) ───────────────────────────
    "医务人员": (
        "医务人员、医生、护士、医护人员、卫生工作者等职业描述",
        "epidemics_labels",
    ),
    "禽、畜类从业人员": (
        "禽类从业人员、畜牧业人员、养殖场工人、家禽接触者、牲畜接触者等",
        "epidemics_labels",
    ),
    "农牧民等野外作业人员": (
        "农民、牧民、野外作业人员、农业工作者、户外劳动者等描述",
        "epidemics_labels",
    ),
    "病原微生物实验室检测人员": (
        "实验室人员、微生物检测人员、病原体检测工作者等描述",
        "epidemics_labels",
    ),
    "可疑动物或动物制品接触史": (
        "动物接触史、野生动物接触、动物制品接触、禽类接触史、宠物接触等描述",
        "epidemics_labels",
    ),
    "可疑环境暴露史": (
        "环境暴露史、疫区暴露、可疑场所接触、高风险环境暴露等描述",
        "epidemics_labels",
    ),
    "外出外来史": (
        "外出史、旅行史、外来人员、流行地区旅居史、境外旅行史等描述",
        "epidemics_labels",
    ),

    # ── 时间 (time_labels) ────────────────────────────────────────
    "当前": (
        "当前时间状态词，如：目前、现在、此次、本次、当前等",
        "time_labels",
    ),
    "既往": (
        "既往时间状态词，如：既往、以前、曾经、过去、历史上等",
        "time_labels",
    ),
    "进行性加重": (
        "进行性加重时间描述，如：进行性、逐渐加重、持续恶化、进展性等",
        "time_labels",
    ),
    "持续": (
        "持续时间描述，如：持续、一直、连续、不间断等",
        "time_labels",
    ),

    # ── 断言 (status_labels) ──────────────────────────────────────
    "肯定": (
        "肯定断言词，如：确有、明确、确实、已证实等表示肯定存在的词",
        "status_labels",
    ),
    "否定": (
        "否定断言词，如：无、未、否认、不伴、未见、排除、没有、不存在等",
        "status_labels",
    ),
    "可疑": (
        "可疑断言词，如：疑似、可能、不除外、考虑、待排、疑诊等",
        "status_labels",
    ),
    "条件性": (
        "条件性断言词，如：若、如果、当…时、在…情况下等条件限定词",
        "status_labels",
    ),
    "假设性": (
        "假设性断言词，如：假设、推测、估计、预计、可能为等假设性描述",
        "status_labels",
    ),
    "与患者本人无关": (
        "与患者本人无关的描述，如：家族史、家属、父母、兄弟姐妹等他人信息",
        "status_labels",
    ),

    # ── 测量实体 (measure_entities) ───────────────────────────────
    "呼吸频率": (
        "呼吸频率指标名称，如：呼吸频率、呼吸次数、RR、呼吸率等",
        "measure_entities",
    ),
    "指氧饱和度": (
        "指氧饱和度指标名称，如：指氧饱和度、血氧饱和度、SpO2、氧饱和度等",
        "measure_entities",
    ),
    "动脉氧分压(PaO2)": (
        "动脉氧分压指标名称，如：动脉氧分压、PaO2、氧分压等",
        "measure_entities",
    ),
    "吸氧浓度(FiO2)": (
        "吸氧浓度指标名称，如：吸氧浓度、FiO2、吸入氧浓度等",
        "measure_entities",
    ),

    # ── 测量属性 (measure_labels) ─────────────────────────────────
    "数值": (
        "测量数值，如具体的数字读数：38.5、120、96%等测量结果数字",
        "measure_labels",
    ),
    "单位": (
        "测量单位，如：℃、次/分、mmHg、%、L/min、bpm等单位描述",
        "measure_labels",
    ),
    "比较符": (
        "比较符号，如：大于、小于、≥、≤、>、<、高于、低于等比较描述",
        "measure_labels",
    ),
    "阈值判断": (
        "阈值判断描述，如：超过正常范围、低于正常值、异常升高、明显下降等",
        "measure_labels",
    ),

    # ── 病毒 (pathogen) ───────────────────────────────────────────
    "新冠病毒": (
        "新冠病毒、SARS-CoV-2、COVID-19、新型冠状病毒等描述",
        "pathogen",
    ),
    "流感病毒": (
        "流感病毒、甲型流感、乙型流感、H1N1、H3N2、influenza等描述",
        "pathogen",
    ),
    "呼吸道合胞病毒": (
        "呼吸道合胞病毒、RSV、合胞病毒等描述",
        "pathogen",
    ),
    "腺病毒": (
        "腺病毒、adenovirus等描述",
        "pathogen",
    ),
    "人偏肺病毒": (
        "人偏肺病毒、hMPV、human metapneumovirus等描述",
        "pathogen",
    ),
    "副流感病毒": (
        "副流感病毒、parainfluenza virus、PIV等描述",
        "pathogen",
    ),
    "普通冠状病毒": (
        "普通冠状病毒、季节性冠状病毒、coronavirus（非新冠）等描述",
        "pathogen",
    ),
    "博卡病毒": (
        "博卡病毒、bocavirus、HBoV等描述",
        "pathogen",
    ),
    "鼻病毒": (
        "鼻病毒、rhinovirus等描述",
        "pathogen",
    ),
    "肠道病毒": (
        "肠道病毒、enterovirus、EV71、柯萨奇病毒等描述",
        "pathogen",
    ),

    # ── 细菌 (bacteria) ───────────────────────────────────────────
    "A族链球菌": (
        "A族链球菌、化脓性链球菌、GAS、group A streptococcus等描述",
        "bacteria",
    ),
    "百日咳鲍特菌": (
        "百日咳鲍特菌、百日咳杆菌、Bordetella pertussis等描述",
        "bacteria",
    ),
    "肺炎链球菌": (
        "肺炎链球菌、肺炎球菌、Streptococcus pneumoniae等描述",
        "bacteria",
    ),
    "流感嗜血杆菌": (
        "流感嗜血杆菌、Haemophilus influenzae、Hi等描述",
        "bacteria",
    ),
    "军团菌": (
        "军团菌、嗜肺军团菌、Legionella pneumophila等描述",
        "bacteria",
    ),
    "肺炎克雷伯菌": (
        "肺炎克雷伯菌、Klebsiella pneumoniae、KP等描述",
        "bacteria",
    ),

    # ── 其他病原体 (other_pathogen) ───────────────────────────────
    "肺炎支原体": (
        "肺炎支原体、Mycoplasma pneumoniae、MP等描述",
        "other_pathogen",
    ),
    "曲霉菌": (
        "曲霉菌、侵袭性曲霉病、Aspergillus等描述",
        "other_pathogen",
    ),
    "隐球菌": (
        "隐球菌、新型隐球菌、Cryptococcus等描述",
        "other_pathogen",
    ),
    "鹦鹉热衣原体": (
        "鹦鹉热衣原体、Chlamydia psittaci、鹦鹉热等描述",
        "other_pathogen",
    ),
    "肺炎衣原体": (
        "肺炎衣原体、Chlamydia pneumoniae等描述",
        "other_pathogen",
    ),
}

# 预计算：提示词列表和标签名列表（顺序一一对应）
PROMPT_LIST: list[str] = [v[0] for v in LABEL_PROMPTS.values()]
LABEL_LIST:  list[str] = list(LABEL_PROMPTS.keys())
FROM_NAME_LIST: list[str] = [v[1] for v in LABEL_PROMPTS.values()]

# GLiNER2 专用格式：{ "标签值": "详细提示词" }
GLINER2_LABELS: dict[str, str] = {k: v[0] for k, v in LABEL_PROMPTS.items()}

# 统一使用 chief_complaint_text，因为 XML 中没有名为 'text' 的组件
PATHOGEN_GROUPS = set()

# ── 关系定义 ──────────────────────────────────────────────────────────────────
# 格式：{ "关系名": {"description": "提示词", "threshold": float, "ls_relation": "LS关系值"} }
# head = 修饰词/来源，tail = 被修饰的目标
#
# 对应 pneumonia.xml 中 <Relations> 定义的三种关系：
#   状态：断言 span → 症状/诊断 span
#   时间：时间 span → 症状 span
#   对应指标：症状 span → 测量实体 span

RELATION_SCHEMA: dict[str, dict] = {
    "状态": {
        "description": (
            "断言修饰词（否定/可疑/肯定/条件性/假设性/与患者本人无关）修饰某个症状或诊断。"
            "head 是断言词 span，tail 是被修饰的症状或诊断 span。"
            "例：'不发热' 中 head='不'(否定)，tail='发热'(症状)。"
            "assertion modifier modifies a symptom or diagnosis span"
        ),
        "threshold": 0.35,
        "ls_relation": "状态",
    },
    "时间": {
        "description": (
            "时间修饰词（当前/既往/持续/进行性加重）修饰某个症状。"
            "head 是时间词 span，tail 是被修饰的症状 span。"
            "例：'既往有哮喘' 中 head='既往'(时间)，tail='哮喘'(症状)。"
            "temporal modifier modifies a symptom span"
        ),
        "threshold": 0.35,
        "ls_relation": "时间",
    },
    "测量": {
        "description": (
            "测量实体或测量属性对应某个症状或临床表现。"
            "head 是测量实体/属性 span（呼吸频率/血氧饱和度/数值/单位等），tail 是目标症状或实体 span。"
            "例：'气促，呼吸频率40次/分' 中 head='呼吸频率'(测量实体)，tail='气促'(症状)。"
            "measurement entity or attribute corresponds to a symptom or clinical finding"
        ),
        "threshold": 0.35,
        "ls_relation": "测量",
    },
}
