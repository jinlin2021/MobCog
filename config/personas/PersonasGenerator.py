import json
import random


FAMILY_NAMES = ["Zhang", "Li", "Wang", "Zhao", "Liu", "Chen", "Yang", "Huang",
                 "Wu", "Zhou", "Zheng", "Sun"]

MALE_GIVEN_NAMES = [
    "Guoqiang", "Jianguo", "Junjie", "Zhiqiang", "Xiaoming", "Wenming",
    "Jianhua", "Liqiang", "Yongqiang", "Haiming", "Desheng", "Jialin"
]

FEMALE_GIVEN_NAMES = [
    "Xiulan", "Yuzhen", "Guiying", "Guilan", "Xiuying", "Xiaohong",
    "Lihua", "Yuhua", "Xiuzhen", "Shuzhen", "Aihua", "Huiqin"
]


def random_name(gender: str) -> str:
    """Return a full name."""
    i_f = random.randrange(len(FAMILY_NAMES))
    family_py = FAMILY_NAMES[i_f]
    if gender == "male":
        i_g = random.randrange(len(MALE_GIVEN_NAMES))
        given_py = MALE_GIVEN_NAMES[i_g]
    else:
        i_g = random.randrange(len(FEMALE_GIVEN_NAMES))
        given_py = FEMALE_GIVEN_NAMES[i_g]
    return f"{family_py} {given_py}"


# ========= 8 profile configs ========= #

PROFILE_CONFIG = {
    "P1": {
        "profile_label": "Healthy, active",
        "age_range": (65, 80),
        # Intact cognition | Robust mobility | Stable, engaged
        "expected_mmse_range": (27, 30),
        "expected_adl_range": (90, 100),
        "expected_gds_range": (0, 3),
        "cognitive_templates": [
            "intact cognition with clear memory and attention",
            "intact cognition and well-preserved executive function",
            "intact cognition and reliable spatial orientation"
        ],
        "mobility_templates": [
            "robust mobility with independent long-distance walking",
            "robust mobility and a steady, confident gait",
            "robust mobility and easy use of stairs and ramps"
        ],
        "affect_templates": [
            "stable mood and actively engaged in daily life",
            "emotionally stable and socially engaged",
            "stable, positive mood with regular participation in activities"
        ]
    },
    "P2": {
        "profile_label": "Cognitively intact, physically frail",
        "age_range": (70, 90),
        # Intact cognition | Physically frail | Stable, concerned
        "expected_mmse_range": (27, 30),
        "expected_adl_range": (60, 85),
        "expected_gds_range": (0, 5),
        "cognitive_templates": [
            "intact cognition with good orientation and reasoning",
            "intact cognition and preserved route knowledge",
            "intact cognition and the ability to follow schedules"
        ],
        "mobility_templates": [
            "physically frail with slow gait and limited endurance",
            "physically frail and easily fatigued when walking",
            "physically frail and unsteady on stairs or slopes"
        ],
        "affect_templates": [
            "emotionally stable but concerned about falls and fatigue",
            "stable mood with ongoing worries about physical safety",
            "emotionally stable yet frequently concerned about overexertion"
        ]
    },
    "P3": {
        "profile_label": "Cognitively intact, severely mobility-limited / disabled",
        "age_range": (70, 90),
        # Near-intact cognition | Severely limited | Variable mood
        "expected_mmse_range": (26, 30),
        "expected_adl_range": (20, 60),
        "expected_gds_range": (0, 7),
        "cognitive_templates": [
            "near-intact cognition with only mild cognitive slowing",
            "near-intact cognition and generally preserved memory",
            "near-intact cognition with slightly reduced processing speed"
        ],
        "mobility_templates": [
            "severely limited mobility and dependent on assistance",
            "severely limited mobility with very little independent walking",
            "severely limited mobility and frequent need for a wheelchair"
        ],
        "affect_templates": [
            "variable mood influenced by pain and accessibility barriers",
            "variable mood with alternating frustration and acceptance",
            "variable mood that shifts with the ease or difficulty of travel"
        ]
    },
    "P4": {
        "profile_label": "Mild cognitive impairment, robust mobility",
        "age_range": (65, 85),
        # Mild impairment | Good mobility | Mild anxiety
        "expected_mmse_range": (21, 26),
        "expected_adl_range": (85, 100),
        "expected_gds_range": (2, 7),
        "cognitive_templates": [
            "mild cognitive impairment with occasional forgetfulness",
            "mild impairment with reduced short-term memory for routes",
            "mild impairment and slightly inconsistent attention"
        ],
        "mobility_templates": [
            "good mobility with independent walking and stair use",
            "good mobility and able to manage long walking distances",
            "good mobility and quick movement between stops or platforms"
        ],
        "affect_templates": [
            "mild anxiety about making mistakes during travel",
            "mild anxiety when routes or plans change unexpectedly",
            "mild, travel-related anxiety despite generally adequate function"
        ]
    },
    "P5": {
        "profile_label": "Mild cognitive impairment, physically frail",
        "age_range": (70, 90),
        # Mild impairment | Frail mobility | Mild depression
        "expected_mmse_range": (21, 26),
        "expected_adl_range": (40, 75),
        "expected_gds_range": (5, 9),
        "cognitive_templates": [
            "mild cognitive impairment with noticeable forgetfulness",
            "mild impairment with difficulty holding complex routes in mind",
            "mild impairment and occasional confusion in busy environments"
        ],
        "mobility_templates": [
            "frail mobility with limited walking distance",
            "frail mobility and unsteady gait on uneven ground",
            "frail mobility with frequent need to rest while walking"
        ],
        "affect_templates": [
            "mild depression with reduced interest in going out",
            "mild depression and a generally low mood",
            "mildly depressed mood that dampens motivation to travel"
        ]
    },
    "P6": {
        "profile_label": "Moderate–severe cognitive impairment, mobile",
        "age_range": (70, 90),
        # Moderate–severe impairment | Mobile with aid | Anxious/unstable
        "expected_mmse_range": (10, 20),
        "expected_adl_range": (50, 85),
        "expected_gds_range": (6, 12),
        "cognitive_templates": [
            "moderate–severe cognitive impairment with marked memory loss",
            "moderate–severe impairment with frequent disorientation",
            "moderate–severe impairment and difficulty following instructions"
        ],
        "mobility_templates": [
            "mobile with aid such as a cane or walker",
            "mobile with aid and able to walk moderate distances",
            "mobile with aid but requiring supervision for safety"
        ],
        "affect_templates": [
            "anxious and emotionally unstable during travel",
            "frequent anxiety and unstable mood in unfamiliar situations",
            "anxious, easily agitated, and emotionally unpredictable"
        ]
    },
    "P7": {
        "profile_label": "Moderate–severe cognitive impairment, severely mobility-limited",
        "age_range": (75, 95),
        # Moderate–severe impairment | Severely limited | Psychological symptoms
        "expected_mmse_range": (8, 20),
        "expected_adl_range": (0, 40),
        "expected_gds_range": (8, 15),
        "cognitive_templates": [
            "moderate–severe cognitive impairment with poor orientation",
            "moderate–severe impairment and severely reduced understanding of routes",
            "moderate–severe impairment with little capacity for independent planning"
        ],
        "mobility_templates": [
            "severely limited mobility and dependent on others for transfers",
            "severely limited mobility and mostly wheelchair-bound",
            "severely limited mobility with virtually no independent walking"
        ],
        "affect_templates": [
            "psychological symptoms such as agitation or apathy",
            "psychological symptoms including mistrust or suspiciousness",
            "psychological symptoms with frequent behavioral disturbance"
        ]
    },
    "P8": {
        "profile_label": "Emotionally vulnerable / socially isolated",
        "age_range": (65, 90),
        # Subjective complaints | Near-normal mobility | Depressed, isolated
        "expected_mmse_range": (26, 30),
        "expected_adl_range": (80, 100),
        "expected_gds_range": (8, 15),
        "cognitive_templates": [
            "subjective cognitive complaints despite near-normal testing",
            "subjective complaints of memory problems with generally normal scores",
            "subjective complaints of slowing or forgetfulness without clear deficits"
        ],
        "mobility_templates": [
            "near-normal mobility with independent walking and transfers",
            "near-normal mobility but infrequent trips outside the home",
            "near-normal mobility with only mild physical slowing"
        ],
        "affect_templates": [
            "depressed mood and socially isolated",
            "depressed, withdrawn, and often feeling alone",
            "persistent low mood with marked social isolation"
        ]
    }
}


# ========= persona generator ========= #

def generate_elder_personas_all_profiles(per_profile: int = 15, seed: int = 42):
    """
    Generate elder personas for all 8 profiles (P1–P8).

    Fields:
      id, profile_id, profile_label,
      gender, name, age,
      cognitive_level, mobility, affect_profile,
      expected_mmse_range, expected_adl_range, expected_gds_range,
      description.
    """
    random.seed(seed)
    personas = []
    global_id = 1

    for profile_id in sorted(PROFILE_CONFIG.keys()):
        cfg = PROFILE_CONFIG[profile_id]
        for _ in range(per_profile):
            gender = random.choice(["male", "female"])
            name = random_name(gender)
            age = random.randint(*cfg["age_range"])

            cognitive_level = random.choice(cfg["cognitive_templates"])
            mobility = random.choice(cfg["mobility_templates"])
            affect_profile = random.choice(cfg["affect_templates"])

            gender_word = "man" if gender == "male" else "woman"

            persona = {
                "id": global_id,
                "profile_id": profile_id,
                "profile_label": cfg["profile_label"],
                "gender": gender,
                "name": name,
                "age": age,
                "cognitive_level": cognitive_level,
                "mobility": mobility,
                "affect_profile": affect_profile,
                "expected_mmse_range": list(cfg["expected_mmse_range"]),
                "expected_adl_range": list(cfg["expected_adl_range"]),
                "expected_gds_range": list(cfg["expected_gds_range"]),
                "description": (
                    f"{name} is a {age}-year-old {gender_word} with "
                    f"{cognitive_level}, {mobility}, and {affect_profile}."
                )
            }

            personas.append(persona)
            global_id += 1

    return personas


def save_personas_to_file(personas, filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(personas, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
  
    personas = generate_elder_personas_all_profiles(per_profile=15, seed=2025)
    
    save_personas_to_file(personas, "elder_personas_P1-P8.json")
