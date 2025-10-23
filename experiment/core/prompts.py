"""
Prompt generation module
"""

# ==================== Definitions ====================

INTIMACY_DEFINITION = (
    "A relationship is intimate if both people are emotionally close and warm to each other. "
    "Otherwise, the relationship is not intimate."
)

FORMALITY_DEFINITION = (
    "A relationship is pleasure-oriented if both people interact socially and their relationship "
    "is not bound by professional rules or regulations. Otherwise, the relationship is task-oriented."
)

HIERARCHY_DEFINITION = (
    "A relationship is equal if both people (a) have the same social status, (b) are at the same "
    "level in the power (c) share similar responsibilities, or (d) have the same role. "
    "Otherwise, the relationship is hierarchical."
)

# Korean definitions
INTIMACY_DEFINITION_KO = (
    "관계가 친밀하다면 두 사람이 서로 감정적으로 가깝고 따뜻한 관계일 때입니다. "
    "그렇지 않으면 그 관계는 친밀하지 않습니다."
)

FORMALITY_DEFINITION_KO = (
    "관계가 비공식적이라면 두 사람이 사회적으로 상호작용하며 그 관계가 직업적인 규칙이나 규제에 "
    "의해 구속되지 않을 때입니다. 그렇지 않으면 그 관계는 공식적입니다."
)

HIERARCHY_DEFINITION_KO = (
    "관계가 평등하다면 두 사람이 (a) 동일한 사회적 지위를 가지고, (b) 권력에서 같은 수준에 있으며, "
    "(c) 비슷한 책임을 지거나, (d) 동일한 역할을 가지고 있을 때입니다. "
    "그렇지 않으면 그 관계는 계층적입니다."
)

    # ==================== Instructions ====================

INTIMACY_INSTRUCTION = (
    "Determine whether the relationship between the two people in the dialogue is intimate or not. "
    "If the relationship is intimate, output 'intimate'. Otherwise, output 'not intimate'. "
    "If the relationship is neither intimate nor not intimate, output 'neutral'."
)

FORMALITY_INSTRUCTION = (
    "Determine whether the relationship between the two people in the dialogue is formal or not. "
    "If the relationship is formal, output 'formal'. Otherwise, output 'informal'. "
    "If the relationship is neither formal nor informal, output 'neutral'."
)

HIERARCHY_INSTRUCTION = (
    "Determine whether the relationship between the two people in the dialogue is equal or not. "
    "If the relationship is equal, output 'equal'. Otherwise, output 'hierarchical'."
)

# Korean instructions
INTIMACY_INSTRUCTION_KO = (
    "대화에서 A와 B, 두 사람 간의 관계가 친밀한지 아닌지를 판단하십시오. "
    "관계가 친밀하다면 'intimate'를 출력하세요. 그렇지 않으면 'not intimate'를 출력하세요. "
    "관계가 친밀하지도 않으면서 그렇지도 않다면 'neutral'을 출력하세요."
)

FORMALITY_INSTRUCTION_KO = (
    "대화에서 A와 B, 두 사람 간의 관계가 공식적인지 아닌지를 판단하십시오. "
    "관계가 공식적이라면 'formal'을 출력하세요. 그렇지 않으면 'informal'을 출력하세요. "
    "관계가 공식적이지도 비공식적이지도 않다면 'neutral'을 출력하세요."
)

HIERARCHY_INSTRUCTION_KO = (
    "대화에서 A와 B, 두 사람 간의 관계가 평등한지 아닌지를 판단하십시오. "
    "관계가 평등하다면 'equal'을 출력하세요. 그렇지 않으면 'hierarchical'을 출력하세요."
)

# ==================== Prompt generation functions ====================

def get_factor_prompt(factor: str, dialogue: list) -> str:
    """
    English factor prompt generation
    
    Args:
        factor: Factor ('intimacy', 'formality', 'hierarchy')
        dialogue: Dialogue list
    
    Returns:
        str: Generated prompt
    """
    basic_prompt = (
        "Based on Following Definition, {Instruction}\n\n"
        "Definition: {Definition}\n\n"
        "Dialogue: {Dialogue}\n\n"
        "Output Format: {{'answer': \\relation}}\n\n"
        "Output: "
    )
    
    if factor == 'intimacy':
        return basic_prompt.format(
            Instruction=INTIMACY_INSTRUCTION,
            Definition=INTIMACY_DEFINITION,
            Dialogue=dialogue
        )
    elif factor == 'formality':
        return basic_prompt.format(
            Instruction=FORMALITY_INSTRUCTION,
            Definition=FORMALITY_DEFINITION,
            Dialogue=dialogue
        )
    elif factor == 'hierarchy':
        return basic_prompt.format(
            Instruction=HIERARCHY_INSTRUCTION,
            Definition=HIERARCHY_DEFINITION,
            Dialogue=dialogue
        )
    else:
        raise ValueError("Invalid factor. Choose from 'intimacy', 'formality', or 'hierarchy'.")


def get_factor_prompt_ko(factor: str, dialogue: list) -> str:
    """
    Korean factor prompt generation
    
    Args:
        factor: Factor ('intimacy', 'formality', 'hierarchy')
        dialogue: Dialogue list
    
    Returns:
        str: Generated prompt
    """
    basic_prompt = (
        "다음 정의를 바탕으로 {Instruction}\n\n"
        "정의: {Definition}\n\n"
        "대화: {Dialogue}\n\n"
        "답은 반드시 JSON 형식으로 답변하시오: {{'answer': \\=}}\n\n"
        "답변: "
    )
    
    if factor == 'intimacy':
        return basic_prompt.format(
            Instruction=INTIMACY_INSTRUCTION_KO,
            Definition=INTIMACY_DEFINITION_KO,
            Dialogue="\n".join(dialogue)
        )
    elif factor == 'formality':
        return basic_prompt.format(
            Instruction=FORMALITY_INSTRUCTION_KO,
            Definition=FORMALITY_DEFINITION_KO,
            Dialogue="\n".join(dialogue)
        )
    elif factor == 'hierarchy':
        return basic_prompt.format(
            Instruction=HIERARCHY_INSTRUCTION_KO,
            Definition=HIERARCHY_DEFINITION_KO,
            Dialogue="\n".join(dialogue)
        )
    else:
        raise ValueError("Invalid factor. Choose from 'intimacy', 'formality', or 'hierarchy'.")


def get_factor_prompt_all(dialogue: list) -> str:
    """
    English prompt generation with all factors
    
    Args:
        dialogue: Dialogue list
    
    Returns:
        str: Generated prompt
    """
    basic_prompt = (
        "Determine the levels of Intimacy, Formality, and Hierarchy between [A] and [B] "
        "in the dialogues below. {Instruction}\n\n"
        "Dialogue:\n\n{Dialogue}\n\n"
        "Output Format: {{'intimacy': \\, 'formality': \\, 'hierarchy': \\}}\n\n"
        "Output: "
    )
    
    instruction_all = (
        f"Intimacy:\n{INTIMACY_INSTRUCTION}\n"
        f"Formality:\n{FORMALITY_INSTRUCTION}\n"
        f"Hierarchy:\n{HIERARCHY_INSTRUCTION}"
    )
    
    return basic_prompt.format(Instruction=instruction_all, Dialogue=dialogue)


def get_factor_prompt_all_ko(dialogue: list) -> str:
    """
    Korean prompt generation with all factors
    
    Args:
        dialogue: Dialogue list
    
    Returns:
        str: Generated prompt
    """
    basic_prompt = (
        "아래의 대화에서 [A]와 [B] 사이의 친밀감(Intimacy), 격식(Formality), "
        "위계(Hierarchy) 수준을 판단하세요.\n\n{Instruction}\n\n"
        "대화:\n\n{Dialogue}\n\n"
        "Output Format: {{'intimacy': \\, 'formality': \\, 'hierarchy': \\}}\n\n"
        "Output: "
    )
    
    instruction_all = (
        f"친밀도:\n{INTIMACY_INSTRUCTION_KO}\n"
        f"격식:\n{FORMALITY_INSTRUCTION_KO}\n"
        f"위계:\n{HIERARCHY_INSTRUCTION_KO}"
    )
    
    return basic_prompt.format(Instruction=instruction_all, Dialogue=dialogue)


# ==================== Relationship prompt ====================

RELATIONSHIP_PROMPT = """Analyze the given dialogue and the information provided about the interactions between characters to classify their relationship. Categorize each relationship into one of the seven relationship categories and specify the sub-category for clarity.

Relationship Categories and Corresponding Sub-Categories:
- Family (e.g., parent, child, siblings)
- Social (e.g., friend, acquaintance, neighbor, complete stranger)
- Romance (e.g., dating, married, engaged, divorcee, ex-boyfriend/girlfriend)
- Organizational (e.g., coworker, colleague, another employee, boss)
- Peer Group (e.g., classmate, sports teammate, club member)
- Parasocial (e.g., fan, hero)
- Role-Based (e.g., law enforcement, individual with authority, mentor, mentee, teacher, student, lawyer, client, doctor, patient)
- Antagonist (e.g., competitor, rival, enemy)
- Other

Output Example (Response in JSON Format):

{{
    "Category": "Family",
    "Sub-Category": "Parent and Child"
}},
{{
    "Category": "Romance",
    "Sub-Category": "Dating"
}},
{{
    "Category": "Social",
    "Sub-Category": "Neighbor"
}}

Dialogue: {Dialogue}
Information: {Information}
Output:
"""

RELATIONSHIP_PROMPT_MOVIE = """This is a part of the movie {movie}. Refer to the given dialogue and the information provided about the interactions between characters to classify their relationship. Categorize each relationship into one of the seven relationship categories and specify the sub-category for clarity.

Relationship Categories and Corresponding Sub-Categories:
- Family (e.g., parent, child, siblings)
- Social (e.g., friend, acquaintance, neighbor, complete stranger)
- Romance (e.g., dating, married, engaged, divorcee, ex-boyfriend/girlfriend)
- Organizational (e.g., coworker, colleague, another employee, boss)
- Peer Group (e.g., classmate, sports teammate, club member)
- Parasocial (e.g., fan, hero)
- Role-Based (e.g., law enforcement, individual with authority, mentor, mentee, teacher, student, lawyer, client, doctor, patient)
- Antagonist (e.g., competitor, rival, enemy)
- Other

Output Example (Response in JSON Format):

{{
    "Category": "Family",
    "Sub-Category": "Parent and Child"
}},
{{
    "Category": "Romance",
    "Sub-Category": "Dating"
}},
{{
    "Category": "Social",
    "Sub-Category": "Neighbor"
}}

Dialogue: {Dialogue}
Information: {Information}
Output:
"""

