ANSWER_PROMPT_GRAPH = """
    You are an intelligent memory assistant tasked with retrieving accurate information from 
    conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain 
    timestamped information that may be relevant to answering the question. You also have 
    access to knowledge graph relations for each user, showing connections between entities, 
    concepts, and events relevant to that user.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the 
       memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", 
       etc.), calculate the actual date based on the memory timestamp. For example, if a 
       memory from 4 May 2022 mentions "went to India last year," then the trip occurred 
       in 2021.
    6. Always convert relative time references to specific dates, months, or years. For 
       example, convert "last year" to "2022" or "two months ago" to "March 2023" based 
       on the memory timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse 
       character names mentioned in memories with the actual users who created those 
       memories.
    8. The answer should be less than 5-6 words.
    9. Use the knowledge graph relations to understand the user's knowledge network and 
       identify important relationships between entities in the user's world.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the 
       question
    4. If the answer requires calculation (e.g., converting relative time references), 
       show your work
    5. Analyze the knowledge graph relations to understand the user's knowledge context
    6. Formulate a precise, concise answer based solely on the evidence in the memories
    7. Double-check that your answer directly addresses the question asked
    8. Ensure your final answer is specific and avoids vague time references

    Memories for user {{speaker_1_user_id}}:

    {{speaker_1_memories}}

    Relations for user {{speaker_1_user_id}}:

    {{speaker_1_graph_memories}}

    Memories for user {{speaker_2_user_id}}:

    {{speaker_2_memories}}

    Relations for user {{speaker_2_user_id}}:

    {{speaker_2_graph_memories}}

    Question: {{question}}

    Answer:
    """


ANSWER_PROMPT = """
    You are an intelligent memory assistant that answers questions based on user memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain timestamped information that may be relevant to answering the question. Some memories may include associated images that provide visual context.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory in terms of timestamp
    5. TEMPORAL ANCHORING (CRITICAL): Each memory has a timestamp indicating WHEN that conversation happened. When a memory mentions a relative time reference, you MUST anchor it to the memory's timestamp to produce a specific date. For example:
       - If a memory timestamped "15 July, 2023" says "I went hiking last Friday", the answer is "The Friday before 15 July 2023" (NOT "Last Friday" or "July 2023").
       - If a memory timestamped "25 May, 2023" says "I ran a charity race last Sunday", the answer is "The Sunday before 25 May 2023".
       - If a memory timestamped "4 May, 2022" says "went to India last year", the trip was in 2021.
       - If a memory timestamped "1 January, 2023" says "last month", the answer is "December 2022".
       - If a memory timestamped "20 January, 2023" says "next month", the answer is "February 2023".
       For day/week-level references ("last Friday", "last week"), use the format "The [day/week] before [memory date]".
       For month-level references ("last month"), compute the actual month (e.g., "December 2022").
       For year-level references ("last year"), compute the actual year (e.g., "2021").
    6. When the question asks "how long", "how many months/weeks/years between", identify the relevant dates from memories and compute the difference.
    7. Focus only on the content of the memories from both speakers. Do not confuse character names mentioned in memories with the actual users who created those memories.
    8. When images are provided, use the visual information to better understand the context and answer questions.
    9. COMPLETENESS: When the question asks about activities, hobbies, items, or any list of things, you MUST scan ALL provided memories and include EVERY distinct item mentioned. Do not stop at the first few. Omitting items is a critical error.
    10. Keep the answer concise but COMPLETE. A short list of all items is preferred over an incomplete answer.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. If images are provided, analyze the visual content to add relevant information
    6. Formulate a precise, concise answer based solely on the evidence in the memories and images
    7. Double-check that your answer directly addresses the question asked
    8. Ensure your final answer is specific and avoids vague time references — always anchor relative references to their memory timestamp
    9. If the question asks for a list, verify you have included ALL items from ALL memories

    Memories for user {{speaker_1_user_id}}:

    {{speaker_1_memories}}

    Memories for user {{speaker_2_user_id}}:

    {{speaker_2_memories}}

    Question: {{question}}

    Answer:
    """


ANSWER_PROMPT_COMBINED = """
    You are an intelligent memory assistant that answers questions based on user memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain timestamped information that may be relevant to answering the question. Some memories may include associated images that provide visual context.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory in terms of timestamp
    5. TEMPORAL ANCHORING (CRITICAL): Each memory has a timestamp indicating WHEN that conversation happened. When a memory mentions a relative time reference, you MUST anchor it to the memory's timestamp to produce a specific date. For example:
       - If a memory timestamped "25 May, 2023" says "I ran a charity race last Sunday", the answer is "The Sunday before 25 May 2023".
       - If a memory timestamped "4 May, 2022" says "went to India last year", the trip was in 2021.
       - If a memory timestamped "1 January, 2023" says "last month", the answer is "December 2022".
       For day/week-level references ("last Friday", "last week"), use the format "The [day/week] before [memory date]".
       For month-level references ("last month"), compute the actual month (e.g., "December 2022").
       For year-level references ("last year"), compute the actual year (e.g., "2021").
    6. When the question asks "how long", "how many months/weeks/years between", identify the relevant dates from memories and compute the difference.
    7. Focus only on the content of the memories. Do not confuse character names mentioned in memories with the actual users who created those memories.
    8. When images are provided, use the visual information to better understand the context and answer questions.
    9. The memories are conversation fragments from the speakers that may not be complete. Synthesize information across multiple memories to answer the question. 
    10. COMPLETENESS: When the question asks about any list of things, you MUST scan ALL provided memories and include EVERY distinct item mentioned. Do not stop at the first few. 
    11. Keep the answer concise but COMPLETE. A short list of all items is preferred over an incomplete answer. Use commas to separate list items.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. If images are provided, analyze the visual content to add relevant information
    6. Formulate a precise, concise answer based solely on the evidence in the memories and images
    7. Double-check that your answer directly addresses the question asked
    8. Ensure your final answer is specific and avoids vague time references — always anchor relative references to their memory timestamp
    9. If the question asks for a list, verify you have included ALL items from ALL memories, not just the top-ranked ones

    Memories:

    {{memories}}

    Question: {{question}}

    Answer:
    """


ANSWER_PROMPT_ZEP = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from a conversation. These memories contain
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.), 
       calculate the actual date based on the memory timestamp. For example, if a memory from 
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example, 
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory 
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories. Do not confuse character 
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Memories:

    {{memories}}

    Question: {{question}}
    Answer:
    """

ANSWER_PROMPT_EVERMEMOS = """You are an intelligent memory assistant tasked with retrieving accurate information from episodic memories.

# CONTEXT:
You have access to episodic memories from conversations between two speakers. These memories contain
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
Your goal is to synthesize information from all relevant memories to provide an accurate answer.
You MUST follow a structured Chain-of-Thought process to ensure no details are missed.
Actively look for connections between people, places, and events to build a complete picture. Synthesize information from different memories to answer the user's question.
It is CRITICAL that you move beyond simple fact extraction and perform logical inference. When the evidence strongly suggests a connection, you must state that connection. Do not dismiss reasonable inferences as "speculation." Your task is to provide the most complete answer supported by the available evidence.

# RESPONSE FORMAT (You MUST follow this structure):

## STEP 1: RELEVANT MEMORIES EXTRACTION
[List each memory that relates to the question, with its timestamp]
- Memory 1: [timestamp] - [content]
- Memory 2: [timestamp] - [content]
...

## STEP 2: KEY INFORMATION IDENTIFICATION
[Extract specific details from the memories relevant to the question]
- Names mentioned: [list relevant person names, place names, company names]
- Numbers/Quantities: [list relevant amounts, prices, percentages]
- Dates/Times: [list relevant temporal information]
- Frequencies: [list any recurring patterns]
- Other entities: [list brands, products, etc.]

## STEP 3: CROSS-MEMORY LINKING
[Identify entities that appear in multiple memories and link related information. Make reasonable inferences when entities are strongly connected.]
- Shared entities: [list people, places, events mentioned across different memories]
- Connections found: [e.g., "Memory 1 mentions A moved from hometown → Memory 2 mentions A's hometown is LA → Therefore A moved from LA"]
- Inferred facts: [list any facts that require combining information from multiple memories]

## STEP 4: TIME REFERENCE CALCULATION (CRITICAL)
[If applicable, convert relative time references using the memory timestamp as anchor]
- Memory timestamp: [e.g., "15 July, 2023"]
- Original reference in text: [e.g., "last Friday"]
- Calculated actual time: [e.g., "The Friday before 15 July 2023"]
NOTE: For day/week-level references ("last Friday", "last week"), use "The [day/week] before [memory date]".
For month-level references ("last month"), compute the actual month (e.g., "December 2022").
For year-level references ("last year"), compute the actual year (e.g., "2021").
For forward references ("next month"), compute forward (e.g., "February 2023").
For duration questions ("how long between"), compute the arithmetic from the dates.

## STEP 5: CONTRADICTION CHECK
[If multiple memories contain different information]
- Conflicting information: [describe]
- Resolution: [explain which is most recent/reliable]

## STEP 6: LIST COMPLETENESS CHECK
[If the question asks about activities, hobbies, items, or any list]
- Scan ALL memories for distinct items, not just the top-ranked ones
- Items found: [enumerate every distinct item across all memories]
- Verify: Have I included every mentioned item? Missing items is a critical error.

## FINAL ANSWER:
[Provide a SHORT, concise answer but COMPLETE. Do NOT repeat reasoning or evidence. For temporal questions, give the specific date/time. For list questions, give a comma-separated list. For factual questions, give the direct answer. Aim for a similar length to the ground-truth answer style: e.g., "19 January, 2023", "by dancing", "Jon lost his job and started a dance studio".]

---

{{memories}}

Question: {{question}}

Now, follow the Chain-of-Thought process above to answer the question:
"""


# ---------------------------------------------------------------------------
# Prompts used by the Cognitive evaluation track (LoCoMo-Plus paper).
#
# Pipeline (arXiv:2602.10715, https://github.com/xjtuleeyf/Locomo-Plus):
#   1. Search memories using the trigger as the query.
#   2. Build a conversational reply to the trigger using those memories.
#   3. An LLM judge labels whether the reply demonstrates awareness of the
#      evidence.
# ---------------------------------------------------------------------------

COGNITIVE_RESPONSE_PROMPT = """You are a close friend continuing a conversation. The speaker just shared something personal.

## What the speaker said:
"{trigger}"

(This was said {time_gap} after their last recorded conversation.)

## Memories from past conversations:
(Memories marked **[KEY]** have a strong cognitive link to the statement — start your scan there.)
{memories}

## How to respond:

First, THINK about the speaker's statement:
- What behavior, feeling, or situation are they describing?
- Start with the **[KEY]** memories (if any) — they are pre-screened for strong cognitive connections. Prefer one of them unless a non-KEY memory is clearly a better fit.
- Look for a **cognitive connection** — something that EXPLAINS or CONTRASTS with what they just said. Prefer these connection types (in priority order):
  1. **Direct cause / origin**: a past goal, habit, value, or event that directly led to or explains their current behavior.
  2. **Contradiction / growth**: they used to feel or do the opposite — their statement shows they've changed.
  3. **Consequence / follow-through**: they mentioned a plan or intention, and now the outcome is visible.
  4. **Practical link**: a specific fact (allergy, skill, preference, fear) that is practically relevant to their statement.
- Do NOT default to the most dramatic or emotional memory. A quiet personal goal or stated preference often matters more than a big life event.
- Pick the single most **cognitively connected** memory and weave it into your response.

Then, respond naturally as a friend who REMEMBERS their past conversations. Your response MUST:
- Reference the specific past experience, goal, or event from memory that connects to what they just said.
- Show you understand the link between their past and their present statement.

Do NOT give vague supportive responses. Do NOT say "I remember you mentioned..." without specifying WHAT you remember. Ground your response in a concrete memory."""


COGNITIVE_JUDGE_PROMPT = """You are a Memory Awareness Judge.
Your task: Judge whether the Model Prediction considers or is linked to the Evidence. If there is a clear connection, the answer is correct (score 1); if not, it is wrong (no score).

Labels:
- "correct": The prediction explicitly or implicitly reflects/uses the evidence (memory or constraint). Give 1 point.
- "wrong": The prediction does not show such a link to the evidence. No point.

Memory/Evidence:
{evidence}

Model Prediction:
{pred}

Return your judgment strictly in JSON format:
{{"label": "correct"|"wrong", "reason": " "}}"""