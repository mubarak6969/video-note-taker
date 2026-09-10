"""A small, hand-curated corpus + golden questions for deterministic RAG
evaluation. Every question's expected evidence is knowable from the
corpus by inspection, so retrieval/attribution behavior can be graded
without a live LLM call.
"""

CORPUS = [
    {
        "source_id": "eval_yt_1",
        "title": "Intro to Photosynthesis",
        "source_type": "youtube_video",
        "chunks": [
            {"text": "Photosynthesis converts sunlight into chemical energy using chlorophyll in plant cells.", "start": 0, "end": 20},
            {"text": "The Calvin cycle fixes carbon dioxide into glucose molecules inside the chloroplast.", "start": 20, "end": 45},
            {"text": "Plants release oxygen into the atmosphere as a byproduct of the light reactions.", "start": 45, "end": 70},
        ],
    },
    {
        "source_id": "eval_pdf_1",
        "title": "Company Handbook",
        "source_type": "pdf_document",
        "chunks": [
            {"text": "Employees are entitled to 24 days of paid leave per calendar year.", "page": 4},
            {"text": "The reimbursement policy code is HB-4471 for approved travel expenses.", "page": 7},
            {"text": "Remote work requests must be submitted two weeks in advance to a manager.", "page": 9},
        ],
    },
    {
        "source_id": "eval_txt_1",
        "title": "Meeting Notes - Q3 Planning",
        "source_type": "text_document",
        "chunks": [
            {"text": "The team agreed to prioritize the mobile app redesign initiative in Q3.", "page": None},
            {"text": "Budget approval code PROJ-8823 was assigned to the redesign initiative.", "page": None},
        ],
    },
]

# category values: exact_keyword | semantic | cross_source | source_attribution
# | insufficient_context | follow_up
GOLDEN_QUESTIONS = [
    {
        "id": "exact_keyword_pdf_reimbursement_code",
        "category": "exact_keyword",
        "question": "What is the reimbursement policy code?",
        "scope_source_id": "eval_pdf_1",
        "expected_text_substring": "HB-4471",
    },
    {
        "id": "exact_keyword_txt_budget_code",
        "category": "exact_keyword",
        "question": "What is budget approval code PROJ-8823 for?",
        "scope_source_id": "eval_txt_1",
        "expected_text_substring": "PROJ-8823",
    },
    {
        "id": "semantic_photosynthesis_breathable_air",
        "category": "semantic",
        # Deliberately paraphrased - no "oxygen"/"byproduct" keyword overlap -
        # so this only passes if semantic (vector) search is doing real work.
        "question": "How do plants make the air breathable for animals?",
        "scope_source_id": "eval_yt_1",
        "expected_text_substring": "oxygen",
    },
    {
        "id": "cross_source_leave_and_budget",
        "category": "cross_source",
        "question": "Tell me about employee leave policy and project budget codes.",
        "scope_source_id": None,  # library-wide
        "expected_source_ids": {"eval_pdf_1", "eval_txt_1"},
    },
    {
        "id": "attribution_pdf_leave_days",
        "category": "source_attribution",
        "question": "How many days of paid leave do employees get?",
        "scope_source_id": "eval_pdf_1",
        "expected_source_type": "pdf_document",
        "expected_title": "Company Handbook",
    },
    {
        "id": "attribution_youtube_calvin_cycle",
        "category": "source_attribution",
        "question": "What does the Calvin cycle do?",
        "scope_source_id": "eval_yt_1",
        "expected_source_type": "youtube_video",
        "expected_title": "Intro to Photosynthesis",
    },
    {
        "id": "insufficient_context_unprocessed_source",
        "category": "insufficient_context",
        "question": "What is the capital of France?",
        "scope_source_id": "a_source_id_that_was_never_processed",
    },
    {
        "id": "insufficient_context_off_topic_within_processed_source",
        "category": "insufficient_context",
        # eval_pdf_1 exists and has chunks - unlike the case above, this
        # tests the relevance *gate*, not an empty pool: pre-gate, this
        # question would have returned the "least bad" PDF chunks anyway.
        "question": "What is the capital of France?",
        "scope_source_id": "eval_pdf_1",
    },
    {
        "id": "followup_photosynthesis_byproduct",
        "category": "follow_up",
        "question": "What about its byproduct?",
        "scope_source_id": "eval_yt_1",
        "chat_history": [
            (
                "What converts sunlight into energy in a plant cell?",
                'Photosynthesis, using chlorophyll (Source 1, 0:00-0:20).',
            )
        ],
    },
]
