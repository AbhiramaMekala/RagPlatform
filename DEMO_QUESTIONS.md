# Demo: ChatGPT vs. my RAG system

Quillfeather Labs is a fictional company whose internal documents exist only in this project.
ChatGPT has never seen them, so it can't answer these questions. At best it says it doesn't know;
at worst it guesses (a hallucination). The RAG system finds the right document and answers with a citation.

This file lives in the project root, OUTSIDE `data/docs/`, so the answer key is never indexed.

| # | Question to ask both | Correct answer (from the docs) | Source file |
|---|---|---|---|
| 1 | What is the dinner per diem at Quillfeather Labs in New York City? | $85 per person | expense-policy |
| 2 | Who has to approve a $7,000 purchase at Quillfeather Labs? | The CFO, Priya Castellanos | expense-policy |
| 3 | What is Project Kestrel and when does it launch? | AI invoice matching in Ledgerlight, target 92% auto-match, launches Nov 18, 2026 | product-roadmap-2026 |
| 4 | What caused incident INC-2291? | Expired TLS certificate on Tollbooth; 47-minute outage on Aug 14, 2026, 1,240 customers affected | incident-inc-2291-postmortem |
| 5 | How many PTO days do Quillfeather employees get, and can they carry any over? | 22 days + 11 holidays; carry over up to 5 days, which expire March 31 | time-off-and-leave |
| 6 | What are Quiet Weeks at Quillfeather Labs? | Company-wide closures (week of July 4 and Dec 24 - Jan 1) that don't use PTO | time-off-and-leave |
| 7 | When is the on-call handoff and how much is the on-call stipend? | Tuesdays 10:00 AM ET; $400 per week | engineering-on-call |
| 8 | How much does the Ledgerlight Growth plan cost? | $199/month, up to 5,000 invoices and 15 users | product-roadmap-2026 |
| 9 | When does the Ledgerlight 3.2 code freeze start? | October 23, 2026 (release Nov 4) | product-roadmap-2026 |
| 10 | Can I deploy to Tollbooth on a Friday afternoon? | No: deploys freeze Fridays after 2 PM ET, and Tollbooth deploys need 2 reviews and a 10 AM - 3 PM ET window | engineering-on-call |

## Suggested demo flow (3 minutes)

1. Open ChatGPT and ask question 3 ("What is Project Kestrel and when does it launch?").
   It doesn't know, or it describes something unrelated.
2. Ask the same question in your app. You get the exact date and details, citation [1] pointing to the roadmap,
   and a high groundedness score.
3. Ask question 4 (INC-2291). Point out the `bm25` tag on the source: keyword search matched the
   exact incident ID, which pure vector search often misses. That's why the project uses hybrid retrieval.
4. Ask "What is Quillfeather's policy on bringing pets to the office?" The documents don't cover it, so your app
   refuses instead of guessing. That's the relevance guardrail.

## Talking point

"LLMs only know their public training data. Companies need answers from their own private
documents, and they need proof of where each answer came from. RAG gives the model the right
documents at question time, cites them, and my guardrails check that the answer actually matches them."
