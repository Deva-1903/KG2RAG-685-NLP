================================================================================
KG²RAG EXPERIMENTAL PIPELINE - COMPREHENSIVE REPORT
================================================================================

SUMMARY TABLE
--------------------------------------------------------------------------------
Config                    Original             Experimental         Improvement    
--------------------------------------------------------------------------------
1000Q, First N              63.4% ( 43.4% exact)    65.1% ( 47.6% exact)            +1.7%

================================================================================
DETAILED RESULTS
================================================================================

Configuration: 1000Q, First N
Experiment folder: ../output/hotpot/exp_1000_first
--------------------------------------------------------------------------------
Original KG²RAG:
  Total Questions: 1000
  Overall Accuracy: 63.4%
  Exact Matches: 434 (43.4%)
  Partial Matches: 200 (20.0%)
  Yes/No Accuracy: 71.7% (60 questions)
  Factoid Accuracy: 62.9% (940 questions)

Experimental KG²RAG:
  Total Questions: 1000
  Overall Accuracy: 65.1%
  Exact Matches: 476 (47.6%)
  Partial Matches: 175 (17.5%)
  Yes/No Accuracy: 70.0% (60 questions)
  Factoid Accuracy: 64.8% (940 questions)

Improvement Analysis:
  Overall Accuracy: +1.7%
  Exact Match: +4.2%
  Factoid Accuracy: +1.9%
  Yes/No Accuracy: -1.7%

================================================================================

OVERALL STATISTICS
--------------------------------------------------------------------------------
Number of experiments: 1
Average improvement: +1.7%
Maximum improvement: +1.7%
Minimum improvement: +1.7%
