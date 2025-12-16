================================================================================
KG²RAG EXPERIMENTAL PIPELINE - COMPREHENSIVE REPORT
================================================================================

SUMMARY TABLE
--------------------------------------------------------------------------------
Config                    Original             Experimental         Improvement    
--------------------------------------------------------------------------------
100Q, First N               62.0% ( 44.0% exact)    69.0% ( 48.0% exact)            +7.0%

================================================================================
DETAILED RESULTS
================================================================================

Configuration: 100Q, First N
Experiment folder: ../output/hotpot/exp_100_first
--------------------------------------------------------------------------------
Original KG²RAG:
  Total Questions: 100
  Overall Accuracy: 62.0%
  Exact Matches: 44 (44.0%)
  Partial Matches: 18 (18.0%)
  Yes/No Accuracy: 66.7% (12 questions)
  Factoid Accuracy: 61.4% (88 questions)

Experimental KG²RAG:
  Total Questions: 100
  Overall Accuracy: 69.0%
  Exact Matches: 48 (48.0%)
  Partial Matches: 21 (21.0%)
  Yes/No Accuracy: 75.0% (12 questions)
  Factoid Accuracy: 68.2% (88 questions)

Improvement Analysis:
  Overall Accuracy: +7.0%
  Exact Match: +4.0%
  Factoid Accuracy: +6.8%
  Yes/No Accuracy: +8.3%

================================================================================

OVERALL STATISTICS
--------------------------------------------------------------------------------
Number of experiments: 1
Average improvement: +7.0%
Maximum improvement: +7.0%
Minimum improvement: +7.0%
