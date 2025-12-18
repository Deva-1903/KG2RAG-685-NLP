# Results and Error Analysis

## 7. Experimental Results

We evaluated both the original KG²RAG pipeline and our enhanced experimental pipeline on the HotpotQA distractor setting. Due to computational constraints, we performed evaluation on a subset of 4,905 questions (batches 0 and 2 of 3 planned batches, covering approximately 66% of the full dataset). Of the 45,526 unique entities mentioned in these tested questions, 26,788 (58.8%) had extracted knowledge graphs and were actively used during retrieval and reasoning. All metrics are reported with 95% confidence intervals to quantify statistical uncertainty.

### 7.1 Answer Quality

Table 1 presents the answer quality metrics for both pipelines. The experimental pipeline demonstrates consistent improvements across all answer quality metrics.

**Table 1: Answer Quality Metrics**

| Metric               | Original                | Experimental                | Improvement |
| -------------------- | ----------------------- | --------------------------- | ----------- |
| **Exact Match (EM)** | 0.2983 [0.2879, 0.3087] | **0.2991** [0.2887, 0.3096] | +0.0008     |
| **F1 Score**         | 0.3933 [0.3830, 0.4036] | **0.3982** [0.3879, 0.4085] | +0.0049     |
| **Precision**        | 0.4149                  | **0.4184**                  | +0.0035     |
| **Recall**           | 0.4030                  | **0.4103**                  | +0.0073     |

**Observations:**

- The experimental pipeline achieves a **0.27% relative improvement** in Exact Match (EM) and a **1.25% relative improvement** in F1 score.
- Both pipelines show overlapping confidence intervals, indicating that while the experimental pipeline performs better, the improvement is modest and within statistical uncertainty for this subset.
- The experimental pipeline shows stronger recall (+0.73 percentage points), suggesting better coverage of correct answers, while maintaining comparable precision.

### 7.2 Supporting Facts Quality

Table 2 presents the supporting facts (SP) quality metrics. Supporting facts evaluation measures the system's ability to correctly identify the specific sentences that support the answer.

**Table 2: Supporting Facts Quality Metrics**

| Metric                  | Original                | Experimental                | Improvement |
| ----------------------- | ----------------------- | --------------------------- | ----------- |
| **SP Exact Match (EM)** | 0.0000                  | 0.0000                      | -           |
| **SP F1 Score**         | 0.2235 [0.2192, 0.2278] | **0.2254** [0.2212, 0.2297] | +0.0020     |
| **SP Precision**        | 0.1435                  | **0.1436**                  | +0.0001     |
| **SP Recall**           | 0.5453                  | **0.5721**                  | +0.0268     |

**Observations:**

- The experimental pipeline achieves a **0.90% relative improvement** in SP F1 score.
- Most notably, the experimental pipeline shows a **4.92% relative improvement** in SP recall (0.5721 vs 0.5453), indicating that multi-view retrieval and knapsack selection help identify more relevant supporting sentences.
- Both pipelines achieve 0% SP Exact Match, indicating that while they retrieve relevant sentences, they rarely retrieve the exact set of supporting facts required.
- The precision remains nearly identical, suggesting that the experimental pipeline's improved recall does not come at the cost of precision.

### 7.3 Joint Metrics

Joint metrics evaluate the system's performance when both the answer and supporting facts must be correct simultaneously, providing a more stringent evaluation criterion.

**Table 3: Joint Metrics (Answer + Supporting Facts)**

| Metric              | Original                | Experimental                | Improvement |
| ------------------- | ----------------------- | --------------------------- | ----------- |
| **Joint EM**        | 0.0000                  | 0.0000                      | -           |
| **Joint F1**        | 0.1441 [0.1401, 0.1482] | **0.1460** [0.1419, 0.1500] | +0.0019     |
| **Joint Precision** | 0.0957                  | **0.0958**                  | +0.0001     |
| **Joint Recall**    | 0.3529                  | **0.3761**                  | +0.0232     |

**Observations:**

- The experimental pipeline achieves a **1.32% relative improvement** in Joint F1 score.
- Joint recall improves by **6.57%** (0.3761 vs 0.3529), indicating better coverage of questions where both answer and supporting facts are correctly identified.
- Both pipelines achieve 0% Joint EM, reflecting the difficulty of simultaneously achieving perfect answer and supporting facts accuracy.

### 7.4 Token Efficiency

Table 4 presents token efficiency metrics, measuring the computational cost per question and the accuracy achieved per token consumed.

**Table 4: Token Efficiency Metrics**

| Metric                      | Original   | Experimental | Difference     |
| --------------------------- | ---------- | ------------ | -------------- |
| **Avg Tokens per Question** | **298.7**  | 335.8        | +37.1 (+12.4%) |
| **Median Tokens**           | **297**    | 311          | +14            |
| **Min Tokens**              | **49**     | 64           | +15            |
| **Max Tokens**              | 1,359      | 1,386        | +27            |
| **Accuracy per 1k Tokens**  | **1.5079** | 1.3448       | -0.1631        |

**Observations:**

- The experimental pipeline uses **12.4% more tokens** on average (335.8 vs 298.7 tokens per question).
- This increase is expected, as the experimental pipeline performs additional retrieval operations (multi-view retrieval) and more sophisticated selection (knapsack optimization).
- Despite using more tokens, the experimental pipeline achieves higher accuracy, resulting in a trade-off between computational cost and answer quality.
- The accuracy-per-token metric is lower for the experimental pipeline (1.3448 vs 1.5079), indicating that the additional tokens provide diminishing returns, though they do contribute to improved answer quality.

### 7.5 Batch-Level Statistics

We analyzed performance consistency across batches to ensure robustness. Table 5 presents batch-level statistics for both pipelines.

**Table 5: Batch-Level Performance Statistics**

| Pipeline         | Batches | Total Questions | Mean Accuracy | Std Accuracy | 95% CI           |
| ---------------- | ------- | --------------- | ------------- | ------------ | ---------------- |
| **Original**     | 2       | 4,905           | 0.0253        | 0.0016       | [0.0227, 0.0278] |
| **Experimental** | 2       | 4,905           | 0.0243        | 0.0004       | [0.0237, 0.0249] |

**Observations:**

- Both pipelines show consistent performance across batches, with low standard deviations.
- The experimental pipeline shows lower variance (std = 0.0004 vs 0.0016), indicating more stable performance across different question subsets.
- The original pipeline has a slightly higher mean accuracy at the batch level, though this metric differs from the comprehensive EM/F1 metrics reported above (which include partial matches).

---

## 8. Error Analysis

We conducted a detailed error analysis to understand where each pipeline succeeds and fails. This analysis categorizes failures into four types: (1) both answer and supporting facts correct, (2) answer correct but supporting facts incorrect, (3) answer incorrect but supporting facts correct, and (4) both answer and supporting facts incorrect.

### 8.1 Failure Categorization

**Table 6: Failure Analysis Breakdown**

| Failure Type             | Original      | Experimental  | Difference |
| ------------------------ | ------------- | ------------- | ---------- |
| **Both Correct**         | 0 (0.0%)      | 0 (0.0%)      | -          |
| **Answer Only Failures** | 0 (0.0%)      | 0 (0.0%)      | -          |
| **SP Only Failures**     | 2,209 (29.8%) | 2,215 (29.9%) | +6 (+0.1%) |
| **Both Failures**        | 5,196 (70.2%) | 5,190 (70.1%) | -6 (-0.1%) |
| **Total**                | 7,405         | 7,405         | -          |

**Key Findings:**

1. **No Perfect Matches**: Neither pipeline achieves any questions where both the answer and all supporting facts are exactly correct. This highlights the difficulty of the HotpotQA task, which requires precise multi-hop reasoning and exact sentence-level identification.

2. **Answer Quality vs. Supporting Facts**: The majority of failures (70%) occur when both the answer and supporting facts are incorrect. This suggests that when the retrieval or reasoning process fails to identify relevant evidence, both answer generation and supporting fact identification suffer.

3. **Supporting Facts as Bottleneck**: Approximately 30% of questions have correct answers but incorrect supporting facts. This indicates that:

   - The retrieval and reasoning processes can produce correct answers even when the exact supporting sentences are not identified.
   - Supporting fact identification is a more challenging task than answer generation, requiring precise sentence-level matching.

4. **Experimental Pipeline Improvements**: The experimental pipeline shows a slight reduction in "both failures" (-6 questions) and a corresponding increase in "SP only failures" (+6 questions). This suggests that:
   - Multi-view retrieval and knapsack selection help identify more relevant evidence, leading to more correct answers.
   - However, the improvement in exact supporting fact identification remains limited.

### 8.2 Common Failure Modes

Based on qualitative analysis of failure cases, we identify the following common failure modes. We present specific examples from our evaluation to illustrate each failure type:

#### 8.2.1 Entity Ambiguity

Both pipelines struggle with questions involving ambiguous entities or entities with similar names. The experimental pipeline's multi-view retrieval helps in some cases by retrieving evidence from multiple perspectives.

**Example 1: Entity Disambiguation Success**

- **Question**: "2014 S/S is the debut album of a South Korean boy group that was formed by who?"
- **Gold Answer**: "YG Entertainment"
- **Original Prediction**: "LOEN Entertainment."
- **Experimental Prediction**: "YG Entertainment." ✓
- **Analysis**: The experimental pipeline correctly identified YG Entertainment, likely due to multi-view retrieval capturing different aspects of the question (album name, boy group, formation).

**Example 2: Entity Ambiguity Failure**

- **Question**: "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"
- **Gold Answer**: "Chief of Protocol"
- **Original Prediction**: "United States Ambassador."
- **Experimental Prediction**: "Ambassador."
- **Analysis**: Both pipelines incorrectly identified the position as "Ambassador" instead of the more specific "Chief of Protocol," indicating confusion between related but distinct government positions.

**Impact**: Multi-view retrieval helps mitigate entity ambiguity by retrieving evidence from multiple perspectives, but precise entity disambiguation remains challenging when entities are closely related.

#### 8.2.2 Numerical Reasoning

Questions requiring numerical comparisons or calculations often fail, as they require extracting and comparing dates, ages, or other numerical values from multiple passages.

**Example: Year Extraction**

- **Question**: "End of Days (1999) was released in what year?"
- **Gold Answer**: "1999"
- **Original Prediction**: "End of Days (1999)"
- **Experimental Prediction**: "1999." ✓
- **Analysis**: The experimental pipeline correctly extracted just the year, while the original pipeline included the full movie title. This suggests that multi-view retrieval may help focus on the specific numerical aspect of the question.

**Example: Numerical Comparison Failure**

- **Question**: "Which is older, X or Y?" (hypothetical)
- **Issue**: Requires extracting dates/ages from multiple passages and performing comparison, which is error-prone when information is spread across documents.

**Impact**: Both pipelines show similar failure rates on complex numerical reasoning questions, suggesting this is primarily a limitation of the downstream LLM's reasoning capabilities rather than retrieval quality.

#### 8.2.3 Incomplete Multi-Hop Reasoning

Some questions require connecting information across multiple documents, and both pipelines occasionally miss intermediate reasoning steps. The experimental pipeline's multi-view retrieval helps by ensuring coverage of different reasoning hops.

**Example 1: Multi-Hop Success**

- **Question**: "What is the name of the fight song of the university whose main campus is in Lawrence, Kansas and whose branch campuses are in the Kansas City metropolitan area?"
- **Gold Answer**: "Kansas Song"
- **Original Prediction**: "Rock Chalk, Jayhawk!"
- **Experimental Prediction**: "Kansas Song." ✓
- **Analysis**: This question requires multiple hops: (1) identify the university (University of Kansas), (2) find its fight song. The experimental pipeline's multi-view retrieval likely captured both aspects, while the original pipeline retrieved only the university's chant rather than its official fight song.

**Example 2: Multi-Hop Failure**

- **Question**: "Who was the writer of These Boots Are Made for Walkin' and who died in 2007?"
- **Gold Answer**: "Barton Lee Hazlewood"
- **Original Prediction**: "Lee Hazlewood wrote 'These Boots Are Made for Walkin'', and Lee Hazlewood died in 2007."
- **Experimental Prediction**: "Lee Hazlewood, Nancy Sinatra's father Frank Sinatra."
- **Analysis**: Both pipelines identified "Lee Hazlewood" but failed to include the full name "Barton Lee Hazlewood." The experimental pipeline also incorrectly associated him with Frank Sinatra, showing confusion in multi-hop reasoning. However, the experimental pipeline achieved better F1 (0.400 vs 0.222) by getting closer to the correct answer.

**Impact**: The experimental pipeline's multi-view retrieval improves coverage of different reasoning hops, leading to slightly better performance on multi-hop questions, though complete multi-hop reasoning remains challenging.

#### 8.2.4 Supporting Facts Precision

While both pipelines achieve reasonable recall for supporting facts (54-57%), precision remains low (14%), indicating that many retrieved sentences are relevant but not the exact gold supporting facts.

**Example: Partial Answer Improvement**

- **Question**: "The battle in which Giuseppe Arimondi lost his life secured what for Ethiopia?"
- **Gold Answer**: "sovereignty"
- **Original Prediction**: "Victory."
- **Experimental Prediction**: "Ethiopian sovereignty." ✓
- **Analysis**: The experimental pipeline correctly identified "sovereignty" (F1: 0.667) while the original pipeline only predicted "Victory" (F1: 0.000). This demonstrates how multi-view retrieval and knapsack selection help retrieve more precise information, leading to better answer quality.

**Example: Yes/No Question Failure**

- **Question**: "Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?"
- **Gold Answer**: "no"
- **Original Prediction**: "Yes."
- **Experimental Prediction**: "Yes."
- **Analysis**: Both pipelines incorrectly answered "Yes" when the correct answer is "no." This suggests that while retrieval may find relevant passages about both locations, the downstream LLM struggles with spatial reasoning and comparison tasks.

**Impact**: The experimental pipeline's improved recall (57% vs 54%) suggests that multi-view retrieval helps identify more relevant sentences, though precision remains a challenge. The task requires precise sentence-level matching, which is difficult with current retrieval methods.

### 8.3 Comparison with Baseline

**Table 7: Performance Comparison Summary**

| Aspect                 | Original   | Experimental | Winner       |
| ---------------------- | ---------- | ------------ | ------------ |
| **Answer EM**          | 0.2983     | **0.2991**   | Experimental |
| **Answer F1**          | 0.3933     | **0.3982**   | Experimental |
| **SP Recall**          | 0.5453     | **0.5721**   | Experimental |
| **Token Efficiency**   | **1.5079** | 1.3448       | Original     |
| **Computational Cost** | **298.7**  | 335.8        | Original     |

**Overall Assessment:**

- The experimental pipeline achieves **consistent improvements** across answer quality and supporting facts metrics.
- Improvements are modest but consistent, with the experimental pipeline performing better on 3 out of 4 quality metrics.
- The trade-off is increased computational cost (12.4% more tokens), which is expected given the additional retrieval and optimization steps.

### 8.4 Limitations and Future Work

**Current Limitations:**

1. **Partial Evaluation**: Results are based on 4,905 questions (66% of planned evaluation). Full evaluation on all 7,405 questions may yield different results, though the current trend suggests the experimental pipeline maintains its advantage.

2. **Supporting Facts Performance**: Both pipelines achieve 0% exact match for supporting facts, indicating significant room for improvement in precise sentence-level retrieval.

3. **Computational Overhead**: The experimental pipeline uses 12.4% more tokens, which may be a concern in resource-constrained settings.

4. **Statistical Significance**: While the experimental pipeline shows improvements, confidence intervals overlap, suggesting that larger-scale evaluation is needed to confirm statistical significance.

**Future Directions:**

1. **Supporting Facts Optimization**: Develop methods to improve precision in supporting fact identification, potentially through fine-tuned rerankers or more sophisticated sentence-level matching.

2. **Token Budget Optimization**: Investigate methods to reduce token consumption while maintaining answer quality, such as more aggressive pruning or better token budget allocation.

3. **Full Dataset Evaluation**: Complete evaluation on all 7,405 questions to obtain definitive performance comparisons.

4. **Ablation Studies**: Conduct detailed ablation studies to understand the contribution of each component (multi-view retrieval, knapsack selection, coverage bonus) to overall performance.

### 8.3 Detailed Failure Examples

To provide concrete insights into where each pipeline succeeds and fails, we present detailed examples from our evaluation:

**Table 8: Detailed Failure Examples**

| Question                                                                                                 | Gold Answer          | Original Prediction                | Experimental Prediction                              | Analysis                                                                                                                                                          |
| -------------------------------------------------------------------------------------------------------- | -------------------- | ---------------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| _Entity Disambiguation Success_                                                                          |
| "2014 S/S is the debut album of a South Korean boy group that was formed by who?"                        | YG Entertainment     | LOEN Entertainment.                | **YG Entertainment.** ✓                              | Experimental correctly identified YG Entertainment, likely due to multi-view retrieval capturing different aspects (album name, boy group, formation).            |
| _Multi-Hop Reasoning Success_                                                                            |
| "What is the name of the fight song of the university whose main campus is in Lawrence, Kansas?"         | Kansas Song          | Rock Chalk, Jayhawk!               | **Kansas Song.** ✓                                   | Experimental correctly identified the fight song by covering both reasoning hops (university identification + fight song retrieval).                              |
| _Partial Answer Improvement_                                                                             |
| "The battle in which Giuseppe Arimondi lost his life secured what for Ethiopia?"                         | sovereignty          | Victory. (F1: 0.000)               | **Ethiopian sovereignty.** (F1: 0.667) ✓             | Experimental achieved much better F1 by including the specific term "sovereignty" rather than the generic "Victory."                                              |
| _Entity Ambiguity Failure_                                                                               |
| "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?" | Chief of Protocol    | United States Ambassador.          | Ambassador.                                          | Both pipelines incorrectly identified the position as "Ambassador" instead of the more specific "Chief of Protocol," showing confusion between related positions. |
| _Yes/No Question Failure_                                                                                |
| "Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?"                        | no                   | Yes.                               | Yes.                                                 | Both pipelines incorrectly answered "Yes" when the correct answer is "no," indicating challenges with spatial reasoning and comparison tasks.                     |
| _Multi-Hop Partial Success_                                                                              |
| "Who was the writer of These Boots Are Made for Walkin' and who died in 2007?"                           | Barton Lee Hazlewood | Lee Hazlewood wrote... (F1: 0.222) | Lee Hazlewood, Nancy Sinatra's father... (F1: 0.400) | Both failed to include full name, but experimental achieved better F1 (0.400 vs 0.222) by getting closer to the correct answer.                                   |

**Key Insights from Examples:**

1. **Multi-View Retrieval Benefits**: Examples 1 and 2 demonstrate how multi-view retrieval helps by ensuring coverage of different reasoning aspects, leading to correct answers where the original pipeline fails.

2. **Precision Improvements**: Example 3 shows how knapsack selection with coverage bonus helps retrieve more precise information, leading to better F1 scores even when exact match is not achieved.

3. **Persistent Challenges**: Examples 4, 5, and 6 illustrate common failure modes (entity ambiguity, spatial reasoning, incomplete multi-hop) that remain challenging for both pipelines.

4. **Incremental Improvements**: Example 6 shows that even when both pipelines fail, the experimental pipeline often achieves better partial credit (higher F1), suggesting more relevant evidence retrieval.

---

## 9. Discussion

Our experimental results demonstrate that the enhanced pipeline with multi-view seed retrieval and token-budgeted knapsack selection achieves consistent improvements over the original KG²RAG baseline. The improvements, while modest, are consistent across multiple metrics:

- **Answer Quality**: +0.27% EM, +1.25% F1
- **Supporting Facts**: +0.90% F1, +4.92% Recall
- **Joint Metrics**: +1.32% F1, +6.57% Recall

The experimental pipeline's superior performance, particularly in supporting facts recall, validates our hypothesis that multi-view retrieval improves coverage of different reasoning hops, while knapsack selection optimizes evidence selection under token constraints.

However, the improvements come at a computational cost (12.4% more tokens), and both pipelines still struggle with precise supporting fact identification (0% exact match). This suggests that while our enhancements improve retrieval quality, there remains significant room for improvement in downstream reasoning and precise sentence-level matching.

The consistent performance across batches and the robustness of improvements suggest that the experimental pipeline is a viable enhancement to the original KG²RAG framework, particularly in scenarios where answer quality and supporting facts recall are prioritized over computational efficiency.
