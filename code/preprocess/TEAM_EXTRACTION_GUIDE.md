# Team Extraction Guide - 5 Parts

## Quick Setup for Each Teammate

### Step 1: Split the Dataset (Do this ONCE, share the split files)

One person runs this to create 5 split files:

```bash
cd code/preprocess
python split_dataset.py
```

This creates:

- `data/hotpotqa/splits/hotpot_dev_distractor_v1_part1.json` (~1,481 questions)
- `data/hotpotqa/splits/hotpot_dev_distractor_v1_part2.json` (~1,481 questions)
- `data/hotpotqa/splits/hotpot_dev_distractor_v1_part3.json` (~1,481 questions)
- `data/hotpotqa/splits/hotpot_dev_distractor_v1_part4.json` (~1,481 questions)
- `data/hotpotqa/splits/hotpot_dev_distractor_v1_part5.json` (~1,481 questions)

**Share these split files with your team** (upload to Google Drive, etc.)

---

### Step 2: Each Teammate Runs Their Part

**Each person:**

1. **Open** `hotpot_extraction_part.py`

2. **Change this line** (line 18):

   ```python
   PART_NUMBER = 1  # Change to your assigned part (1, 2, 3, 4, or 5)
   ```

3. **Run**:
   ```bash
   python hotpot_extraction_part.py
   ```

That's it! The script will:

- Load your assigned part
- Extract KGs for all entities in that part
- Save to `data/hotpotqa/kgs/extract_subkgs/` (same directory for everyone)

---

## Team Assignment

| Teammate | Part Number | Questions | File to Process                       |
| -------- | ----------- | --------- | ------------------------------------- |
| Person 1 | 1           | ~1,481    | `hotpot_dev_distractor_v1_part1.json` |
| Person 2 | 2           | ~1,481    | `hotpot_dev_distractor_v1_part2.json` |
| Person 3 | 3           | ~1,481    | `hotpot_dev_distractor_v1_part3.json` |
| Person 4 | 4           | ~1,481    | `hotpot_dev_distractor_v1_part4.json` |
| Person 5 | 5           | ~1,481    | `hotpot_dev_distractor_v1_part5.json` |

---

## What Each Person Does

### Person 1:

```python
# In hotpot_extraction_part.py, line 18:
PART_NUMBER = 1
```

### Person 2:

```python
PART_NUMBER = 2
```

### Person 3:

```python
PART_NUMBER = 3
```

### Person 4:

```python
PART_NUMBER = 4
```

### Person 5:

```python
PART_NUMBER = 5
```

---

## Important Notes

### ✅ Safe Parallel Processing

- All teammates can run **at the same time**
- Each saves to the same `extract_subkgs/` directory
- If an entity appears in multiple parts, it's only processed once (skipped if file exists)
- No conflicts - safe to run in parallel!

### 📁 Output Location

- Everyone saves to: `data/hotpotqa/kgs/extract_subkgs/`
- Files are named by entity: `{Entity_Name}.json`
- No need to merge - just make sure all files are in the same directory

### ⏱️ Estimated Time

- **With GPU**: ~12-20 hours per part
- **CPU only**: ~35-55 hours per part
- Results save incrementally (safe to disconnect)

---

## After Everyone Finishes

1. **Collect all KG files** from `extract_subkgs/` directory
2. **Verify count**: Should have hundreds/thousands of `.json` files
3. **Use for full KG²RAG**: All files in `extract_subkgs/` are ready to use

---

## Troubleshooting

### "File not found" error

- Make sure you've downloaded the split files from your teammate
- Check the file is in: `data/hotpotqa/splits/hotpot_dev_distractor_v1_part{X}.json`

### "Ollama not running"

- Start Ollama: `ollama serve` (or use Colab setup)
- Verify: `ollama list`

### Progress seems slow

- Check if GPU is enabled (Colab: Runtime → Change runtime type → GPU)
- Each part takes 12-20 hours with GPU

---

## Example: Colab Setup

If using Google Colab:

1. **Upload split files** to Google Drive
2. **Update paths** in `hotpot_extraction_part.py`:
   ```python
   DATA_DIR = '/content/drive/MyDrive/685-Project/KG2RAG-main/data/hotpotqa/splits'
   OUT_DIR = '/content/drive/MyDrive/685-Project/KG2RAG-main/data/hotpotqa/kgs/extract_subkgs'
   ```
3. **Set PART_NUMBER** (line 18)
4. **Run the script**

---

## Summary

**One person**: Run `split_dataset.py` → Share split files

**Each teammate**:

1. Set `PART_NUMBER = X` (your part)
2. Run `python hotpot_extraction_part.py`
3. Done! ✅

**All files go to the same directory** - no merging needed!
