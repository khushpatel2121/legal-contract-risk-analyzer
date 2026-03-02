# Legal Contract Risk Analyzer

Fine-tuned Phi-3-mini on the CUAD dataset to identify legally-significant 
clauses in contracts and surface risk to non-expert users.

## Stack
| Component | Technology |
|-----------|------------|
| Base Model | Phi-3-mini-4k-instruct |
| Fine-tuning | QLoRA (4-bit) |
| Dataset | CUAD (41 clause types) |
| API | FastAPI |
| Frontend | Streamlit |
| Monitoring | Evidently AI |
| Training | Kaggle Notebooks |

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```