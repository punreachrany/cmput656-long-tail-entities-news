# A Fine-Grained Study of Long-Tail Entities in News Articles

## Project Overview

TODO

---

## Dataset

- **Signal-1M (Signal Media One Million News Articles)**  
  A large-scale news corpus commonly used for entity-centric research.  
  https://research.signal-ai.com/newsir16/signal-dataset.html

---

## 🧠 Methods

### Named Entity Recognition (NER)
- **General:** spaCy, Stanza  
- **Latest:** NameTag 3  

### Entity Linking (EL)
- **BLINK (by Meta AI)**  
  Used to link detected entities to Wikipedia, based on fine-tuned BERT architectures
  https://github.com/facebookresearch/BLINK

---

## Output

- Detection of long-tail entities in news articles  
- Construction of a **fine-grained entity taxonomy**

---

## 📚 Related Work

- Esquivel, J., Albakour, D., Martinez, M., Corney, D., & Moussa, S.  
  *On the Long-Tail Entities in News*

---

## Tools

- Python  
- spaCy, Stanza, NameTag 3  
- BLINK (Meta AI)