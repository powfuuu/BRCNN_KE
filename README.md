# Named Entity Recognition in Chinese Cyber Threat Intelligence Combining Biaffine Residual Convolution and Knowledge Expansion

Source code for paper: Named Entity Recognition in Chinese Cyber Threat Intelligence Combining Biaffine Residual Convolution and Knowledge Expansion

## 1.Environments

- python (3.8.0)
- cuda (11.4)

## 2. Dependencies

```bash
>> pip install -r requirements.txt
```

## 3. Dataset

- [CDTier](https://github.com/MuYu-z/CDTier/tree/main/data/Entity%20extraction)


## 4. Training(BRCNN)

```bash
>> python main.py --config ./config/cdtier.json
```

## 5. Knowledge Expansion(BRCNN+KE)

```bash
>> python main.py --config ./config/cdtier.json
>> python predict.py --config ./config/cdtier.json
>> python util/merge.py
>> python util/remove_duplicate.py
# Modify the parameter num_res_blocks in the main.py，predict.py file to 6
>> python main.py --config ./config/cdtier.json
```
