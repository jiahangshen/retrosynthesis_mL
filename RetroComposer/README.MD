# Environment
We include the comprehensive packages and versions of our environment in the requirements.txt file.
Note that not all packages in the file are necessary.
Here are some important packages and their versions:

```
rdkit-pypi                    2021.9.5.1
torch                         1.7.1+cu110
torch-geometric               1.7.2
torch-scatter                 2.0.6
torch-sparse                  0.6.9
```

We conduct retrosynthesis by composing templates and applying the templates to target molecules.


## Preprocessing
We need first extract reaction templates, and decompose each template into 
product smarts and reactant smarts which are later canonicalized to be used
as token. Run the following script to preprocesss the USPTO-50K dataset:
```
python extract_templates.py   
```

After the program ends, there will be four json files in the data path 
(data/USPTO50K). Please carefully check the scripts and code if you 
can not find these json files:
```
data/USPTO50K/templates_cano_train.json
data/USPTO50K/templates_test.json
data/USPTO50K/templates_train.json
data/USPTO50K/templates_valid.json
```

Then we generate training data:
```
python prepare_mol_graph.py --retro   
```


## Training


```
python chem_doml.py  --device 0
```

## Testing
```
python chem_doml.py  --device 1 --input_model_file model.pt --test_only
```
### Run testing on validation and training splits:
```
# run validation split
python run_retro.py  --device 1 --input_model_file model.pt --test_only  --eval_split valid 

# run training split
python run_retro.py  --device 1 --input_model_file model.pt --test_only  --eval_split train

# for with types setting
python run_retro.py  --device 1 --input_model_file model.pt --test_only  --eval_split valid --typed 
python run_retro.py  --device 1 --input_model_file model.pt --test_only  --eval_split train --typed
```

You can find three json files in the log directory:
```
logs/USPTO50K/uspto50k/beam_result_test.json
logs/USPTO50K/uspto50k/beam_result_train.json
logs/USPTO50K/uspto50k/beam_result_valid.json

# for with types setting
logs/USPTO50K/uspto50k_typed/beam_result_test.json
logs/USPTO50K/uspto50k_typed/beam_result_train.json
logs/USPTO50K/uspto50k_typed/beam_result_valid.json
```
results in Domain 1 ~ 10
```
Domain 1: 65.14% (6978/10712)
Domain 2: 65.60% (5669/8642)
Domain 3: 65.22% (2175/3335)
Domain 4: 65.97% (314/476)
Domain 5: 63.52% (296/466)
Domain 6: 63.72% (3765/5909)
Domain 7: 64.69% (2109/3260)
Domain 8: 64.55% (335/519)
Domain 9: 63.41% (752/1186)
Domain 10: 65.69% (90/137)

```
Path of pretrained checkpoint is 'logs/USPTO50K/uspto50k/retro_model.pt'


