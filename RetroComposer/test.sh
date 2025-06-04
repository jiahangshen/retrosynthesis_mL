python extract_templates.py   
python prepare_mol_graph.py --retro   
python run_retro.py  --device 0 > doml_train_stage1.log 2>&1 &
python chem_doml.py  --device 0 --input_model_file model1.pt --test_only
python chem_doml.py  --device 0 --input_model_file model0_1.pt --test_only
python chem_doml.py  --device 0 --input_model_file model0_5.pt --test_only
python chem_doml.py  --device 0 --input_model_file model0.pt --test_only



# run validation split
python run_retro.py  --device 0  --input_model_file model.pt --test_only  --eval_split valid 

# run training split
python run_retro.py  --device 0   --input_model_file model.pt --test_only  --eval_split train
python prepare_mol_graph.py
python run_ranking.py  --device 0
python run_ranking.py  --device 0 --test_only --input_model_file model.pt
