def get_config():
	input_params = {}
	input_params['map1'] = "grid"
	input_params['map2'] = "jelly"
	input_params['map3'] = "l"
	input_params['maps_train'] = {0: ["grid", "jelly"], 1: ["jelly", "l"], 2: ["grid", "l"]}
	input_params['map_test'] = {0: "l", 1: "grid", 2: "jelly"}
	input_params['max_epochs'] = 2
	input_params['seed'] = 12345
	input_params['datafile_path'] = "./data/"
	input_params['save_filepath'] = "./tracks/"
	input_params['word_embedding'] = "./data/WordEmbeddings.bin"
	input_params['folds'] = 3

	return input_params


def get_model_config():
	model_params = {}
	model_params['hidden_size'] = 128
	model_params['optimizer'] = "Adam"
	model_params['dropout_rate'] = 0.9
	model_params['learning_rate'] = 0.01
	model_params['beam_size'] = 1
	model_params['dim_lang'] = 524
	model_params['dim_world'] = 78  # pre-defined world representations : 6 + 4 * (8+3+6+1) = 78
	model_params['dim_action'] = 4  # forward, left, right, stop

	return model_params
