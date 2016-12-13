{
	# ARCHITECTURE
	'cortex_size' : 1.0,
	'lgn_size' : 1.5,
	'visual_field_size' : 3.5,
	'cortex_density' : 96,
	'settle_steps' : 16,
	'num_gaussian_stim' : 2,


	# LGN
	'retina_to_lgn_strength' : 1.5,
	'lateral_lgn_strength' : 0.6,
	'lateral_lgn_gain' : 0.11,
	'lgn_center_sigma' : 0.07385,
	'lgn_surround_sigma' : 0.2954,
	'lgn_lateral_sigma' : 0.25,

	# Homeostasis
	'homeostatic_learning_rate'  : 0.01,
	'target_activity' : 0.24,
	'activity_average_smoothing' : 0.991,
	
	# Hebbian adaptation
	'learning_rate' : 0.2,


	# time constant Model
	'membrane_time_constant_E' : 2, #ms
	'membrane_time_constant_I' : 0.5, #ms
}