eval_cfg = {
    "dist_fcn": "center_distance",
    "dist_ths": {
        'Vehicle': [0.19, 0.95, 1.72],
        'Mattress': [0.08, 0.38, 0.69],
        'Ball': [0.03, 0.13, 0.23],
        'Gazebo': [0.29, 1.44, 2.59],
        'Canopy': [0.32, 1.6, 2.88],
        'Chair': [0.05, 0.24, 0.43],
        'Small_table': [0.06, 0.29, 0.52],
        'Big_table': [0.08, 0.41, 0.73],
        'Tripod_pipe': [0.06, 0.29, 0.52],
        'Barrel': [0.05, 0.26, 0.47],
        'Tent': [0.21, 1.07, 1.92],
        'Box': [0.05, 0.23, 0.41],
        'Drone_case': [0.07, 0.36, 0.64],
        'Small_bin': [0.03, 0.16, 0.29],
        'Ramp': [0.07, 0.34, 0.62],
        'Sitting_person': [0.06, 0.31, 0.57],
        'Standing_person': [0.05, 0.25, 0.45],
        'Cone': [0.01, 0.06, 0.11],
        'Flag': [0.06, 0.31, 0.56],
        'Pipe': [0.02, 0.1, 0.17],
        'Porta_potty': [0.11, 0.57, 1.02],
        'Play_tunnel': [0.03, 0.17, 0.3],
    }, # [min_dx_dy * [0.1, 0.5, 0.9]]
    "dist_th_tp": {
        'Vehicle': 0.95,
        'Mattress': 0.38,
        'Ball': 0.13,
        'Gazebo': 1.44,
        'Canopy': 1.6,
        'Chair': 0.24,
        'Small_table': 0.29,
        'Big_table': 0.41,
        'Tripod_pipe': 0.29,
        'Barrel': 0.26,
        'Tent': 1.07,
        'Box': 0.23,
        'Drone_case': 0.36,
        'Small_bin': 0.16,
        'Ramp': 0.34,
        'Sitting_person': 0.31,
        'Standing_person': 0.25,
        'Cone': 0.06,
        'Flag': 0.31,
        'Pipe': 0.1,
        'Porta_potty': 0.57,
        'Play_tunnel': 0.17,
    }, # min_dx_dy * 0.5
    "min_recall": 0.1,
    "min_precision": 0.1,
    "max_boxes_per_sample": 500,
    "mean_ap_weight": 3
}

DETECTION_NAMES = ['Vehicle', 'Mattress', 'Ball', 'Gazebo', 'Canopy', 'Chair', 'Small_table', 'Big_table',
                   'Tripod_pipe', 'Barrel', 'Tent', 'Box', 'Drone_case', 'Small_bin', 'Ramp', 'Sitting_person',
                   'Standing_person', 'Cone', 'Flag', 'Pipe', 'Porta_potty', 'Play_tunnel']
PI_ORIENTATION_NAMES = ['Mattress', 'Gazebo', 'Canopy', 'Small_table', 'Big_table', 'Box', 'Drone_case', 'Pipe',
                        'Play_tunnel']
NO_ORIENTATION_NAMES = ['Ball', 'Barrel', 'Small_bin', 'Cone', 'Flag']
TP_METRICS = ['trans_err', 'scale_err', 'orient_err']
