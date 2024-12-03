import copy
import pickle
import os
from pathlib import Path
import numpy as np

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate


class Wisar3dDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = os.path.join(self.root_path, 'ImageSets', (self.split + '.txt'))
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if os.path.exists(split_dir) else None

        self.name_to_class = dataset_cfg.name_to_class

        self.wisar3d_infos = []
        self.include_data(self.mode)

    def include_data(self, mode):
        self.logger.info('Loading Wisar3d dataset.')
        wisar3d_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                wisar3d_infos.extend(infos)

        self.wisar3d_infos.extend(wisar3d_infos)
        self.logger.info('Total samples for Wisar3d dataset: %d' % (len(wisar3d_infos)))

    def get_label(self, idx):
        label_file = self.root_split_path / 'label' / ('%s.txt' % idx)
        assert label_file.exists(), f'Cant find label file: {label_file}'
        with open(label_file, 'r') as f:
            lines = f.readlines()

        # [N, 8]: (category_name x y z dx dy dz heading_angle)
        gt_boxes = []
        gt_names = []
        for line in lines:
            line_list = line.strip().split(' ')
            gt_names.append(line_list[0])
            gt_box = line_list[1:7] + [line_list[-1]] # take only rz
            gt_boxes.append(gt_box)
        return np.array(gt_boxes, dtype=np.float32), np.array(gt_names)

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'points' / ('%s.npy' % idx)
        assert lidar_file.exists()
        points_all = np.load(lidar_file)

        # normalize the intensity
        if points_all.shape[-1] > 3:
            points_all[:, 3] /= (2 ** 16)

        return points_all

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.sample_id_list) * self.total_epochs

        return len(self.wisar3d_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.wisar3d_infos)

        info = copy.deepcopy(self.wisar3d_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        points = self.get_lidar(sample_idx)
        input_dict = {
            'frame_id': self.sample_id_list[index],
            'points': points
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            gt_names = annos['name']
            gt_boxes_lidar = annos['gt_boxes_lidar']
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.wisar3d_infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def wisar3d_kitti_eval(eval_det_annos, eval_gt_annos):
            from ..kitti import kitti_utils
            from .wisar3d_kitti_eval import get_official_eval_result

            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            ap_result_str, ap_dict = get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=class_names, name_to_class=self.name_to_class
            )
            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.wisar3d_infos]

        def wisar3d_nuscenes_eval(eval_det_annos, eval_gt_annos):
            import json
            from .wisar3d_nuscenes_eval import DetectionConfig, NuScenesEval, format_nuscene_results
            from .wisar3d_eval_utils import eval_cfg

            output_path = Path(kwargs['output_path'])
            output_path.mkdir(exist_ok=True, parents=True)

            # Load config file and deserialize it.
            eval_config = DetectionConfig.deserialize(eval_cfg)

            map_name_to_class = {k: v + 1 for k, v in self.name_to_class.items()}

            nusc_eval = NuScenesEval(
                gt_annos=eval_gt_annos,
                det_annos=eval_det_annos,
                config=eval_config,
                output_dir=str(output_path),
                map_name_to_class=map_name_to_class,
                verbose=True,
            )
            metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)

            with open(output_path / 'metrics_summary.json', 'r') as f:
                metrics = json.load(f)

            result_str, result_dict = format_nuscene_results(metrics, self.class_names)
            result_dict['metrics_summary'] = metrics_summary
            return result_str, result_dict

        if kwargs['eval_metric'] == 'wisar3d_kitti':
            ap_result_str, ap_dict = wisar3d_kitti_eval(eval_det_annos, eval_gt_annos)
        elif kwargs['eval_metric'] == 'wisar3d_nuscenes':
            ap_result_str, ap_dict = wisar3d_nuscenes_eval(eval_det_annos, eval_gt_annos)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict

    def get_infos(self, class_names, num_workers=4, has_label=True, sample_id_list=None, num_features=4):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': num_features, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            if has_label:
                annotations = {}
                gt_boxes_lidar, name = self.get_label(sample_idx)
                annotations['name'] = name
                annotations['gt_boxes_lidar'] = gt_boxes_lidar[:, :7]
                info['annos'] = annotations

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list

        # create a thread pool to improve the velocity
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('wisar3d_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]

        # Output the num of all classes in database
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def create_label_file_with_name_and_box(class_names, gt_names, gt_boxes, save_label_path):
        with open(save_label_path, 'w') as f:
            for idx in range(gt_boxes.shape[0]):
                boxes = gt_boxes[idx]
                name = gt_names[idx]
                if name not in class_names:
                    continue
                line = "{x} {y} {z} {l} {w} {h} {angle} {name}\n".format(
                    x=boxes[0], y=boxes[1], z=(boxes[2]), l=boxes[3],
                    w=boxes[4], h=boxes[5], angle=boxes[6], name=name
                )
                f.write(line)


def create_wisar3d_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = Wisar3dDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split = 'train', 'val'
    num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

    train_filename = save_path / ('wisar3d_infos_%s.pkl' % train_split)
    val_filename = save_path / ('wisar3d_infos_%s.pkl' % val_split)

    print('------------------------Start to generate data infos------------------------')

    dataset.set_split(train_split)
    wisar3d_infos_train = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(wisar3d_infos_train, f)
    print('Wisar3d info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    wisar3d_infos_val = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(val_filename, 'wb') as f:
        pickle.dump(wisar3d_infos_val, f)
    print('Wisar3d info train file is saved to %s' % val_filename)

    print('------------------------Start create groundtruth database for data augmentation------------------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)
    print('------------------------Data preparation done------------------------')


if __name__ == '__main__':
    import sys

    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_wisar3d_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_wisar3d_infos(
            dataset_cfg=dataset_cfg,
            class_names=list(dataset_cfg.name_to_class.keys()),
            data_path=ROOT_DIR / 'data' / 'wisar3d',
            save_path=ROOT_DIR / 'data' / 'wisar3d',
        )

