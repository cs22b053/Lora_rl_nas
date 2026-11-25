class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'cityscapes':
            # Return the absolute path to your cityscapes folder
            return '/scratch/ajin/Autodeeplab/autodeeplab/cityscapes/'
        elif dataset == 'pascal':
            return '/scratch/ajin/Autodeeplab/autodeeplab/pascal/'
        elif dataset == 'sbd':
            return '/scratch/ajin/Autodeeplab/autodeeplab/sbd/'
        elif dataset == 'coco':
            return '/scratch/ajin/Autodeeplab/autodeeplab/coco/'
        else:
            print(f'Dataset {dataset} not available.')
            raise NotImplementedError