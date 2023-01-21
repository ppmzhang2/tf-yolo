import csv
from typing import List

__all__ = ['CocoAnnotation']


class CocoAnnotation:

    @staticmethod
    def _trans_imagetag_dict(dc: dict):
        return {
            'imageid': dc['id'],
            'name': dc['file_name'],
            'height': dc['height'],
            'width': dc['width'],
            'url': dc['coco_url'],
            'data': '',
        }

    @staticmethod
    def _trans_cate_dict(dc: dict):
        return {
            'cateid': dc['id'],
            'name': dc['name'],
        }

    @staticmethod
    def _trans_box_dict(dc: dict):
        return {
            'boxid': dc['id'],
            'imageid': dc['image_id'],
            'cateid': dc['category_id'],
            'box1': dc['bbox'][0],
            'box2': dc['bbox'][1],
            'box3': dc['bbox'][2],
            'box4': dc['bbox'][3],
        }

    @staticmethod
    def _dicts2csv(f, seq: List[dict], csvpath: str):
        seq_ = list(map(f, seq))
        headers = seq_[0].keys()

        with open(csvpath, "w", encoding='utf8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(seq_)

    @classmethod
    def imgtag2csv(cls, seq: List[dict], csvpath: str):
        return cls._dicts2csv(cls._trans_imagetag_dict, seq, csvpath)

    @classmethod
    def cate2csv(cls, seq: List[dict], csvpath: str):
        return cls._dicts2csv(cls._trans_cate_dict, seq, csvpath)

    @classmethod
    def box2csv(cls, seq: List[dict], csvpath: str):
        return cls._dicts2csv(cls._trans_box_dict, seq, csvpath)
