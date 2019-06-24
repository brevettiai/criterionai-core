from criterionai_tools.web_tools import CriterionWeb
from pandas import MultiIndex

def find(tree, key, value):
    items = tree.get("children", []) if isinstance(tree, dict) else tree
    for item in items:
        if item[key] == value:
            yield item
        else:
            yield from find(item, key, value)


def find_path(tree, key, value, path=()):
    items = tree.get("children", []) if isinstance(tree, dict) else tree
    for item in items:
        if item[key] == value:
            yield (*path, item)
        else:
            yield from find_path(item, key, value, (*path, item))

def get_dataset_tag_records(tag_connection_info, datasets):
    tag_parser = TagParser(tag_connection_info=tag_connection_info)
    ds_tag_records = {}
    ds_tag_fields = set()
    for ds in datasets:
        ds_tag_records[ds["id"]], fields = tag_parser.build_tag_records(ds["tags"])
        for rec in ds_tag_records[ds["id"]]:
            rec["Dataset"] = ds["name"]
        for ff in fields + [("Dataset", "Dataset")]:
            ds_tag_fields.add(ff)
    return ds_tag_records, list(ds_tag_fields)


class TagParser():
    def __init__(self, tag_connection_info):
        criterion = CriterionWeb(tag_connection_info.username, tag_connection_info.password)
        tags = criterion.get_tag()
        self.unfolded = self._unfold_tag_list(tags)

    def build_tag_records(self, ds_input_tags):
        tag_tree = self._build_tag_tree(ds_input_tags)

        keys = ["tag_" + kk.replace("-", "_") for kk in tag_tree.keys()]
        key_pretty_names = [self.unfolded[kk]["name"] for kk in tag_tree.keys()]

        values = [[self.unfolded[ki]["name"] for ki in vv.keys()] for vv in tag_tree.values()]

        indices = MultiIndex.from_product(iterables=values, names=keys).to_flat_index()
        tag_records = [None] * len(indices)
        for ii, ind in enumerate(indices):
            tag_records[ii] = {kk: ind[ii] for ii, kk in enumerate(keys)}
        return tag_records, [(kk, kn) for kk, kn in zip(keys, key_pretty_names)]

    def _unfold_tag_list(self, tag_list):
        dicts = {}
        for st in tag_list:
            dicts.update({tt["id"]: tt for tt in tag_list})
            dicts.update(self._unfold_tag_list(st["children"]))
        return dicts


    def _build_tag_tree(self, ds_input_tags):
        tree = {'root_id': {}}
        get_pair = lambda child: (child, self.unfolded[child]["parentId"])
        for tag in ds_input_tags:
            child, parent= get_pair(tag["id"])
            while parent is not None:
                tree.setdefault(parent, {})[child] = tree.setdefault(child, {})
                child, parent= get_pair(parent)
            tree["root_id"][child] = tree.setdefault(child, {})
        return tree["root_id"]
