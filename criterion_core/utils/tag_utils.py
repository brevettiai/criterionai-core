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

