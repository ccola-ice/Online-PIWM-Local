import re
import collections
import json

# input json path
json_path = '/home/gdg/InteractionRL/Dreamer_Inter/python/interaction_dreamerv3/cleanrl/run-sac_small_10-tag-episode_score.json'

# output jasonl path
jsonl_path = '/home/gdg/InteractionRL/Dreamer_Inter/python/interaction_dreamerv3/cleanrl/scores.jsonl'

output_bystep = collections.defaultdict(dict)
with open(json_path, 'r', encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        for data_point in data:
            step = data_point[1]
            value = data_point[2]
            output_bystep[step]['episode/score'] = float(value)

output_lines = ''.join([
    json.dumps({'step': step, **scalars}) + '\n'
    for step, scalars in output_bystep.items()])

with open(jsonl_path, 'a', encoding="utf-8") as f:
    f.write(output_lines)
