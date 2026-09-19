import xir
from collections import Counter

model = "yolo11n_bullet_target.xmodel"
graph = xir.Graph.deserialize(model)

ops = graph.get_ops()

counter = Counter(op.get_type() for op in ops)

print("Total XIR ops:", len(ops))
print()

for op_type, count in sorted(
    counter.items(),
    key=lambda x: (-x[1], x[0])
):
    print(f"{op_type:25s}: {count}")
