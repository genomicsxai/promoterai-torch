"""
Read the variables of a TF2 Keras SavedModel with numpy alone (no TensorFlow).

A SavedModel directory stores its weights in two places that, together, give
every variable's Keras name and value:

- `saved_model.pb`: a `SavedModel` proto whose `SavedObjectGraph` lists every
  tracked object. Variable nodes carry a `SavedVariable` with the Keras
  variable name (e.g. "meta_former_block/dense/kernel:0").
- `variables/variables.index` + `variables/variables.data-*`: a TF tensor
  bundle. The index is a LevelDB-format SSTable mapping checkpoint keys
  (e.g. "layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE") to
  `BundleEntryProto`s (dtype, shape, shard, offset, size); the data shards hold
  the raw little-endian tensor bytes. Its `_CHECKPOINTABLE_OBJECT_GRAPH` entry is
  a `TrackableObjectGraph` proto whose node ids are the same as the
  SavedObjectGraph's, and which names each variable node's checkpoint key.

Only the handful of protobuf fields needed for that are decoded, with a small
wire-format reader rather than generated proto classes. Uncompressed index
blocks and unpartitioned variables are supported, which is what Keras writes;
anything else raises rather than being guessed at.

Verified bitwise-identical to `tf_keras.models.load_model(path).weights` on all
four published Illumina PromoterAI SavedModels (hg38, hg38_finetune, hg38_mm10,
hg38_mm10_finetune), including the first-appearance order of the
`shortcut_{species}{N}` heads that `convert_tf_weights` records as
`species_order`.
"""

from __future__ import annotations

import os
import struct

import numpy as np

# ── protobuf wire format ─────────────────────────────────────────────────────


def _varint(buf, i: int) -> tuple[int, int]:
    """Decode a base-128 varint at buf[i]; return (value, next index)."""
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def _fields(buf) -> dict[int, list]:
    """Decode one protobuf message into {field number: [raw values]}.

    Varint fields decode to int; length-delimited fields (strings, bytes,
    sub-messages, packed repeats) stay as bytes for the caller to interpret.
    """
    out: dict[int, list] = {}
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        num, wire_type = key >> 3, key & 7
        if wire_type == 0:
            value, i = _varint(buf, i)
        elif wire_type == 1:
            value, i = bytes(buf[i : i + 8]), i + 8
        elif wire_type == 2:
            length, i = _varint(buf, i)
            value, i = bytes(buf[i : i + length]), i + length
        elif wire_type == 5:
            value, i = bytes(buf[i : i + 4]), i + 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire_type}")
        out.setdefault(num, []).append(value)
    return out


def _first(msg: dict, num: int, default=None):
    values = msg.get(num)
    return values[0] if values else default


# Field numbers, from tensorflow/core/protobuf/*.proto.
_SAVED_MODEL_META_GRAPHS = 2  # SavedModel.meta_graphs
_META_GRAPH_OBJECT_GRAPH = 7  # MetaGraphDef.object_graph_def
_OBJECT_GRAPH_NODES = 1  # SavedObjectGraph.nodes / TrackableObjectGraph.nodes
_SAVED_OBJECT_CHILDREN = 1  # SavedObject.children / TrackableObject.children
_REFERENCE_NODE_ID, _REFERENCE_LOCAL_NAME = 1, 2  # ObjectReference
_SAVED_OBJECT_VARIABLE = 7  # SavedObject.variable
_SAVED_VARIABLE_NAME = 6  # SavedVariable.name
_TRACKABLE_ATTRIBUTES = 2  # TrackableObject.attributes
_ATTRIBUTE_NAME = 1  # SerializedTensor.name
_ATTRIBUTE_CHECKPOINT_KEY = 3  # SerializedTensor.checkpoint_key
_HEADER_NUM_SHARDS = 1  # BundleHeaderProto.num_shards
_ENTRY_DTYPE, _ENTRY_SHAPE, _ENTRY_SHARD = 1, 2, 3  # BundleEntryProto
_ENTRY_OFFSET, _ENTRY_SIZE, _ENTRY_SLICES = 4, 5, 7
_SHAPE_DIM, _DIM_SIZE = 2, 1  # TensorShapeProto.dim, .Dim.size

# ── tensor-bundle index (LevelDB SSTable) ────────────────────────────────────

_TABLE_MAGIC = 0xDB4775248B80FB57
_FOOTER_LENGTH = 48


def _block_entries(block: bytes):
    """Yield (key, value) pairs from one SSTable block (prefix-compressed keys)."""
    num_restarts = struct.unpack_from("<I", block, len(block) - 4)[0]
    end = len(block) - 4 - 4 * num_restarts
    i, key = 0, b""
    while i < end:
        shared, i = _varint(block, i)
        non_shared, i = _varint(block, i)
        value_length, i = _varint(block, i)
        key = key[:shared] + block[i : i + non_shared]
        i += non_shared
        yield key, block[i : i + value_length]
        i += value_length


def _read_block(raw: bytes, handle: bytes, path: str) -> bytes:
    """Read the block a BlockHandle (offset varint, size varint) points to."""
    offset, i = _varint(handle, 0)
    size, _ = _varint(handle, i)
    # Each block is followed by a 1-byte compression type and a 4-byte CRC.
    if raw[offset + size] != 0:
        raise ValueError(f"{path}: compressed index blocks are not supported")
    return raw[offset : offset + size]


def _read_table(path: str) -> dict[bytes, bytes]:
    """Read every entry of a tensor-bundle index file."""
    with open(path, "rb") as handle:
        raw = handle.read()
    if (
        len(raw) < _FOOTER_LENGTH
        or struct.unpack_from("<Q", raw, len(raw) - 8)[0] != _TABLE_MAGIC
    ):
        raise ValueError(f"{path} is not a TensorFlow tensor-bundle index")
    footer = raw[-_FOOTER_LENGTH:]
    _, i = _varint(footer, 0)  # metaindex block offset (unused)
    _, i = _varint(footer, i)  # metaindex block size (unused)
    entries = {}
    for _, data_handle in _block_entries(_read_block(raw, footer[i:], path)):
        entries.update(_block_entries(_read_block(raw, data_handle, path)))
    return entries


# ── tensor bundle ────────────────────────────────────────────────────────────

# DataType enum (tensorflow/core/framework/types.proto) -> numpy dtype.
_NUMPY_DTYPES = {
    1: "<f4",  # DT_FLOAT
    2: "<f8",  # DT_DOUBLE
    3: "<i4",  # DT_INT32
    4: "u1",  # DT_UINT8
    5: "<i2",  # DT_INT16
    6: "i1",  # DT_INT8
    9: "<i8",  # DT_INT64
    10: "?",  # DT_BOOL
    17: "<u2",  # DT_UINT16
    19: "<f2",  # DT_HALF
    22: "<u4",  # DT_UINT32
    23: "<u8",  # DT_UINT64
}
_DT_STRING = 7


class _TensorBundle:
    """Random access to the tensors of `<prefix>.index` + `<prefix>.data-*`."""

    def __init__(self, prefix: str):
        self.prefix = prefix
        table = _read_table(prefix + ".index")
        header = _fields(table.pop(b"", b""))  # "" holds the BundleHeaderProto
        self.num_shards = _first(header, _HEADER_NUM_SHARDS, 1)
        self.entries = {key.decode(): _fields(value) for key, value in table.items()}
        self._shards: dict[int, np.memmap] = {}

    def _shard(self, shard_id: int) -> np.memmap:
        if shard_id not in self._shards:
            path = f"{self.prefix}.data-{shard_id:05d}-of-{self.num_shards:05d}"
            self._shards[shard_id] = np.memmap(path, dtype="u1", mode="r")
        return self._shards[shard_id]

    def read(self, key: str):
        """Return the tensor at checkpoint key as an ndarray (or bytes for a scalar string)."""
        entry = self.entries[key]
        if entry.get(_ENTRY_SLICES):
            raise ValueError(f"{key}: partitioned variables are not supported")
        dtype = _first(entry, _ENTRY_DTYPE)
        shape = tuple(
            _first(_fields(dim), _DIM_SIZE, 0)
            for dim in _fields(_first(entry, _ENTRY_SHAPE, b"")).get(_SHAPE_DIM, [])
        )
        offset = _first(entry, _ENTRY_OFFSET, 0)
        size = _first(entry, _ENTRY_SIZE, 0)
        data = self._shard(_first(entry, _ENTRY_SHARD, 0))[offset : offset + size]

        if dtype == _DT_STRING:
            # Varint length of every element, a CRC32C of the lengths, then the bytes.
            data = data.tobytes()
            lengths, i = [], 0
            for _ in range(int(np.prod(shape)) if shape else 1):
                length, i = _varint(data, i)
                lengths.append(length)
            i += 4
            strings = []
            for length in lengths:
                strings.append(data[i : i + length])
                i += length
            return strings if shape else strings[0]
        if dtype not in _NUMPY_DTYPES:
            raise ValueError(f"{key}: unsupported tensor dtype (DataType enum {dtype})")
        return np.frombuffer(data, dtype=_NUMPY_DTYPES[dtype]).reshape(shape).copy()


# ── public API ───────────────────────────────────────────────────────────────


def _children(nodes: list[dict], node_id: int):
    """Yield (child node id, local name) for each child edge of a SavedObject."""
    for ref in nodes[node_id].get(_SAVED_OBJECT_CHILDREN, []):
        ref = _fields(ref)
        yield _first(ref, _REFERENCE_NODE_ID, 0), _first(ref, _REFERENCE_LOCAL_NAME, b"")


# Root children whose variables are not model weights: the optimizer's slot
# variables (e.g. "AdamW/m/dense/kernel") and step counter, and compiled
# metrics' accumulators ("total", "count") under keras_api/metrics.
_NON_WEIGHT_ROOT_CHILDREN = {b"optimizer", b"keras_api"}


def _model_variable_node_ids(nodes: list[dict]) -> list[int]:
    """Node ids of the model's variables, in `keras_model.weights` order.

    Keras serializes `model.variables` as the root's `variables` child, whose
    children "0", "1", ... are the variables in order; use that when present.
    Otherwise fall back to every variable reachable from the root other than
    through the optimizer or compiled metrics, in node-id order.
    """
    is_variable = [_SAVED_OBJECT_VARIABLE in node for node in nodes]
    root = dict((name, child) for child, name in _children(nodes, 0))
    if b"variables" in root:
        listed = sorted(
            ((int(name), child) for child, name in _children(nodes, root[b"variables"])
             if name.isdigit()),
        )
        ids = [child for _, child in listed]
        if ids and all(is_variable[i] for i in ids):
            return ids

    seen = {0}
    stack = [c for c, name in _children(nodes, 0) if name not in _NON_WEIGHT_ROOT_CHILDREN]
    while stack:
        node_id = stack.pop()
        if node_id not in seen:
            seen.add(node_id)
            stack.extend(child for child, _ in _children(nodes, node_id))
    return [i for i in sorted(seen) if is_variable[i]]


def read_savedmodel_variables(path: str | os.PathLike) -> dict[str, np.ndarray]:
    """Read every variable of a TF2 Keras SavedModel directory, without TensorFlow.

    Returns {Keras variable name without the ":0" suffix: ndarray} for the
    model's own variables -- the same set and order as
    `tf_keras.models.load_model(path).weights`, without the optimizer's slot
    variables or compiled metrics' accumulators that a compiled model's
    SavedModel also stores. The order matters to `convert_tf_weights`, which
    takes `species_order` from the first appearance of each output head.
    """
    path = os.fspath(path)
    pb_path = os.path.join(path, "saved_model.pb")
    if not os.path.isfile(pb_path):
        raise FileNotFoundError(f"{path} is not a SavedModel directory (no saved_model.pb)")
    with open(pb_path, "rb") as handle:
        saved_model = _fields(handle.read())

    meta_graphs = saved_model.get(_SAVED_MODEL_META_GRAPHS, [])
    if not meta_graphs:
        raise ValueError(f"{pb_path} contains no MetaGraphDef")
    object_graph = _first(_fields(meta_graphs[0]), _META_GRAPH_OBJECT_GRAPH)
    if object_graph is None:
        raise ValueError(
            f"{pb_path} has no object graph; only TF2 (Keras) SavedModels are supported"
        )
    nodes = [_fields(node) for node in _fields(object_graph).get(_OBJECT_GRAPH_NODES, [])]

    bundle = _TensorBundle(os.path.join(path, "variables", "variables"))
    checkpoint_graph = _fields(bundle.read("_CHECKPOINTABLE_OBJECT_GRAPH"))
    checkpoint_nodes = checkpoint_graph.get(_OBJECT_GRAPH_NODES, [])

    weights: dict[str, np.ndarray] = {}
    for node_id in _model_variable_node_ids(nodes):
        variable = _first(nodes[node_id], _SAVED_OBJECT_VARIABLE)
        name = _first(_fields(variable), _SAVED_VARIABLE_NAME, b"").decode()
        name = name.removesuffix(":0")
        if node_id >= len(checkpoint_nodes):
            raise ValueError(f"Variable {name!r} has no entry in the checkpoint object graph")
        attributes = [
            _fields(a)
            for a in _fields(checkpoint_nodes[node_id]).get(_TRACKABLE_ATTRIBUTES, [])
        ]
        keys = [
            _first(a, _ATTRIBUTE_CHECKPOINT_KEY).decode()
            for a in attributes
            if _first(a, _ATTRIBUTE_NAME) == b"VARIABLE_VALUE"
        ]
        if not keys:
            raise ValueError(f"Variable {name!r} has no VARIABLE_VALUE in the checkpoint")
        if name in weights:
            raise ValueError(f"Duplicate variable name {name!r} in {path}")
        weights[name] = bundle.read(keys[0])

    if not weights:
        raise ValueError(f"No variables found in {path}")
    return weights
