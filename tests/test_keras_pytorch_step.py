from tests.keras_pytorch_step import normalize_keras_outputs


def test_normalize_keras_outputs_prefers_species_order_over_name_heuristic():
    """convert_tf_weights' species_order (first-appearance order of shortcut_{species}{N}
    weights) is ground truth for output_heads' index order and need not be
    human/hg38-first. A dict whose keys would sort mouse-first under the name
    heuristic must still be reordered to match a given species_order that says
    otherwise, or a multi-species comparison silently pairs the wrong head's
    prediction with the wrong target/weights.
    """
    pred = {"mouse": "mouse_tensor", "human": "human_tensor"}

    pred_list, derived_species = normalize_keras_outputs(
        pred, species_order=("mouse", "human")
    )

    assert pred_list == ["mouse_tensor", "human_tensor"]
    assert derived_species == ("mouse", "human")


def test_normalize_keras_outputs_falls_back_to_heuristic_without_species_order():
    pred = {"mouse": "mouse_tensor", "human": "human_tensor"}

    pred_list, derived_species = normalize_keras_outputs(pred)

    assert pred_list == ["human_tensor", "mouse_tensor"]
    assert derived_species == ("human", "mouse")


def test_normalize_keras_outputs_falls_back_when_species_order_mismatches_keys():
    """A species_order that doesn't match the dict's actual keys (e.g. a checkpoint
    converted before species_order was recorded, passed a stale/wrong value) must
    not be trusted blindly -- fall back to the heuristic instead of silently
    reordering by something that doesn't correspond to these keys at all.
    """
    pred = {"mouse": "mouse_tensor", "human": "human_tensor"}

    pred_list, derived_species = normalize_keras_outputs(
        pred, species_order=("rat", "human")
    )

    assert pred_list == ["human_tensor", "mouse_tensor"]
    assert derived_species == ("human", "mouse")


def test_normalize_keras_outputs_passes_through_tuple():
    pred_list, derived_species = normalize_keras_outputs(("a", "b"))

    assert pred_list == ["a", "b"]
    assert derived_species is None
