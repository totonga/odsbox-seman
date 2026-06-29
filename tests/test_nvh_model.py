from odsbox.proto.ods_pb2 import Model


def test_nvh_model(nvh_model: Model) -> None:
    """Test the NVH model."""
    assert nvh_model is not None
    assert nvh_model.entities.get("Project") is not None
