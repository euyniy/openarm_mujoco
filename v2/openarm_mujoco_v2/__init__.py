from importlib.resources import files
from pathlib import Path

from .joint_resolver import JointResolver

__all__ = ["JointResolver"]

def asset_path(relative: str) -> str:
    """
    Return an absolute filesystem path to an asset inside this package.
    Example: asset_path("openarm_bimanual.xml")
    """
    p = files("openarm_mujoco_v2").joinpath(relative)
    return str(Path(p))


def openarm_bimanual_paths() -> list[str]:
    """
    Returns the list of the absolute path to bimanual file and
    the other required files/directories.
    """
    return [
        asset_path("openarm_v20_bimanual.xml"),
        asset_path("assets"),
    ]

def openarm_cell_xml() -> str:
    return asset_path("cell.xml")

def openarm_pedestal_xml() -> str:
    return asset_path("pedestal.xml")

def openarm_bimanual_xml() -> str:
    return asset_path("openarm_v20_bimanual.xml")
