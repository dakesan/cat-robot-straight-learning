"""Convert STEP geometry to metre-scale visual meshes without inferring hardware."""

import base64
import gzip
import hashlib
import importlib.metadata
import json
import math
import re
from pathlib import Path

import numpy as np

from .config import CadVisual, RobotConfig, write_json

ASSET_ROOT = Path(__file__).parent / "assets"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sanitized_step(data: bytes) -> bytes:
    """Redact export header fields and Shapr3D's temporary-file KEY property."""
    text = data.decode("utf-8")
    header, separator, body = text.partition("DATA;")
    schema = re.search(r"FILE_SCHEMA\s*\(.*?\);", header, re.DOTALL)
    if not separator or schema is None or not text.startswith("ISO-10303-21;"):
        raise ValueError("Expected a text STEP file with a FILE_SCHEMA and DATA section")
    body = re.sub(
        r"(DESCRIPTIVE_REPRESENTATION_ITEM\s*\(\s*'KEY'\s*,\s*)'(?:[^']|'')*'(\s*\)\s*;)",
        r"\1''\2",
        body,
    )
    return (
        "ISO-10303-21;\nHEADER;\n"
        "FILE_DESCRIPTION(('Cat robot chassis geometry'),'2;1');\n"
        "FILE_NAME('chassis.step','',(''),(''),'','','');\n"
        + schema.group()
        + "\nENDSEC;\nDATA;"
        + body
    ).encode("utf-8")


def convert_step(source: Path, destination: Path, deflection_mm: float = 0.15) -> dict:
    """Keep solid locations and names, tessellate in mm, then export vertices in m."""
    if destination.exists():
        raise ValueError("CAD destination already exists; choose a new directory")
    if not math.isfinite(deflection_mm) or deflection_mm <= 0:
        raise ValueError("Tessellation deflection must be positive and finite")
    try:
        from OCP.Bnd import Bnd_Box
        from OCP.BRep import BRep_Tool
        from OCP.BRepBndLib import BRepBndLib
        from OCP.BRepCheck import BRepCheck_Analyzer
        from OCP.BRepGProp import BRepGProp
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.GProp import GProp_GProps
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_Reader
        from OCP.TColStd import TColStd_SequenceOfAsciiString
        from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SOLID
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopLoc import TopLoc_Location
        from OCP.TopoDS import TopoDS
    except ImportError as error:
        raise ValueError("STEP conversion requires Python 3.12-3.14 and uv sync --group cad") from error

    content = sanitized_step(source.read_bytes())
    destination.mkdir(parents=True)
    step = destination / "chassis.step"
    step.write_bytes(content)
    reader = STEPControl_Reader()
    if reader.ReadFile(str(step)) != IFSelect_RetDone:
        raise ValueError("Open CASCADE could not read the STEP file")
    length, angle, solid_angle = (TColStd_SequenceOfAsciiString() for _ in range(3))
    reader.FileUnits(length, angle, solid_angle)
    units = [length.Value(i).ToCString() for i in range(1, length.Length() + 1)]
    reader.SetSystemLengthUnit(1.0)
    if not reader.TransferRoots():
        raise ValueError("STEP file has no transferable roots")
    shape = reader.OneShape()
    if not BRepCheck_Analyzer(shape).IsValid():
        raise ValueError("STEP boundary representation is invalid")
    mesher = BRepMesh_IncrementalMesh(shape, deflection_mm, False, 0.15, False)
    if not mesher.IsDone():
        raise ValueError("STEP tessellation failed")
    solids = []
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    while explorer.More():
        solid = explorer.Current()
        bbox = Bnd_Box()
        BRepBndLib.AddOptimal_s(solid, bbox, False, False)
        bounds = np.array(bbox.Get()).reshape(2, 3) / 1000
        properties = GProp_GProps()
        BRepGProp.VolumeProperties_s(solid, properties)
        center = properties.CentreOfMass()
        volume = properties.Mass() / 1e9
        if volume <= 0:
            raise ValueError("CAD solid must have a positive enclosed volume")
        entity = reader.WS().TransferReader().EntityFromShapeResult(solid, 1)
        label = entity.Name().ToCString() if hasattr(entity, "Name") else ""
        vertices, triangles = [], []
        faces = TopExp_Explorer(solid, TopAbs_FACE)
        while faces.More():
            face = TopoDS.Face_s(faces.Current())
            location = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation_s(face, location)
            offset = len(vertices)
            for i in range(1, triangulation.NbNodes() + 1):
                point = triangulation.Node(i).Transformed(location.Transformation())
                vertices.append([point.X() / 1000, point.Y() / 1000, point.Z() / 1000])
            for i in range(1, triangulation.NbTriangles() + 1):
                indices = [j - 1 + offset for j in triangulation.Triangle(i).Get()]
                if face.Orientation() == TopAbs_REVERSED:
                    indices.reverse()
                triangles.append(indices)
            faces.Next()
        name = f"solid-{len(solids) + 1:02d}"
        obj = "# Metres; original STEP coordinates and solid location retained.\n"
        obj += "".join("v " + " ".join(f"{v:.12g}" for v in p) + "\n" for p in vertices)
        obj += "".join("f " + " ".join(str(i + 1) for i in t) + "\n" for t in triangles)
        mesh = obj.encode("ascii")
        (destination / f"{name}.obj").write_bytes(mesh)
        solids.append(
            {
                "name": name,
                "step_name": label,
                "file": f"{name}.obj",
                "sha256": sha256(mesh),
                "bounds_m": bounds.tolist(),
                "volume_m3": volume,
                "uniform_density_centroid_m": [center.X() / 1000, center.Y() / 1000, center.Z() / 1000],
                "vertices": len(vertices),
                "triangles": len(triangles),
            }
        )
        explorer.Next()
    if not solids:
        raise ValueError("STEP file contains no solids")
    bounds = np.array([item["bounds_m"] for item in solids])
    combined = np.array([bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)])
    volumes = np.array([item["volume_m3"] for item in solids])
    centers = np.array([item["uniform_density_centroid_m"] for item in solids])
    manifest = {
        "schema_version": 1,
        "source_file": "chassis.step",
        "source_sha256": sha256(content),
        "source_length_units": units,
        "import_length_unit": "millimetre",
        "mesh_length_unit": "metre",
        "converter": "cadquery-ocp-novtk " + importlib.metadata.version("cadquery-ocp-novtk"),
        "linear_deflection_mm": deflection_mm,
        "angular_deflection_rad": 0.15,
        "bounds_m": combined.tolist(),
        "dimensions_m": (combined[1] - combined[0]).tolist(),
        "total_volume_m3": float(volumes.sum()),
        "uniform_density_centroid_m": np.average(centers, axis=0, weights=volumes).tolist(),
        "mass_kg": None,
        "hardware_alignment": "unconfirmed",
        "solids": solids,
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def load_bundle(directory: Path) -> tuple[dict, dict[str, bytes]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    meshes = {}
    for solid in manifest["solids"]:
        data = (directory / solid["file"]).read_bytes()
        if sha256(data) != solid["sha256"]:
            raise ValueError("CAD mesh does not match its conversion manifest")
        meshes[solid["file"]] = data
    return manifest, meshes


def visual_bundle(visual: CadVisual) -> tuple[dict, dict[str, bytes]]:
    manifest, meshes = load_bundle(ASSET_ROOT / visual.asset)
    if manifest["source_sha256"] != visual.source_sha256:
        raise ValueError("CAD asset differs from the configured STEP source")
    return manifest, {f"cad/{name}": data for name, data in meshes.items()}


def read_obj(data: bytes) -> tuple[list, list]:
    vertices, triangles = [], []
    for line in data.decode("ascii").splitlines():
        values = line.split()
        if values and values[0] == "v":
            vertices.append([float(v) for v in values[1:4]])
        elif values and values[0] == "f":
            triangles.append([int(v) - 1 for v in values[1:4]])
    return vertices, triangles


def inspection_html(directory: Path, config: RobotConfig | None = None) -> str:
    """A self-contained rotatable view of the source CAD frame, with no hardware guesses."""
    manifest, meshes = load_bundle(directory)
    parts = []
    for solid in manifest["solids"]:
        vertices, triangles = read_obj(meshes[solid["file"]])
        parts.append({"name": solid["step_name"], "vertices": vertices, "triangles": triangles})
    frame = None
    if config is not None and config.cad_visual is not None:
        import mujoco

        visual = config.cad_visual
        if visual.source_sha256 != manifest["source_sha256"]:
            raise ValueError("Configured CAD frame belongs to a different STEP asset")
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, np.array(visual.quaternion_wxyz))
        frame = {"alignment": visual.alignment}
        for label, vector in zip(("forward", "left", "up"), matrix.reshape(3, 3)):
            axis = int(np.argmax(np.abs(vector)))
            if np.isclose(abs(vector[axis]), 1):
                frame[label] = ("+" if vector[axis] > 0 else "−") + "XYZ"[axis]
            else:
                frame[label] = str(np.round(vector, 3).tolist())
    payload = json.dumps(
        {"manifest": manifest, "parts": parts, "frame": frame}, separators=(",", ":")
    )
    encoded = base64.b64encode(gzip.compress(payload.encode(), mtime=0)).decode("ascii")
    template = (Path(__file__).parent / "cad_view.html").read_text()
    return template.replace("__CAD_DATA__", encoded)
